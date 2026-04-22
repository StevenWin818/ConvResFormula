"""
src/engine/trainer.py
训练循环、AMP 与优化器控制（AR）。
"""
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from typing import Optional

# 导入 SIGReg-like 损失
try:
    from ..models.sigreg_loss import SIGRegLikeLoss
except (ImportError, ValueError):
    # 如果相对导入失败，尝试直接导入
    import sys
    from pathlib import Path
    sigreg_path = Path(__file__).parent.parent / "models"
    sys.path.insert(0, str(sigreg_path))
    from sigreg_loss import SIGRegLikeLoss

class ARTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        scaler=None,
        amp_enabled=False,
        amp_dtype=torch.float16,
        channels_last=False,
        label_smoothing=0.0,
        target_ignore_index=-100,
        ctc_weight=0.5,
        ohem_ratio=0.5,
        ema_model=None,
        enable_sigreg: bool = False,
        sigreg_weight: float = 0.1,
        sigreg_num_projections: int = 1024,
        sigreg_collapse_threshold: float = 0.01,
    ):
        """
        Args:
            model: LatexOCRModel 实例
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备 (cuda/cpu)
            scaler: torch.amp.GradScaler 实例 (仅 fp16 需要)
            amp_enabled: 是否启用 autocast
            amp_dtype: autocast 使用的精度类型
            label_smoothing: 交叉熵标签平滑系数
            enable_sigreg: 是否启用 SIGReg-like 辅助损失
            sigreg_weight: SIGReg 损失权重
            sigreg_num_projections: SIGReg 随机投影次数
            sigreg_collapse_threshold: embedding 坍塌检测阈值
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = amp_dtype
        self.channels_last = bool(channels_last)
        self.label_smoothing = float(label_smoothing)
        self.ctc_weight = float(ctc_weight)
        self.ohem_ratio = float(ohem_ratio)
        self.ema_model = ema_model
        
        # SIGReg-like 参数
        self.enable_sigreg = bool(enable_sigreg)
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_loss_fn: Optional[SIGRegLikeLoss] = None
        if self.enable_sigreg and self.sigreg_weight > 0:
            self.sigreg_loss_fn = SIGRegLikeLoss(
                num_projections=int(sigreg_num_projections),
                collapse_threshold=float(sigreg_collapse_threshold),
                seed=None,
            ).to(device)
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing 必须在 [0,1) 区间，当前为 {self.label_smoothing}")
        if not (0.0 < self.ohem_ratio <= 1.0):
            raise ValueError(f"ohem_ratio 必须在 (0,1] 区间，当前为 {self.ohem_ratio}")
        
        self.pad_id = int(getattr(model, "pad_id", 0))
        self.eos_id = getattr(model, "eos_id", None)
        self.target_ignore_index = int(target_ignore_index)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.target_ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        self.ctc_blank_id = int(getattr(model, "ctc_blank_id", -1))
        self.ctc_criterion = None
        if self.ctc_blank_id >= 0:
            self.ctc_criterion = nn.CTCLoss(blank=self.ctc_blank_id, zero_infinity=True)

    def _compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        image_chunk: torch.Tensor,
        ar_targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.ctc_criterion is None:
            return ctc_logits.new_zeros(())

        time_steps, batch_size, _ = ctc_logits.size()
        with torch.no_grad():
            # 基于真实图像有效列估计 CTC 时间长度，减少对右侧纯 padding 区域的冗余 blank 拟合。
            import torch.nn.functional as F

            downsampled = F.max_pool2d(image_chunk, kernel_size=32, stride=32)
            valid_cols = (downsampled.squeeze(1) > 1e-5).any(dim=1)
            input_lengths = valid_cols.sum(dim=1).to(device=ctc_logits.device, dtype=torch.long)
            input_lengths = input_lengths.clamp(min=1, max=time_steps)

        target_chunks = []
        target_lengths_list = []
        for row_idx in range(batch_size):
            target_row = ar_targets[row_idx][valid_mask[row_idx]]
            if self.eos_id is not None:
                target_row = target_row[target_row != int(self.eos_id)]
            target_chunks.append(target_row)
            target_lengths_list.append(int(target_row.numel()))

        target_lengths = torch.tensor(target_lengths_list, device=ctc_logits.device, dtype=torch.long)
        if int(target_lengths.sum().item()) == 0:
            return ctc_logits.new_zeros(())

        targets = torch.cat([row for row in target_chunks if row.numel() > 0], dim=0).to(dtype=torch.long)
        # 强制使用 FP32 执行 CTC 的 log_softmax，避免 AMP(fp16) 下数值下溢导致 NaN。
        log_probs = ctc_logits.float().log_softmax(dim=-1)
        return self.ctc_criterion(log_probs, targets, input_lengths, target_lengths)

    def _optimizer_step(self) -> bool:
        if self.scaler is not None:
            old_scale = float(self.scaler.get_scale())
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = float(self.scaler.get_scale())
            return new_scale >= old_scale
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            return True
    def _train_step_with_retry(self, images, ar_inputs, ar_targets):
        """
        独立封装的动态微批次与 OOM 拦截引擎。
        返回: (batch_loss_sum, batch_acc_sum, success_flag)
        """
        batch_size = images.size(0)
        chunks = 1

        while chunks <= batch_size:
            try:
                self.optimizer.zero_grad(set_to_none=True)
                chunk_size = math.ceil(batch_size / chunks)

                batch_loss_sum = 0.0
                batch_ce_sum = 0.0
                batch_ctc_sum = 0.0
                batch_sigreg_sum = 0.0
                batch_acc_sum = 0.0

                for i in range(0, batch_size, chunk_size):
                    img_chunk = images[i : i + chunk_size]
                    ar_in_chunk = ar_inputs[i : i + chunk_size]
                    ar_tgt_chunk = ar_targets[i : i + chunk_size]

                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                        outputs = self.model(images=img_chunk, tgt_seq=ar_in_chunk, is_causal=True, return_aux=True)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                            ctc_logits = outputs[1] if len(outputs) > 1 else None
                            sigreg_embedding = outputs[2] if len(outputs) > 2 else None
                        else:
                            logits = outputs
                            ctc_logits = None
                            sigreg_embedding = None

                        vocab_size = logits.size(-1)
                        token_losses = self.criterion(logits.view(-1, vocab_size), ar_tgt_chunk.view(-1))
                        token_losses = token_losses.view(ar_tgt_chunk.size(0), -1)
                        valid_mask = ar_tgt_chunk != self.target_ignore_index
                        valid_counts = valid_mask.sum(dim=1).clamp(min=1).to(dtype=token_losses.dtype)

                        per_sample_losses = (token_losses * valid_mask.to(dtype=token_losses.dtype)).sum(dim=1) / valid_counts
                        keep_count = max(1, math.ceil(per_sample_losses.numel() * self.ohem_ratio))
                        top_losses = torch.topk(per_sample_losses, k=keep_count, largest=True).values
                        ce_loss = top_losses.mean()

                        ctc_loss = logits.new_zeros(())
                        if ctc_logits is not None:
                            ctc_loss = self._compute_ctc_loss(ctc_logits, img_chunk, ar_tgt_chunk, valid_mask)

                        # SIGReg-like 损失
                        sigreg_loss = logits.new_zeros(())
                        if self.enable_sigreg and sigreg_embedding is not None and self.sigreg_loss_fn is not None:
                            # sigreg_embedding: [chunk_size, seq_len, d_model]
                            # 构造对应的掩码（如果需要）
                            sigreg_mask = None  # 默认不使用掩码，使用全部 embedding
                            sigreg_loss = self.sigreg_loss_fn(sigreg_embedding, mask=sigreg_mask)

                        loss = (ce_loss + self.ctc_weight * ctc_loss + self.sigreg_weight * sigreg_loss) / chunks

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    preds = logits.argmax(dim=-1)
                    if valid_mask.any():
                        chunk_acc = (preds[valid_mask] == ar_tgt_chunk[valid_mask]).float().mean().item()
                    else:
                        chunk_acc = 0.0

                    batch_loss_sum += loss.item() * chunks
                    batch_ce_sum += ce_loss.item()
                    batch_ctc_sum += ctc_loss.item()
                    batch_sigreg_sum += sigreg_loss.item() if self.enable_sigreg else 0.0
                    batch_acc_sum += chunk_acc * (img_chunk.size(0) / batch_size)

                optimizer_stepped = self._optimizer_step()
                if optimizer_stepped:
                    self.scheduler.step()
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)
                return batch_loss_sum, batch_ce_sum, batch_ctc_sum, batch_sigreg_sum, batch_acc_sum, True

            except torch.cuda.OutOfMemoryError:
                try:
                    del logits
                except NameError:
                    pass
                try:
                    del loss
                except NameError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunks *= 2
                if chunks <= batch_size:
                    tqdm.write(f"OOM 拦截: 当前 Batch 自动切分为 {chunks} 份微批次...")

        tqdm.write("极端案例: 该 Batch 拆分后仍 OOM，已跳过。")
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0, 0.0, 0.0, 0.0, False

    def train_epoch(self, dataloader, epoch, log_interval=50):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_ctc = 0.0
        total_sigreg = 0.0
        total_acc = 0.0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device, non_blocking=True)
            if self.channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)

            if "decoder_inputs" in batch and "labels" in batch:
                ar_inputs = batch["decoder_inputs"].to(self.device, non_blocking=True)
                ar_targets = batch["labels"].to(self.device, non_blocking=True).contiguous()
            else:
                clean_token_ids = batch["clean_token_ids"].to(self.device, non_blocking=True)
                ar_inputs = clean_token_ids[:, :-1]
                ar_targets = clean_token_ids[:, 1:].contiguous()
                if self.target_ignore_index != self.pad_id:
                    ar_targets = ar_targets.masked_fill(ar_targets == self.pad_id, self.target_ignore_index)

            batch_loss_sum, batch_ce_sum, batch_ctc_sum, batch_sigreg_sum, batch_acc_sum, success = self._train_step_with_retry(images, ar_inputs, ar_targets)

            if not success:
                continue

            total_loss += batch_loss_sum
            total_ce += batch_ce_sum
            total_ctc += batch_ctc_sum
            total_sigreg += batch_sigreg_sum
            total_acc += batch_acc_sum
            valid_batches += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{batch_loss_sum:.3f}",
                        "CE": f"{batch_ce_sum:.3f}",
                        "CTC": f"{batch_ctc_sum:.3f}",
                        "SIGReg": f"{batch_sigreg_sum:.3f}" if self.enable_sigreg else "N/A",
                        "Acc": f"{batch_acc_sum*100:.1f}%",
                        "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        avg_ce = total_ce / valid_batches if valid_batches > 0 else 0.0
        avg_ctc = total_ctc / valid_batches if valid_batches > 0 else 0.0
        avg_sigreg = total_sigreg / valid_batches if valid_batches > 0 else 0.0
        avg_acc = total_acc / valid_batches if valid_batches > 0 else 0.0
        return avg_loss, avg_ce, avg_ctc, avg_sigreg, avg_acc