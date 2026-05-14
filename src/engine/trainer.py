"""
src/engine/trainer.py
训练循环、AMP 与优化器控制（AR）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from typing import Optional

from src.models.sigreg_loss import SIGRegLikeLoss

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
        bow_weight=0.5,
        ctc_weight: Optional[float] = None,
        ohem_ratio=0.5,
        ema_model=None,
        enable_sigreg: bool = False,
        sigreg_weight: float = 0.1,
        sigreg_num_projections: int = 1024,
        sigreg_collapse_threshold: float = 0.01,
        decoder_input_noise_ratio: float = 0.08,
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
            decoder_input_noise_ratio: 解码器输入随机 Token 替换概率
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
        # 兼容旧参数: 若显式传入 ctc_weight，则映射到 bow_weight
        if ctc_weight is not None:
            bow_weight = ctc_weight
        self.bow_weight = float(bow_weight)
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
        
        # 解码器输入噪声参数
        self.decoder_input_noise_ratio = float(decoder_input_noise_ratio)
        if not (0.0 <= self.decoder_input_noise_ratio <= 1.0):
            raise ValueError(f"decoder_input_noise_ratio 必须在 [0,1] 区间，当前为 {self.decoder_input_noise_ratio}")
        
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

    def _optimizer_step(self) -> bool:
        if self.scaler is not None:
            old_scale = float(self.scaler.get_scale())
            self.scaler.unscale_(self.optimizer)
            
            # Gradient probe BEFORE clip
            try:
                enc_norm = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=float('inf')).item()
                dec_norm = torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=float('inf')).item()
                # tqdm.write(f"Grad Norm Probes -> Enc: {enc_norm:.3f}, Dec: {dec_norm:.3f}")
            except Exception:
                pass
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = float(self.scaler.get_scale())
            return new_scale >= old_scale
        else:
            # Gradient probe BEFORE clip
            try:
                enc_norm = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), max_norm=float('inf')).item()
                dec_norm = torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=float('inf')).item()
            except Exception:
                pass
                
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
                batch_bow_sum = 0.0
                batch_sigreg_sum = 0.0
                batch_acc_sum = 0.0

                for i in range(0, batch_size, chunk_size):
                    img_chunk = images[i : i + chunk_size]
                    ar_in_chunk = ar_inputs[i : i + chunk_size]
                    ar_tgt_chunk = ar_targets[i : i + chunk_size]

                    # ==========================================
                    # 注入 Decoder Input Noise (缓解曝光偏差)
                    # ==========================================
                    noise_ratio = self.decoder_input_noise_ratio

                    if noise_ratio > 0.0:
                        # 生成与 ar_in_chunk 同形状的随机概率矩阵
                        prob_mask = torch.rand_like(ar_in_chunk, dtype=torch.float) < noise_ratio

                        # 保护关键 Token：绝对不能对 padding 注入噪声
                        valid_mask = (ar_in_chunk != self.pad_id) & (ar_in_chunk != self.target_ignore_index)

                        # 只有在有效区域，且命中概率的，才进行替换
                        apply_mask = prob_mask & valid_mask

                        # 生成全局随机 Token (范围在 0 到 vocab_size-1 之间)
                        # 使用固定的 [MASK] Token (ID=4) 替换被选中的 Token
                        # 避免使用 pad_id，因为 pad_id 会被 tgt_key_padding_mask 完全从注意力矩阵中抹除
                        mask_id = getattr(self.model, "mask_id", 4)
                        
                        # 应用噪声：命中 apply_mask 的位置替换为 MASK Token
                        ar_in_chunk_noisy = torch.where(apply_mask, torch.tensor(mask_id, dtype=ar_in_chunk.dtype, device=ar_in_chunk.device), ar_in_chunk)
                    else:
                        ar_in_chunk_noisy = ar_in_chunk
                    # ==========================================

                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                        outputs = self.model(images=img_chunk, tgt_seq=ar_in_chunk_noisy, is_causal=True, return_aux=True)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                            bow_logits = outputs[1] if len(outputs) > 1 else None
                            sigreg_embedding = outputs[2] if len(outputs) > 2 else None
                        else:
                            logits = outputs
                            bow_logits = None
                            sigreg_embedding = None

                        vocab_size = logits.size(-1)
                        token_losses = self.criterion(logits.view(-1, vocab_size), ar_tgt_chunk.view(-1))
                        token_losses = token_losses.view(ar_tgt_chunk.size(0), -1)
                        # 直接求和并除以有效 Token 数，CrossEntropyLoss(ignore_index=...) 已经自动将无效位置设为 0
                        valid_counts = (ar_tgt_chunk != self.target_ignore_index).sum(dim=1).clamp(min=1)
                        per_sample_losses = token_losses.sum(dim=1) / valid_counts.to(dtype=token_losses.dtype)
                        keep_count = max(1, math.ceil(per_sample_losses.numel() * self.ohem_ratio))
                        top_losses = torch.topk(per_sample_losses, k=keep_count, largest=True).values
                        ce_loss = top_losses.mean()

                        bow_loss = logits.new_zeros(())
                        if bow_logits is not None:
                            vocab_size = bow_logits.size(-1)
                            sample_count = bow_logits.size(0)
                            bow_targets = torch.zeros((sample_count, vocab_size), device=self.device, dtype=bow_logits.dtype)

                            for b_idx in range(sample_count):
                                seq = ar_tgt_chunk[b_idx]
                                valid_tokens = seq[(seq != self.target_ignore_index) & (seq != self.pad_id)]
                                if self.eos_id is not None:
                                    valid_tokens = valid_tokens[valid_tokens != int(self.eos_id)]
                                valid_tokens = valid_tokens[(valid_tokens >= 0) & (valid_tokens < vocab_size)]
                                if valid_tokens.numel() > 0:
                                    bow_targets[b_idx].scatter_(0, valid_tokens.long(), 1.0)

                            bow_loss = F.binary_cross_entropy_with_logits(bow_logits, bow_targets)

                        # SIGReg-like 损失
                        sigreg_loss = logits.new_zeros(())
                        if self.enable_sigreg and sigreg_embedding is not None and self.sigreg_loss_fn is not None:
                            # 通过池化重建视觉 padding mask (True 表示图片有效区域)
                            with torch.no_grad():
                                stride = getattr(self.model.encoder, "downsample_stride", 32)
                                downsampled = F.max_pool2d(img_chunk, kernel_size=stride, stride=stride)
                                sigreg_mask = (downsampled.view(img_chunk.size(0), -1) > 1e-5)
                            
                            sigreg_loss = self.sigreg_loss_fn(sigreg_embedding, mask=sigreg_mask)

                        loss = (ce_loss + self.bow_weight * bow_loss + self.sigreg_weight * sigreg_loss) / chunks

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
                    batch_bow_sum += bow_loss.item()
                    batch_sigreg_sum += sigreg_loss.item() if self.enable_sigreg else 0.0
                    batch_acc_sum += chunk_acc * (img_chunk.size(0) / batch_size)

                optimizer_stepped = self._optimizer_step()
                if optimizer_stepped:
                    self.scheduler.step()
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)
                return batch_loss_sum, batch_ce_sum, batch_bow_sum, batch_sigreg_sum, batch_acc_sum, True

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
        return 0.0, 0.0, 0.0, 0.0, 0.0, False

    def train_epoch(self, dataloader, epoch, log_interval=50):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_bow = 0.0
        total_sigreg = 0.0
        total_acc = 0.0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            if getattr(self, "request_stop", False):
                tqdm.write("\n[WARNING] Received stop request (SIGTERM). Breaking epoch loop...")
                break

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

            batch_loss_sum, batch_ce_sum, batch_bow_sum, batch_sigreg_sum, batch_acc_sum, success = self._train_step_with_retry(images, ar_inputs, ar_targets)

            if not success:
                continue

            total_loss += batch_loss_sum
            total_ce += batch_ce_sum
            total_bow += batch_bow_sum
            total_sigreg += batch_sigreg_sum
            total_acc += batch_acc_sum
            valid_batches += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{batch_loss_sum:.3f}",
                        "CE": f"{batch_ce_sum:.3f}",
                        "BoW": f"{batch_bow_sum:.3f}",
                        "SIGReg": f"{batch_sigreg_sum:.3f}" if self.enable_sigreg else "N/A",
                        "Acc": f"{batch_acc_sum*100:.1f}%",
                        "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        avg_ce = total_ce / valid_batches if valid_batches > 0 else 0.0
        avg_bow = total_bow / valid_batches if valid_batches > 0 else 0.0
        avg_sigreg = total_sigreg / valid_batches if valid_batches > 0 else 0.0
        avg_acc = total_acc / valid_batches if valid_batches > 0 else 0.0
        interrupted = getattr(self, "request_stop", False)
        return avg_loss, avg_ce, avg_bow, avg_sigreg, avg_acc, interrupted