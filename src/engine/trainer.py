"""
src/engine/trainer.py
训练循环、AMP 与优化器控制（AR）。
"""
import torch
import torch.nn as nn
import math
from tqdm import tqdm

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
        label_smoothing=0.0,
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
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = amp_dtype
        self.label_smoothing = float(label_smoothing)
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing 必须在 [0,1) 区间，当前为 {self.label_smoothing}")
        
        self.pad_id = int(getattr(model, "pad_id", 0))
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, label_smoothing=self.label_smoothing)

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def train_epoch(self, dataloader, epoch, log_interval=50):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device, non_blocking=True)
            clean_token_ids = batch["clean_token_ids"].to(self.device, non_blocking=True)

            # AR Teacher Forcing：输入去掉最后一个 token，目标右移一位
            ar_inputs = clean_token_ids[:, :-1]
            ar_targets = clean_token_ids[:, 1:].contiguous()

            batch_size = images.size(0)
            success = False
            chunks = 1

            while (not success) and chunks <= batch_size:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    chunk_size = math.ceil(batch_size / chunks)
                    batch_loss_sum = 0.0
                    batch_acc_sum = 0.0

                    for i in range(0, batch_size, chunk_size):
                        img_chunk = images[i : i + chunk_size]
                        ar_in_chunk = ar_inputs[i : i + chunk_size]
                        ar_tgt_chunk = ar_targets[i : i + chunk_size]

                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                            logits = self.model(images=img_chunk, tgt_seq=ar_in_chunk, is_causal=True)
                            loss = self.criterion(logits.view(-1, logits.size(-1)), ar_tgt_chunk.view(-1)) / chunks

                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        preds = logits.argmax(dim=-1)
                        valid_mask = ar_tgt_chunk != self.pad_id
                        if valid_mask.any():
                            chunk_acc = (preds[valid_mask] == ar_tgt_chunk[valid_mask]).float().mean().item()
                        else:
                            chunk_acc = 0.0

                        batch_loss_sum += loss.item() * chunks
                        batch_acc_sum += chunk_acc * (img_chunk.size(0) / batch_size)

                    self._optimizer_step()
                    self.scheduler.step()
                    success = True

                except torch.cuda.OutOfMemoryError:
                    if 'logits' in locals():
                        del logits
                    if 'loss' in locals():
                        del loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    chunks *= 2

            if not success:
                continue

            total_loss += batch_loss_sum
            total_acc += batch_acc_sum
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({"Loss": f"{batch_loss_sum:.3f}", "Acc": f"{batch_acc_sum*100:.1f}%"})
                
        return total_loss / len(dataloader), total_acc / len(dataloader)