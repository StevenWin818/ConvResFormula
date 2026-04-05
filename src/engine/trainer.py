"""
src/engine/trainer.py
训练循环、AMP、优化器控制与 MLM 专属监控。
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple

class MLMTrainer:
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
        
        # ignore_index=-100 自动忽略不需要预测的 Token
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)

    def _train_step_single(self, images, masked_token_ids, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            logits = self.model(images=images, tgt_seq=masked_token_ids, is_causal=False)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = self.criterion(logits_flat, labels_flat)
        return loss, logits

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def _calculate_masked_accuracy(self, logits, labels):
        """计算掩码位置的预测准确率"""
        preds = torch.argmax(logits, dim=-1)
        # 找出真正被 Mask 或替换，且需要预测的位置 (label != -100)
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        
        correct = (preds[mask] == labels[mask]).float().sum()
        total = mask.sum().float()
        return (correct / total).item()

    def train_epoch(self, dataloader, epoch, log_interval=50):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # 1. 数据迁移到训练设备
            images = batch["images"].to(self.device, non_blocking=True)
            masked_token_ids = batch["masked_token_ids"].to(self.device, non_blocking=True)
            labels = batch.get("mlm_labels", batch["labels"]).to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)

            loss, logits = self._train_step_single(images, masked_token_ids, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self._optimizer_step()
            self.scheduler.step()

            loss_val = float(loss.item())
            acc = self._calculate_masked_accuracy(logits, labels)

            # 4. 统计与日志记录
            total_loss += loss_val
            total_acc += acc
            
            if batch_idx % log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                postfix = {
                    "loss": f"{loss_val:.4f}",
                    "mask_acc": f"{acc*100:.2f}%",
                    "lr": f"{current_lr:.2e}",
                }
                pbar.set_postfix(postfix)
                
        return total_loss / len(dataloader), total_acc / len(dataloader)