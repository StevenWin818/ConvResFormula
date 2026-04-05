"""
src/engine/trainer.py
训练循环、AMP、优化器控制与 MLM 专属监控。
"""
import torch
import torch.nn as nn
from tqdm import tqdm

class MLMTrainer:
    def __init__(self, model, optimizer, scheduler, device, scaler=None):
        """
        Args:
            model: LatexOCRModel 实例
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备 (cuda/cpu)
            scaler: torch.cuda.amp.GradScaler 实例 (用于混合精度训练)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        
        # ignore_index=-100 自动忽略不需要预测的 Token
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

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
            # 1. 数据迁移到 GPU
            images = batch["images"].to(self.device)
            masked_token_ids = batch["masked_token_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 2. 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # 注意：MLM 模式下，is_causal 必须为 False
                logits = self.model(images=images, tgt_seq=masked_token_ids, is_causal=False)
                
                # 展平以便计算 CrossEntropy (Batch * SeqLen, VocabSize)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                loss = self.criterion(logits_flat, labels_flat)
                
            # 3. 反向传播与优化
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # 梯度裁剪防爆炸 (Transformer 必备)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            self.scheduler.step()
            
            # 4. 统计与日志记录
            acc = self._calculate_masked_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc
            
            if batch_idx % log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "mask_acc": f"{acc*100:.2f}%",
                    "lr": f"{current_lr:.2e}"
                })
                
        return total_loss / len(dataloader), total_acc / len(dataloader)