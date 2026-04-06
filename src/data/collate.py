"""
批处理组装与动态并行 Masking (MLM Collate)。
"""
import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class MLMCollate:
    def __init__(
        self,
        pad_token_id,
        mask_token_id,
        vocab_size,
        special_token_ids,
        bos_token_id=None,
        mask_prob=0.15,
        dynamic_mask_prob=False,
        dynamic_mask_min=0.1,
        dynamic_mask_max=1.0,
        shape_quant_size=64,
        max_seq_len=256,
        area_limit=98304 * 1.5,
    ):
        """
        初始化 MLM 动态批处理。
        
        Args:
            pad_token_id (int): [PAD] 对应的 Token ID
            mask_token_id (int): [MASK] 对应的 Token ID
            vocab_size (int): 词表大小，用于随机词替换
            special_token_ids (set or list): 特殊 Token 的 ID 集合 (如 PAD, UNK, BOS, EOS, MASK)
            bos_token_id (int, optional): [BOS] 对应的 Token ID，用于保护句首 token 不被掩盖
            mask_prob (float): 掩盖概率，默认 15% (0.15)
            dynamic_mask_prob (bool): 是否按 batch 随机采样掩码概率
            dynamic_mask_min (float): 动态掩码下限
            dynamic_mask_max (float): 动态掩码上限
            shape_quant_size (int): 图像 padding 形状量化步长，默认 64
            max_seq_len (int): 文本最大长度硬截断，默认 256
            area_limit (float): 批次图像面积保护上限
        """
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids)
        self.bos_token_id = bos_token_id
        self.mask_prob = mask_prob
        self.dynamic_mask_prob = bool(dynamic_mask_prob)
        self.dynamic_mask_min = float(dynamic_mask_min)
        self.dynamic_mask_max = float(dynamic_mask_max)
        self.shape_quant_size = int(shape_quant_size)
        self.max_seq_len = int(max_seq_len)
        self.area_limit = float(area_limit)

        if not (0.0 <= self.mask_prob <= 1.0):
            raise ValueError(f"mask_prob 必须在 [0,1]，当前为 {self.mask_prob}")
        if not (0.0 <= self.dynamic_mask_min <= 1.0 and 0.0 <= self.dynamic_mask_max <= 1.0):
            raise ValueError(
                f"dynamic_mask_min/max 必须在 [0,1]，当前为 {self.dynamic_mask_min}/{self.dynamic_mask_max}"
            )
        if self.dynamic_mask_min > self.dynamic_mask_max:
            raise ValueError(
                f"dynamic_mask_min 不能大于 dynamic_mask_max，当前为 {self.dynamic_mask_min}>{self.dynamic_mask_max}"
            )
        if self.shape_quant_size <= 0:
            raise ValueError(f"shape_quant_size 必须 > 0，当前为 {self.shape_quant_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len 必须 > 0，当前为 {self.max_seq_len}")
        if self.area_limit <= 0:
            raise ValueError(f"area_limit 必须 > 0，当前为 {self.area_limit}")

    def __call__(self, batch):
        """
        处理批量数据，执行动态 MLM 掩盖。
        
        期望的 batch 结构: List[Dict]
        - item["image"]: 图像 Tensor
        - item["input_ids"]: 文本 Token 序列 (List[int] 或 Tensor)
        """
        images = []
        input_ids_list = []
        
        for item in batch:
            images.append(item["image"])
            
            seq = item["input_ids"]
            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq, dtype=torch.long)

            if seq.numel() > self.max_seq_len:
                seq = seq[: self.max_seq_len]

            input_ids_list.append(seq)
            
        # 1. 组装图像并进行 2D Padding (支持动态分辨率)
        if len(images) > 0 and isinstance(images[0], torch.Tensor):
            max_h = max(int(img.shape[1]) for img in images)
            max_w = max(int(img.shape[2]) for img in images)

            # 形状粗粒度量化：减少 unique shape，避免 allocator thrashing。
            q = self.shape_quant_size
            max_h = ((max_h + q - 1) // q) * q
            max_w = ((max_w + q - 1) // q) * q

            area_limit = int(self.area_limit)
            if max_h * max_w > area_limit:
                # 当批次面积过大时，按比例缩放到 area_limit 内，避免极端慢批次。
                scale = math.sqrt(area_limit / float(max_h * max_w))
                target_h = max(q, ((int(max_h * scale) + q - 1) // q) * q)
                target_w = max(q, ((int(max_w * scale) + q - 1) // q) * q)

                resized_images = []
                for img in images:
                    _, h, w = img.shape
                    h_int = int(h)
                    w_int = int(w)
                    if h_int > target_h or w_int > target_w:
                        scale_hw = min(target_h / max(h_int, 1), target_w / max(w_int, 1))
                        new_h = max(q, ((int(h_int * scale_hw) + q - 1) // q) * q)
                        new_w = max(q, ((int(w_int * scale_hw) + q - 1) // q) * q)
                        new_h = min(new_h, target_h)
                        new_w = min(new_w, target_w)
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=(new_h, new_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    resized_images.append(img)

                images = resized_images
                max_h = max(int(img.shape[1]) for img in images)
                max_w = max(int(img.shape[2]) for img in images)
                max_h = ((max_h + q - 1) // q) * q
                max_w = ((max_w + q - 1) // q) * q

            batch_size = len(images)

            batched_images = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

            for i, img in enumerate(images):
                _, h, w = img.shape
                batched_images[i, :, :h, :w] = img

            images = batched_images
                
        # 2. 组装 Token 序列并填充 Padding
        clean_token_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)

        # 保留一份“干净序列”给未来 AR 分支，MLM 分支在副本上做掩盖
        masked_token_ids = clean_token_ids.clone()
        mlm_labels = clean_token_ids.clone()
        
        # 3. 动态 MLM 掩盖策略
        cur_mask_prob = self.mask_prob
        if self.dynamic_mask_prob:
            cur_mask_prob = float(torch.empty(1).uniform_(self.dynamic_mask_min, self.dynamic_mask_max).item())

        # 创建与输入形状一致的概率矩阵
        probability_matrix = torch.full(mlm_labels.shape, cur_mask_prob)
        
        # 🚀 修复：绝对不能保护 [EOS] 和 [PAD]！必须让模型学会预测它们！
        # 只保护 [BOS] (句首)
        if self.bos_token_id is not None:
            probability_matrix.masked_fill_(mlm_labels == self.bos_token_id, value=0.0)
        
        # 为了防止 [PAD] 占据过多的 Loss 权重导致模型变懒，我们可以略微降低 [PAD] 被选中的概率
        pad_mask = (mlm_labels == self.pad_token_id)
        # 将 [PAD] 位置的掩码概率强制降低到 0.05 (只让它稍微学一下怎么填 PAD 就够了)
        probability_matrix = torch.where(pad_mask, torch.full_like(probability_matrix, 0.05), probability_matrix)
            
        # 使用伯努利分布采样，生成 15% 的掩盖掩码 (True 表示选中该位置)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 【Label 忽略机制】没有被选中的 Token (即不需要模型进行预测的位置)，将其 Label 设为 -100
        mlm_labels[~masked_indices] = -100

        # 在选中的 15% 中：
        # 【80% 替换为 [MASK]】
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        masked_token_ids[indices_replaced] = self.mask_token_id

        # 【10% 替换为随机词】 (在剩下的 20% 里随机选一半，即 10%)
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.vocab_size, mlm_labels.shape, dtype=torch.long)
        masked_token_ids[indices_random] = random_words[indices_random]

        # 【10% 保持不变】 (已被 masked_indices 选中，但没有被替换，所以保持原始 token id 不变，且 label 不是 -100)
        
        return {
            "images": images,
            "clean_token_ids": clean_token_ids,
            "masked_token_ids": masked_token_ids,
            "mlm_labels": mlm_labels,
            # 保持向后兼容，旧训练器仍可读取 labels
            "labels": mlm_labels,
        }
     max_h = max(int(img.shape[1]) for img in images)
                max_w = max(int(img.shape[2]) for img in images)
                max_h = ((max_h + q - 1) // q) * q
                max_w = ((max_w + q - 1) // q) * q

            batch_size = len(images)

            batched_images = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

            for i, img in enumerate(images):
                _, h, w = img.shape
                batched_images[i, :, :h, :w] = img

            images = batched_images
                
        # 2. 组装 Token 序列并填充 Padding
        clean_token_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)

        # 保留一份“干净序列”给未来 AR 分支，MLM 分支在副本上做掩盖
        masked_token_ids = clean_token_ids.clone()
        mlm_labels = clean_token_ids.clone()
        
        # 3. 动态 MLM 掩盖策略
        cur_mask_prob = self.mask_prob
        if self.dynamic_mask_prob:
            cur_mask_prob = float(torch.empty(1).uniform_(self.dynamic_mask_min, self.dynamic_mask_max).item())

        # 创建与输入形状一致的概率矩阵
        probability_matrix = torch.full(mlm_labels.shape, cur_mask_prob)
        
        # 保护特殊 Token，不要将 [PAD], [BOS], [EOS] 等变成 [MASK]
        for special_id in self.special_token_ids:
            probability_matrix.masked_fill_(mlm_labels == special_id, value=0.0)
            
        # 使用伯努利分布采样，生成 15% 的掩盖掩码 (True 表示选中该位置)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 【Label 忽略机制】没有被选中的 Token (即不需要模型进行预测的位置)，将其 Label 设为 -100
        mlm_labels[~masked_indices] = -100

        # 在选中的 15% 中：
        # 【80% 替换为 [MASK]】
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        masked_token_ids[indices_replaced] = self.mask_token_id

        # 【10% 替换为随机词】 (在剩下的 20% 里随机选一半，即 10%)
        indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.vocab_size, mlm_labels.shape, dtype=torch.long)
        masked_token_ids[indices_random] = random_words[indices_random]

        # 【10% 保持不变】 (已被 masked_indices 选中，但没有被替换，所以保持原始 token id 不变，且 label 不是 -100)
        
        return {
            "images": images,
            "clean_token_ids": clean_token_ids,
            "masked_token_ids": masked_token_ids,
            "mlm_labels": mlm_labels,
            # 保持向后兼容，旧训练器仍可读取 labels
            "labels": mlm_labels,
        }
