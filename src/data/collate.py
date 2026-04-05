"""
批处理组装与动态并行 Masking (MLM Collate)。
"""
import torch
from torch.nn.utils.rnn import pad_sequence

class MLMCollate:
    def __init__(self, pad_token_id, mask_token_id, vocab_size, special_token_ids, mask_prob=0.15):
        """
        初始化 MLM 动态批处理。
        
        Args:
            pad_token_id (int): [PAD] 对应的 Token ID
            mask_token_id (int): [MASK] 对应的 Token ID
            vocab_size (int): 词表大小，用于随机词替换
            special_token_ids (set or list): 特殊 Token 的 ID 集合 (如 PAD, UNK, BOS, EOS, MASK)，这些不会被掩盖
            mask_prob (float): 掩盖概率，默认 15% (0.15)
        """
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids)
        self.mask_prob = mask_prob

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
            input_ids_list.append(seq)
            
        # 1. 组装图像并进行 2D Padding (支持动态分辨率)
        if len(images) > 0 and isinstance(images[0], torch.Tensor):
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            batch_size = len(images)

            batched_images = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

            for i, img in enumerate(images):
                _, h, w = img.shape
                batched_images[i, :, :h, :w] = img

            images = batched_images
                
        # 2. 组装 Token 序列并填充 Padding
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        
        # 默认 Labels 和输入完全一致
        labels = input_ids.clone()
        
        # 3. 动态 MLM 掩盖策略
        # 创建与输入形状一致的概率矩阵
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        # (重要) 保护特殊 Token，不要将 [PAD], [BOS], [EOS] 等变成 [MASK]
        for special_id in self.special_token_ids:
            probability_matrix.masked_fill_(labels == special_id, value=0.0)
            
        # 使用伯努利分布采样，生成 15% 的掩盖掩码 (True 表示选中该位置)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 【Label 忽略机制】没有被选中的 Token (即不需要模型进行预测的位置)，将其 Label 设为 -100
        labels[~masked_indices] = -100

        # 在选中的 15% 中：
        # 【80% 替换为 [MASK]】
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 【10% 替换为随机词】 (在剩下的 20% 里随机选一半，即 10%)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 【10% 保持不变】 (已被 masked_indices 选中，但没有被替换，所以保持原始 token id 不变，且 label 不是 -100)
        
        return {
            "images": images,
            "masked_token_ids": input_ids,
            "labels": labels
        }
