"""单样本推理脚本入口。"""
import torch
import torch.nn.functional as F

def infer_mlm_iterative(model, image, tokenizer, max_len=256, max_iter=5, device="cuda"):
    """
    MLM 架构的 Mask-Predict 并行推理。
    """
    model.eval()
    
    pad_id = tokenizer.token_to_id("[PAD]")
    mask_id = tokenizer.token_to_id("[MASK]")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    
    # 1. 全掩码初始化
    token_ids = torch.full((1, max_len), pad_id, dtype=torch.long, device=device)
    token_ids[0, 0] = bos_id
    token_ids[0, 1:] = mask_id 
    
    with torch.no_grad():
        # 视觉特征在迭代中不变，提前缓存避免重复执行 ConvNeXt-V2
        memory = model.encode(image)
        for step in range(max_iter):
            # 仅做解码迭代，is_causal=False 以启用双向注意力
            logits = model.decode(memory=memory, tgt_seq=token_ids, is_causal=False)
            
            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
            # 更新当前为 MASK 的位置
            is_mask = (token_ids == mask_id)
            token_ids[is_mask] = preds[is_mask]
            
            if step == max_iter - 1:
                break
                
            # 2. 计算本轮需要重新掩盖的 Token 比例 (线性退火)
            mask_ratio = 1.0 - ((step + 1) / max_iter)
            num_mask = int((max_len - 1) * mask_ratio)
            
            if num_mask == 0:
                break
                
            # 3. 寻找置信度最低的位置并重掩盖
            valid_positions = (token_ids != bos_id) & (token_ids != pad_id)
            # 将不应被掩盖的位置置信度设为无穷大
            valid_probs = max_probs.masked_fill(~valid_positions, float('inf'))
            
            # 取置信度最低的 num_mask 个位置
            _, least_confident_indices = torch.topk(valid_probs, num_mask, dim=-1, largest=False)
            
            # 重新打上 [MASK]
            token_ids[0, least_confident_indices[0]] = mask_id

    # 4. 序列截断与解码
    output_ids = token_ids[0].cpu().tolist()
    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]
        
    return tokenizer.decode(output_ids, skip_special_tokens=True)