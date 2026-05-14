"""自回归推理脚本入口。"""

import torch


@torch.no_grad()
def batched_infer_ar(model, images, tokenizer, max_len=160):
    """标准的自回归 (AR) 贪心生成算法"""
    device = images.device
    batch_size = images.size(0)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    # 1. 只跑一次视觉编码
    memory, memory_padding_mask = model.encode(images)

    # 2. 以 [BOS] 作为起始序列
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 3. 逐步解码下一个 token
    for _ in range(max_len - 1):
        logits = model.decode(memory, generated, memory_padding_mask, is_causal=True)
        next_tokens = logits[:, -1, :].argmax(dim=-1)

        next_tokens = next_tokens.masked_fill(finished, pad_id)
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        finished |= (next_tokens == eos_id)

        if finished.all():
            break

    return generated


@torch.no_grad()
def infer_ar(model, image, tokenizer, max_len=160, device="cuda"):
    """单样本 AR 推理包装函数"""
    model.eval()
    eos_id = tokenizer.token_to_id("[EOS]")
    image = image.to(device)
    generated = batched_infer_ar(model=model, images=image, tokenizer=tokenizer, max_len=max_len)

    output_ids = generated[0].cpu().tolist()
    if eos_id in output_ids:
        output_ids = output_ids[: output_ids.index(eos_id)]
    return tokenizer.decode(output_ids, skip_special_tokens=True)