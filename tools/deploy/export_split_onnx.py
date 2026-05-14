# python export_split_onnx.py --checkpoint checkpoints/model_epoch_33.pth --vocab-cache cache/vocab.pkl --output-dir checkpoints --quantize
import argparse
import os
import pickle
import sys
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 引用你的模型定义
from src.models.decoder import UnifiedSeq2Seq as Seq2Seq

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2

# ==========================================
# 1. 辅助函数
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CROHME Seq2Seq model to Split ONNX")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "model_epoch_42.pth"),
        help="Path to the .pth checkpoint.",
    )
    parser.add_argument(
        "--vocab-cache",
        default=os.path.join(PROJECT_ROOT, "cache", "vocab.pkl"),
        help="Path to the cached vocab pickle.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Directory to save encoder.onnx and decoder.onnx.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=1200) 
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", help="同时导出 INT8 量化模型")
    return parser.parse_args()

def load_vocab_size(cache_path: str) -> int:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Vocab cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    char2id = cache.get("char2id")
    return len(char2id)

def build_model(vocab_size: int, hidden_dim: int, num_layers: int, dropout: float, device: torch.device) -> Seq2Seq:
    model = Seq2Seq(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=PAD_ID,
        sos_idx=SOS_ID,
        eos_idx=EOS_ID,
        dropout=dropout,
    )
    model.to(device)
    model.eval()
    return model

def load_checkpoint(model: Seq2Seq, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"🔄 Loading weights from {checkpoint_path} ...")
    state = torch.load(checkpoint_path, map_location=device)
    payload = state.get("model", state)
    model.load_state_dict(payload)

# ==========================================
# 2. 定义 ONNX 包装器
# ==========================================

class EncoderExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, features, feature_lengths):
        encoder_outputs, hidden, enc_lengths = self.encoder(features, feature_lengths)
        # 🔥 修改：把 enc_lengths 也返回，防止 feature_lengths 被优化掉
        return encoder_outputs, hidden, enc_lengths

class DecoderStepExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
    
    def forward(self, input_token, hidden, encoder_outputs, enc_mask):
        logits, next_hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs, enc_mask)
        return logits, next_hidden, attn_weights

# ==========================================
# 3. 主逻辑
# ==========================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    vocab_size = load_vocab_size(args.vocab_cache)
    print(f"📚 Vocab size: {vocab_size}")
    model = build_model(vocab_size, args.hidden_dim, args.num_layers, args.dropout, device)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)  # 防御：确保参数在目标设备

    os.makedirs(args.output_dir, exist_ok=True)
    enc_path = os.path.join(args.output_dir, "encoder.onnx")
    dec_path = os.path.join(args.output_dir, "decoder.onnx")
    enc_q_path = os.path.join(args.output_dir, "encoder_int8.onnx")
    dec_q_path = os.path.join(args.output_dir, "decoder_int8.onnx")

    # -------------------------------------------------------
    # 导出 Encoder
    # -------------------------------------------------------
    print(f"\n🛠️  Exporting Encoder to {enc_path} ...")
    encoder_wrapper = EncoderExport(model)
    
    dummy_feat = torch.randn(1, args.max_points, 8, device=device)
    dummy_feat_len = torch.tensor([args.max_points], dtype=torch.long, device=device)
    
    torch.onnx.export(
        encoder_wrapper,
        (dummy_feat, dummy_feat_len),
        enc_path,
        input_names=["features", "feature_lengths"],
        # 🔥 修改：output_names 增加 "enc_lengths"
        output_names=["encoder_outputs", "hidden", "enc_lengths"], 
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "feature_lengths": {0: "batch"},
            "encoder_outputs": {0: "batch", 1: "time"},
            "hidden": {1: "batch"}, 
            "enc_lengths": {0: "batch"} # 新增动态轴
        },
        opset_version=args.opset,
        dynamo=False 
    )
    print("✅ Encoder Exported!")

    if args.quantize:
        print(f"⚙️  Quantizing Encoder -> {enc_q_path} ...")
        # 避免 ConvInteger 兼容性问题，仅量化 MatMul/Gemm（保持卷积为 FP32）
        quantize_dynamic(enc_path, enc_q_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        print("✅ Encoder INT8 Exported!")

    # -------------------------------------------------------
    # 导出 Decoder (Step)
    # -------------------------------------------------------
    print(f"\n🛠️  Exporting Decoder Step to {dec_path} ...")
    decoder_wrapper = DecoderStepExport(model)

    with torch.no_grad():
        dummy_enc_out, dummy_hidden, _ = encoder_wrapper(dummy_feat, dummy_feat_len)
    # 获取真实的下采样时间长度 (例如 150)
    downsampled_time = dummy_enc_out.shape[1]
    print(f"ℹ️ Encoder downsampling: {args.max_points} -> {downsampled_time}")

    dummy_input_token = torch.tensor([1], dtype=torch.long, device=device) 
    
    # 🔥【修改点】去掉中间的维度，变为 [1, Time]
    dummy_enc_mask = torch.ones((1, downsampled_time), dtype=torch.bool, device=device)

    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_token, dummy_hidden, dummy_enc_out, dummy_enc_mask),
        dec_path,
        input_names=["input_token", "hidden", "encoder_outputs", "enc_mask"],
        output_names=["logits", "next_hidden", "attn_weights"],
        
        dynamic_axes={
            "input_token": {0: "batch"},
            "hidden": {1: "batch"},
            "encoder_outputs": {0: "batch", 1: "time"},
            "enc_mask": {0: "batch", 1: "time"},
            "logits": {0: "batch"},
            "next_hidden": {1: "batch"},
            
            # 注意力权重: [B, T]
            "attn_weights": {0: "batch", 1: "time"} 
        },
        opset_version=args.opset,
        dynamo=False
    )
    print("✅ Decoder Exported!")

    if args.quantize:
        print(f"⚙️  Quantizing Decoder -> {dec_q_path} ...")
        quantize_dynamic(dec_path, dec_q_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        print("✅ Decoder INT8 Exported!")
    print("\n🎉 All Done!")

if __name__ == "__main__":
    main()