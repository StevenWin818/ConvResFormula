# python onnx_infer.py --quant_mode int8
import argparse
import os
import random
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_1d import parse_inkml

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2

# Beam Search & 长度惩罚默认参数（与 predict 保持一致的入口形式）
BEAM_WIDTH_DEFAULT = 6
REP_PENALTY_DEFAULT = 1.1
NO_REPEAT_NGRAM_DEFAULT = 5
MAX_TOKEN_REPEAT_DEFAULT = 10
EOS_BONUS_DEFAULT = 0.05
MIN_DECODE_LEN_DEFAULT = 10 
BRACKET_PENALTY_DEFAULT = 0.7


def resolve_providers(use_quant: bool) -> List[str]:
    """选择最合适的 ORT Provider 顺序，量化优先 DML/DNNL/QNN。"""
    available = ort.get_available_providers()
    ordered: List[str] = []
    if use_quant:
        for candidate in ("DmlExecutionProvider", "DnnlExecutionProvider", "QNNExecutionProvider"):
            if candidate in available:
                ordered.append(candidate)
    if "CPUExecutionProvider" in available:
        ordered.append("CPUExecutionProvider")
    if not ordered:
        ordered = available or ["CPUExecutionProvider"]
    return ordered


def get_dynamic_alpha(feat_len_int: int) -> float:
    """根据有效点数自动计算 alpha，策略与 predict.py 一致。"""
    MIN_PTS, MAX_PTS = 100, 800
    MIN_ALPHA, MAX_ALPHA = 0.9, 1.15
    if feat_len_int <= MIN_PTS:
        return MIN_ALPHA
    if feat_len_int >= MAX_PTS:
        return MAX_ALPHA
    ratio = (feat_len_int - MIN_PTS) / (MAX_PTS - MIN_PTS)
    return MIN_ALPHA + ratio * (MAX_ALPHA - MIN_ALPHA)


def load_vocab(cache_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    char2id = cache["char2id"]
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char


def strokes_to_features(strokes: List[np.ndarray], max_points: int = 1200) -> Tuple[np.ndarray, int]:
    """与训练端一致的 8D 特征: [x, y, dx, dy, pen_up, sin_θ, cos_θ, pen_down]"""
    points = []
    for stroke in strokes:
        if stroke.size == 0:
            continue
        for i, (x, y) in enumerate(stroke):
            pen_up = 1.0 if i == len(stroke) - 1 else 0.0
            pen_down = 1.0 if i == 0 else 0.0
            if i == 0:
                dx, dy = 0.0, 0.0
            else:
                prev_x, prev_y = stroke[i - 1]
                dx, dy = x - prev_x, y - prev_y
            speed = (dx * dx + dy * dy) ** 0.5
            if speed > 1e-8:
                sin_t, cos_t = dy / speed, dx / speed
            else:
                sin_t, cos_t = 0.0, 0.0
            points.append([x, y, dx, dy, pen_up, sin_t, cos_t, pen_down])

    if not points:
        return np.zeros((1, max_points, 8), dtype=np.float32), 0

    arr = np.array(points, dtype=np.float32)
    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
    x_max, y_max = arr[:, 0].max(), arr[:, 1].max()
    scale = 1.0 / max(x_max - x_min, y_max - y_min, 1e-6)
    arr[:, 0] = (arr[:, 0] - x_min) * scale
    arr[:, 1] = (arr[:, 1] - y_min) * scale
    arr[:, 2] *= scale
    arr[:, 3] *= scale

    length = min(arr.shape[0], max_points)
    padded = np.zeros((max_points, 8), dtype=np.float32)
    padded[:length] = arr[:length]
    return padded[np.newaxis, ...], length


def tokens_to_latex(tokens: List[int], id2char: Dict[int, str]) -> str:
    pieces = []
    for tid in tokens:
        if tid in (PAD_ID, SOS_ID):
            continue
        if tid == EOS_ID:
            break
        pieces.append(id2char.get(tid, ""))
    return "".join(pieces)


def sanitize_latex(latex: str) -> str:
    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {v: k for k, v in opens.items()}
    stack = []
    out = []
    for ch in latex:
        if ch in opens:
            stack.append(ch)
            out.append(ch)
        elif ch in closes:
            if stack and stack[-1] == closes[ch]:
                stack.pop()
                out.append(ch)
            else:
                continue
        else:
            out.append(ch)
    if not stack:
        return "".join(out)
    out_rev = []
    for ch in reversed(out):
        if stack and ch == stack[-1]:
            stack.pop()
            continue
        out_rev.append(ch)
    return "".join(reversed(out_rev))


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    # 数值稳定的 log_softmax
    m = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - m
    logsumexp = np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
    return stabilized - logsumexp


def _beam_search_decode(
    sess_enc: ort.InferenceSession,
    sess_dec: ort.InferenceSession,
    feat: np.ndarray,
    feat_len: int,
    max_tokens: int,
    beam_width: int,
    alpha: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_repeat_times: int,
    eos_bonus: float,
    min_decode_len: int,
    bracket_pairs: Dict[int, int] | None,
    bracket_penalty: float,
) -> List[int]:
    # 编码器前向
    enc_out, hidden, enc_lengths = sess_enc.run(None, {
        "features": feat,
        "feature_lengths": np.array([feat_len], dtype=np.int64),
    })

    T = enc_out.shape[1]
    mask = (np.arange(T)[np.newaxis, :] >= enc_lengths.reshape(-1, 1)).astype(np.bool_)

    close_to_open = {v: k for k, v in bracket_pairs.items()} if bracket_pairs else None

    def violates_ngram(seq: List[int], token_id: int) -> bool:
        if no_repeat_ngram_size <= 1 or len(seq) < no_repeat_ngram_size - 1:
            return False
        ngram = seq[-(no_repeat_ngram_size - 1):] + [token_id]
        for i in range(len(seq) - no_repeat_ngram_size + 1):
            if seq[i:i + no_repeat_ngram_size] == ngram:
                return True
        return False

    def apply_repetition_penalty(logits: np.ndarray, seq: List[int]) -> np.ndarray:
        if repetition_penalty <= 1.0:
            return logits
        logits = logits.copy()
        for token_id in set(seq):
            logit = logits[0, token_id]
            logits[0, token_id] = logit * repetition_penalty if logit < 0 else logit / repetition_penalty
        return logits

    def would_break_bracket(seq: List[int], token_id: int) -> bool:
        if not close_to_open or token_id not in close_to_open:
            return False
        need_open = close_to_open[token_id]
        opens = seq.count(need_open)
        closes = seq.count(token_id)
        return closes + 1 > opens

    def bracket_unmatched(seq: List[int]) -> int:
        if not bracket_pairs:
            return 0
        penalty = 0
        for open_id, close_id in bracket_pairs.items():
            diff = seq.count(open_id) - seq.count(close_id)
            if diff > 0:
                penalty += diff
        return penalty

    beams: List[Tuple[float, np.ndarray, List[int]]] = [(0.0, hidden, [SOS_ID])]
    completed: List[Tuple[float, List[int]]] = []

    for _ in range(max_tokens):
        new_beams: List[Tuple[float, np.ndarray, List[int]]] = []
        for cum_logp, h, seq in beams:
            if seq[-1] == EOS_ID:
                completed.append((cum_logp, seq))
                continue

            inp = np.array([seq[-1]], dtype=np.int64)
            logits, next_hidden = sess_dec.run(None, {
                "input_token": inp,
                "hidden": h,
                "encoder_outputs": enc_out,
                "enc_mask": mask,
            })
            

            logits = apply_repetition_penalty(logits, seq)
            if eos_bonus != 0.0 and len(seq) >= min_decode_len:
                logits[0, EOS_ID] = logits[0, EOS_ID] + eos_bonus
            if max_repeat_times > 0:
                from collections import Counter
                cnts = Counter(seq)
                for tid, cnt in cnts.items():
                    if cnt >= max_repeat_times:
                        logits[0, tid] = -1e9

            log_probs = _log_softmax_np(logits)
            topk_width = min(beam_width * 3, log_probs.shape[1])
            topk_idx = np.argpartition(-log_probs[0], topk_width - 1)[:topk_width]
            topk_idx = topk_idx[np.argsort(-log_probs[0, topk_idx])]

            accepted = 0
            for token_id in topk_idx:
                if max_repeat_times > 0 and seq.count(int(token_id)) >= max_repeat_times:
                    continue
                if violates_ngram(seq, int(token_id)):
                    continue
                if would_break_bracket(seq, int(token_id)):
                    continue
                new_seq = seq + [int(token_id)]
                new_beams.append((cum_logp + float(log_probs[0, token_id]), next_hidden, new_seq))
                accepted += 1
                if accepted >= beam_width:
                    break

            if accepted == 0:
                token_id = int(np.argmax(log_probs[0]))
                new_seq = seq + [token_id]
                new_beams.append((cum_logp + float(log_probs[0, token_id]), next_hidden, new_seq))

        def score(item: Tuple[float, np.ndarray, List[int]]) -> float:
            base = item[0] / (len(item[2]) ** alpha)
            return base - bracket_penalty * bracket_unmatched(item[2])

        new_beams.sort(key=score, reverse=True)
        beams = new_beams[:beam_width]
        if not new_beams:
            break

    if not completed:
        completed = [(b[0], b[2]) for b in beams]

    completed.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
    return completed[0][1]


def greedy_decode(sess_enc: ort.InferenceSession, sess_dec: ort.InferenceSession, feat: np.ndarray, feat_len: int, max_tokens: int) -> List[int]:
    enc_out, hidden, enc_lengths = sess_enc.run(None, {
        "features": feat,
        "feature_lengths": np.array([feat_len], dtype=np.int64),
    })
    enc_out = enc_out  # [1, T, 2H]
    hidden = hidden    # [1, 1, H]
    enc_lengths = enc_lengths  # [1]

    T = enc_out.shape[1]
    mask = (np.arange(T)[np.newaxis, :] >= enc_lengths.reshape(-1, 1)).astype(np.bool_)

    tokens = [SOS_ID]
    input_token = np.array([SOS_ID], dtype=np.int64)
    for _ in range(max_tokens):
        logits, next_hidden = sess_dec.run(None, {
            "input_token": input_token,
            "hidden": hidden,
            "encoder_outputs": enc_out,
            "enc_mask": mask,
        })
        if logits.ndim == 3:
            logits = logits.squeeze(1)
        next_token = int(np.argmax(logits, axis=1)[0])
        tokens.append(next_token)
        hidden = next_hidden
        input_token = np.array([next_token], dtype=np.int64)
        if next_token == EOS_ID:
            break
    return tokens


def pick_files(root: str, k: int) -> List[str]:
    if not os.path.exists(root):
        return []
    inks = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".inkml"):
                inks.append(os.path.join(r, f))
    if not inks:
        return []
    return random.sample(inks, min(k, len(inks)))


def main():
    parser = argparse.ArgumentParser(description="ONNX inference (quant or non-quant)")
    parser.add_argument("--encoder", default="checkpoints/encoder.onnx")
    parser.add_argument("--decoder", default="checkpoints/decoder.onnx")
    parser.add_argument("--cache", default="cache/vocab.pkl")
    parser.add_argument("--max_points", type=int, default=1200)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--root", default="data/raw/test", help="InkML root; fallback to train if empty")
    parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH_DEFAULT, help="Beam Search 宽度，1为贪婪")
    parser.add_argument("--rep_penalty", type=float, default=REP_PENALTY_DEFAULT, help="大于1抑制已出现 token")
    parser.add_argument("--no_repeat_ngram", type=int, default=NO_REPEAT_NGRAM_DEFAULT, help="阻止重复 n-gram，0 关闭")
    parser.add_argument("--max_token_repeat", type=int, default=MAX_TOKEN_REPEAT_DEFAULT, help="同一 token 最多出现次数，0 关闭")
    parser.add_argument("--eos_bonus", type=float, default=EOS_BONUS_DEFAULT, help="鼓励结束符偏置")
    parser.add_argument("--min_decode_len", type=int, default=MIN_DECODE_LEN_DEFAULT, help="达到该长度后才给 EOS 加偏置")
    parser.add_argument("--bracket_penalty", type=float, default=BRACKET_PENALTY_DEFAULT, help="括号不平衡惩罚系数")
    parser.add_argument(
        "--quant_mode",
        choices=["none", "int8"],
        default="none",
        help="选择加载的 ONNX 版本：none=FP32；int8=量化版 encoder_int8/decoder_int8",
    )
    args = parser.parse_args()

    # 如果使用默认路径，按 quant_mode 自动切换
    default_enc = "checkpoints/encoder.onnx"
    default_dec = "checkpoints/decoder.onnx"
    quant_enc = "checkpoints/encoder_int8.onnx"
    quant_dec = "checkpoints/decoder_int8.onnx"
    if args.quant_mode == "int8":
        if args.encoder == default_enc:
            args.encoder = quant_enc
        if args.decoder == default_dec:
            args.decoder = quant_dec

    char2id, id2char = load_vocab(args.cache)

    # 括号映射，用于 Beam Search 约束
    bracket_pairs = {}
    for op, cl in [("(", ")"), ("[", "]"), ("{", "}")]:
        op_id, cl_id = char2id.get(op), char2id.get(cl)
        if op_id is not None and cl_id is not None:
            bracket_pairs[op_id] = cl_id
    if not bracket_pairs:
        bracket_pairs = None

    providers = resolve_providers(args.quant_mode == "int8")
    print(f"🧠 ORT Providers: {providers}")
    try:
        sess_enc = ort.InferenceSession(args.encoder, providers=providers)
        sess_dec = ort.InferenceSession(args.decoder, providers=providers)
    except Exception as e:
        if args.quant_mode == "int8":
            print(f"⚠️ INT8 模型加载失败（{type(e).__name__}: {e}）。\n   - 已尝试 Provider: {providers}\n   - 可用 Provider: {ort.get_available_providers()}\n   请确认已安装支持量化算子的 onnxruntime（建议 >=1.16，或安装 onnxruntime-directml/onnxruntime-gpu），或重新导出 FP32 模型。即将自动回退 FP32。")
            args.quant_mode = "none"
            args.encoder = default_enc
            args.decoder = default_dec
            providers = resolve_providers(False)
            sess_enc = ort.InferenceSession(args.encoder, providers=providers)
            sess_dec = ort.InferenceSession(args.decoder, providers=providers)
        else:
            raise

    print(f"✅ encoder: {args.encoder}")
    print(f"✅ decoder: {args.decoder}")

    files = pick_files(args.root, 5)
    if not files:
        fallback = "data/raw/train"
        files = pick_files(fallback, 5)
        if not files:
            print("❌ No inkml files found")
            return
        else:
            print(f"⚠️ 使用备用目录: {fallback}")

    for path in files:
        strokes, label = parse_inkml(path)
        if strokes is None:
            print(f"⚠️ 跳过无法解析: {path}")
            continue
        feat, feat_len = strokes_to_features(strokes, max_points=args.max_points)
        current_alpha = get_dynamic_alpha(feat_len)

        if args.beam_width > 1:
            tokens = _beam_search_decode(
                sess_enc,
                sess_dec,
                feat,
                feat_len,
                max_tokens=args.max_tokens,
                beam_width=args.beam_width,
                alpha=current_alpha,
                repetition_penalty=args.rep_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
                max_repeat_times=args.max_token_repeat,
                eos_bonus=args.eos_bonus,
                min_decode_len=args.min_decode_len,
                bracket_pairs=bracket_pairs,
                bracket_penalty=args.bracket_penalty,
            )
        else:
            tokens = greedy_decode(sess_enc, sess_dec, feat, feat_len, args.max_tokens)
        latex_raw = tokens_to_latex(tokens, id2char)
        latex = sanitize_latex(latex_raw)
        print(f"\n==== 文件: {os.path.basename(path)}")
        print(f"🔍 预测: {latex}")
        if label:
            print(f"📝 标签: {label}")


if __name__ == "__main__":
    main()
