"""ONNX 导出兼容性回归测试：默认路径不应包含 CTC 辅助头。"""

import os
import tempfile

import onnx
import torch

from src.models.latex_ocr_model import LatexOCRModel


class DefaultPathWrapper(torch.nn.Module):
    def __init__(self, model: LatexOCRModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        # 不显式传 return_aux，必须保持历史推理路径。
        return self.model(images=images, tgt_seq=tgt_seq, is_causal=True)


class AuxPathWrapper(torch.nn.Module):
    def __init__(self, model: LatexOCRModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor):
        # 训练专用辅助路径：用于对比图中是否包含 CTC 分支参数。
        return self.model(images=images, tgt_seq=tgt_seq, is_causal=True, return_aux=True)


def _export(wrapper: torch.nn.Module, path: str) -> None:
    wrapper.eval()
    with torch.no_grad():
        images = torch.randn(1, 1, 64, 96)
        tgt = torch.randint(0, 100, (1, 12), dtype=torch.long)
        torch.onnx.export(
            wrapper,
            (images, tgt),
            path,
            input_names=["images", "tgt_seq"],
            output_names=["out0", "out1"] if isinstance(wrapper, AuxPathWrapper) else ["logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "tgt_seq": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            dynamo=False,
        )


def _has_ctc_head(path: str) -> bool:
    graph = onnx.load(path).graph
    init_names = [init.name for init in graph.initializer]
    return any("ctc_head" in name for name in init_names)


def main() -> None:
    model = LatexOCRModel(
        vocab_size=100,
        d_model=64,
        pad_id=0,
        eos_id=3,
        vision_model_name="convnextv2_pico",
        vision_pretrained=False,
        use_gradient_checkpointing=False,
    )

    with tempfile.TemporaryDirectory() as td:
        default_path = os.path.join(td, "default.onnx")
        aux_path = os.path.join(td, "aux.onnx")

        _export(DefaultPathWrapper(model), default_path)
        _export(AuxPathWrapper(model), aux_path)

        assert not _has_ctc_head(default_path), "默认导出路径不应包含 ctc_head 参数"
        assert _has_ctc_head(aux_path), "辅助导出路径应包含 ctc_head 参数"

    print("ONNX_RETURN_AUX_COMPAT_OK")


if __name__ == "__main__":
    main()
