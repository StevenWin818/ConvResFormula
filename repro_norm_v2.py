import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval import robust_normalize_tex, strip_font_styles

def test_normalization(gt, pred, label=""):
    norm_gt = robust_normalize_tex(strip_font_styles(gt))
    norm_pred = robust_normalize_tex(strip_font_styles(pred))
    
    print(f"Label: {label}")
    print(f"GT:   {gt}")
    print(f"PRED: {pred}")
    print(f"Norm GT:   {norm_gt}")
    print(f"Norm PRED: {norm_pred}")
    print(f"Equal:     {norm_gt == norm_pred}")
    print("-" * 20)
    return norm_gt == norm_pred

cases = [
    (r"\mathrm {G z} ={ \frac {D _{H }}{ L}} \mathrm {Re} \, \mathrm {P r}", 
     r"\mathrm {G z} ={ D_{ H} \over L} \mathrm {Re} \, \mathrm {P r}", "Frac vs Over"),
    (r"f_{X | Y= y }(x)", r"f_{X \mid Y= y }(x)", "Pipe vs Mid"),
    (r"P _{r}= P _{t }{ {G ^{2} \lambda ^{2} \sigma } \over { {( 4 \pi ) }^{3} R ^{4 }}} \propto { \frac { \sigma }{R ^{4 }}}",
     r"P _{r}= P _{t }{ \frac {G ^{2} \lambda ^{2} \sigma }{ (4 \pi )^{3} R ^{4 }}} \propto { \frac { \sigma }{R ^{4 }}}", "Nested Over/Frac"),
    (r"\mathbf {Z} ={ \begin {pmatrix}0& -1\\ -1& 0 \end {pmatrix}}", 
     r"\mathbf {Z} ={ \begin {pmatrix}0& -1\\ -1& 0 \end {pmatrix}}", "Identical Matrix"),
    (r"{ \begin {pmatrix} u &0\\0& 1 \end {pmatrix}}", 
     r"{ \begin {pmatrix} u &0\\0& 1 \end {pmatrix}}", "Identical Matrix 2"),
]

all_passed = True
for gt, pred, label in cases:
    if not test_normalization(gt, pred, label):
        all_passed = False

if all_passed:
    print("All normalization tests passed!")
else:
    print("Some normalization tests failed.")
