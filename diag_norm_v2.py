import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval import robust_normalize_tex, strip_font_styles, _unwrap_redundant_braces, ARG_COMMANDS

def test_diag(gt):
    print(f"Original: {gt}")
    stripped = strip_font_styles(gt)
    print(f"Stripped: {stripped}")
    
    # Trace _unwrap_redundant_braces
    def debug_unwrap(tex, depth=0):
        out = []
        i = 0
        while i < len(tex):
            if tex[i] == "{" : # Simplified for diag
                j = find_match(tex, i)
                if j != -1:
                    inner = tex[i+1:j]
                    inner = debug_unwrap(inner, depth+1)
                    prev_text = "".join(out).strip()
                    prev = prev_text[-1] if prev_text else ""
                    
                    is_cmd_arg = False
                    for cmd in ARG_COMMANDS:
                        if prev_text.endswith(cmd):
                            is_cmd_arg = True
                            break
                    
                    if not is_cmd_arg and prev == "}" and "\\frac" in prev_text:
                        is_cmd_arg = True
                        
                    nxt_text = tex[j+1:].strip()
                    nxt = nxt_text[0] if nxt_text else ""
                    
                    is_sub = (prev in "^_" or nxt in "^_")
                    is_single = len(inner) == 1 # Simplified
                    
                    keep = False
                    if is_cmd_arg and not is_single: keep = True
                    elif is_sub and not is_single: keep = True
                    
                    print(f"DEPTH {depth}: inner={inner}, prev_text={repr(prev_text)}, is_cmd_arg={is_cmd_arg}, is_sub={is_sub}, keep={keep}")
                    
                    if keep: out.append("{" + inner + "}")
                    else: out.append(inner)
                    i = j + 1
                    continue
            out.append(tex[i])
            i += 1
        return "".join(out)

    def find_match(text, open_idx):
        depth = 0
        for i in range(open_idx, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0: return i
        return -1

    unwrapped = debug_unwrap(stripped)
    print(f"Unwrapped: {unwrapped}")
    
    final = robust_normalize_tex(gt)
    print(f"Final: {final}")

case = r"P _{r}= P _{t }{ {G ^{2} \lambda ^{2} \sigma } \over { {( 4 \pi ) }^{3} R ^{4 }}} \propto { \frac { \sigma }{R ^{4 }}}"
test_diag(case)
