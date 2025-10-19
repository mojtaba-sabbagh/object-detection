# detection/utils.py
from typing import Dict, Mapping, Any

def merge_counts(dst: Dict[str, int], src: Mapping[str, Any] | None) -> Dict[str, int]:
    if not src:
        return dst
    for k, v in src.items():
        try:
            dst[k] = dst.get(k, 0) + int(v)
        except (TypeError, ValueError):
            # ignore non-numeric values gracefully
            pass
    return dst
