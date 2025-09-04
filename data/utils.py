import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def to_yaml_safe(obj):
    """복소수/넘파이/판다스 포함 임의 객체를 YAML-safe 기본형으로 재귀 변환"""
    # None / 기본형 그대로
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # 복소수 → {'re': float, 'im': float}
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}

    # numpy 스칼라
    if isinstance(obj, np.generic):
        return to_yaml_safe(obj.item())

    # numpy 배열: list로 바꾸고 내부 재귀
    if isinstance(obj, np.ndarray):
        return to_yaml_safe(obj.tolist())

    # pandas Series/Index: list로
    if isinstance(obj, (pd.Series, pd.Index)):
        return to_yaml_safe(obj.tolist())

    # list/tuple: 각 원소 재귀
    if isinstance(obj, (list, tuple)):
        return [to_yaml_safe(v) for v in obj]

    # dict: 값 재귀 (키는 문자열화)
    if isinstance(obj, dict):
        return {str(k): to_yaml_safe(v) for k, v in obj.items()}

    # 그 외 알 수 없는 타입은 문자열로
    return str(obj)

def dump_info_yaml_from_df(df2: pd.DataFrame, out_path: Path):
    info_dict = df2.iloc[0].to_dict()
    safe_dict = to_yaml_safe(info_dict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(safe_dict, f, sort_keys=False, allow_unicode=True, width=120)
    print(f"[save] wrote {out_path}")

# --- Load YAML back to DataFrame (rebuild complex numbers) ---
def from_yaml_safe(obj):
    """to_yaml_safe로 저장된 구조를 다시 파이썬 타입(특히 complex)으로 복원"""
    if isinstance(obj, dict):
        # {'re': x, 'im': y} 패턴은 complex로 복원
        if set(obj.keys()) == {"re", "im"} and \
           isinstance(obj["re"], (int, float)) and isinstance(obj["im"], (int, float)):
            return complex(obj["re"], obj["im"])
        return {k: from_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_yaml_safe(v) for v in obj]
    return obj