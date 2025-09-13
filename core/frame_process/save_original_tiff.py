# core/frame_process/save_original_tiff.py

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from config.utils import load_mat_info

def save_original_frames(img: torch.Tensor, info: dict, d_file_name: str, vol: int):
    """
    MATLAB의 Original frame 저장 로직을 Python으로 재현:
    - img: complex tensor (z, x, frames)
    - info: calibration dict with 'saveOriginal', 'segmentOnly', 'noiseFloor', 'dBRange', ...
    - d_file_name: base path for output filename
    - vol: 1-based volume idx
    """

    if not info.get("saveOriginal", False) or info.get("segmentOnly", False):
        return

    output_path = Path(
    f"/home/work/OCT_DL/CDAC_OCT/CDAC_PYTHON/cache/KAIST/{Path(d_file_name).stem}_Orig_{vol:02d}.tif"
)
    if output_path.exists():
        output_path.unlink()

    z, xN, F = img.shape
    noise_floor = float(info.get("noiseFloor", 0))
    dB_range = float(info.get("dBRange", 40))

    frames = []
    for f in range(F):
        int_img = img[:, :, f]
        intensity = (int_img.real**2 + int_img.imag**2).cpu().numpy()

        # convert to dB and scale to uint16
        int_log = 10 * np.log10(intensity + 1e-12) - noise_floor
        # normalize [0, dB_range] → [0, 65535]
        scaled = np.clip(int_log / dB_range, 0, 1) * (2**16 - 1)
        img_uint16 = scaled.astype(np.uint16)

        pil_img = Image.fromarray(img_uint16, mode='I;16')
        frames.append(pil_img)

        if (f % 10) == 0:
            print(f"Saving original volume {vol}, frame {f+1}/{F}")

    # 첫 프레임부터 나머지를 append하여 multi-page TIFF 저장
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        compression="none"
    )
    print(f"Saved original TIFF: {output_path}")