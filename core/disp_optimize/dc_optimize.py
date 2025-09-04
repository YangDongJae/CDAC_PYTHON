#core/dc_optimize.py
import numpy as np
import torch
from scipy.optimize import differential_evolution
from pathlib import Path
from typing import Dict

def dispersion_compensate(volume: torch.Tensor, disp_param, device):
    d1, d2 = disp_param
    _, _, W = volume.shape
    k = torch.linspace(-1, 1, W, device=device)
    phase = torch.exp(1j * (d1 * k**2 + d2 * k**3))

    vol_f = torch.fft.fft(volume.to(torch.complex64), dim=-1)
    vol_f *= phase
    vol_c = torch.fft.ifft(vol_f, dim=-1)

    return vol_c.abs()  # still on GPU

def sharpness_metric(volume: torch.Tensor):
    return torch.var(volume).item()

def eval_disp(disp_param, volume_gpu: torch.Tensor, device):
    comped = dispersion_compensate(volume_gpu, disp_param, device)
    return -sharpness_metric(comped)

def load_ref_volume(info: dict, data_path,frames: int = 10) -> np.ndarray:
    """
    Load the first N frames from raw .data file and return their average
    (only scan lines, not flyback). Efficient and memory-safe.
    """
    scan_lines = info["numScanLines"]
    flyback_lines = info["numFlybackLines"]
    lines_per_frame = scan_lines + flyback_lines
    samples = info["numSamples"]


    frame_size = lines_per_frame * samples
    raw_frames = []

    with open(data_path, "rb") as f:
        for i in range(frames):
            offset = i * frame_size * 2  # uint16 = 2 bytes
            f.seek(offset)
            frame = np.fromfile(f, count=frame_size, dtype=np.uint16)
            if frame.size != frame_size:
                raise ValueError(f"Frame {i}: expected {frame_size}, got {frame.size}")
            frame = frame.reshape((lines_per_frame, samples))
            valid = frame[:scan_lines, :]  # exclude flyback
            raw_frames.append(valid)

    avg = np.mean(raw_frames, axis=0).astype(np.float32)  # shape: (scan_lines, samples)
    return avg

def optimize_dispersion_coefficients(info: Dict, ref_vol: np.ndarray, useCUDA: bool = False):
    # autoDisp가 0이면 실행하지 않음
    if info.get("autoDisp", 0) == 0:
        print("🚫 Dispersion Compensation disabled.")
        return

    # ref_vol이 None이면 에러
    if ref_vol is None:
        raise ValueError("ref_vol is required but not provided.")

    device = torch.device("cuda" if useCUDA and torch.cuda.is_available() else "cpu")
    volume_gpu = torch.from_numpy(ref_vol).unsqueeze(0).to(device=device, dtype=torch.float32)

    # 초기 계수값: dispComp 또는 fallback to [0, 0]
    d1_init, d2_init = info.get("dispCompParam", [0.0, 0.0])
    print(f"\n🚀 Starting 3-stage Dispersion Optimization (device={device})")

    # Stage 1: optimize d1
    res_d1 = differential_evolution(
        lambda d1: eval_disp([d1[0], d2_init], volume_gpu, device),
        bounds=[(d1_init - 25, d1_init + 25)],
        disp=True,
    )
    d1_opt = res_d1.x[0]

    # Stage 2: optimize d2
    res_d2 = differential_evolution(
        lambda d2: eval_disp([d1_opt, d2[0]], volume_gpu, device),
        bounds=[(d2_init - 5, d2_init + 5)],
        disp=True,
    )
    d2_opt = res_d2.x[0]

    # Stage 3: refine (d1, d2)
    res_final = differential_evolution(
        lambda d: eval_disp(d, volume_gpu, device),
        bounds=[(d1_opt - 1, d1_opt + 1), (d2_opt - 0.1, d2_opt + 0.1)],
        disp=True,
    )
    final_d1, final_d2 = res_final.x
    info["dispComp"] = [final_d1, final_d2]  # 결과 갱신
    print(f"✅ Final optimized dispersion coefficients: {info['dispComp']}")