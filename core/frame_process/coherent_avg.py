# core/frame_process/coherent_avg.py

import torch
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import gc # 가비지 컬렉션 모듈 import

# 멀티프로세싱을 위한 전역 도우미 함수
def _process_single_frame(args):
    """
    단일 2D 프레임을 처리하고, 원래의 인덱스와 함께 반환합니다.
    """
    frame_idx, img_frame_np, depth, scanN, numCoherent = args
    
    out_lines = scanN * numCoherent
    re_resample = np.linspace(1, out_lines, num=scanN)

    # 보간
    interp = np.array([
        np.interp(np.arange(1, out_lines + 1), re_resample, img_frame_np[d, :], left=0, right=0)
        for d in range(depth)
    ])
    
    # 이동 평균
    kernel = np.ones(numCoherent) / numCoherent
    avg_frame = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, interp)
    
    # 원래 인덱스와 함께 처리된 프레임을 반환
    return frame_idx, avg_frame


def coherent_average_and_fft(
    img: torch.Tensor,
    Info: dict,
    cache_path: Path
):
    """
    - imap_unordered를 사용해 메모리 효율적으로 결과를 취합합니다.
    - 명시적인 메모리 정리를 통해 안정성을 높입니다.
    """
    assert Info.get("coherentAvg", False), "Info.coherentAvg must be True to apply this."
    
    depth, scanN, frames = img.shape
    numCoherent = int(Info["numCoherentAvg"])
    numFT = int(Info["numFTSamples"])
    device = img.device

    # 1. 최종 결과를 담을 빈 NumPy 배열을 미리 생성
    out_lines = scanN * numCoherent
    # 입력 텐서와 동일한 dtype을 사용하되, complex가 아니면 float32로 처리
    dtype = img.cpu().numpy().dtype
    img_coh_np = np.zeros((depth, out_lines, frames), dtype=dtype)

    # 2. 병렬 처리할 작업 목록 생성
    tasks = [(f, img[:, :, f].cpu().numpy(), depth, scanN, numCoherent) for f in range(frames)]
    
    # 입력 텐서는 더 이상 필요 없으므로 메모리에서 해제
    del img
    gc.collect()

    print(f"Starting coherent averaging for {frames} frames using {cpu_count()} CPU cores...")

    # 3. imap_unordered를 사용하여 결과를 하나씩 받아서 채워넣기
    with Pool(processes=cpu_count()) as pool:
        # imap_unordered는 작업이 완료되는 순서대로 (인덱스, 결과)를 반환
        for i, (frame_idx, processed_frame) in enumerate(pool.imap_unordered(_process_single_frame, tasks)):
            img_coh_np[:, :, frame_idx] = processed_frame
            # 진행 상황 표시 (선택 사항)
            print(f"\rProcessing... {i+1}/{frames} frames completed.", end="")

    print("\nCoherent averaging finished.")

    # 4. 결과 취합 및 텐서 변환
    img_coh = torch.from_numpy(img_coh_np).to(device)
    
    # 더 이상 필요 없는 NumPy 배열을 메모리에서 명시적으로 해제
    del img_coh_np
    gc.collect()

    # 5. FFT depth 축
    fringes_coh = torch.fft.fft(img_coh, n=numFT, dim=0)

    # 6. 캐시 저장
    res = {
        "img_coh": img_coh.cpu(),
        "fringes_coh": fringes_coh.cpu()
    }
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, cache_path)
    
    # 더 이상 필요 없는 대용량 텐서들을 반환 전에 정리
    del img_coh, fringes_coh, res
    gc.collect()

    # 저장된 파일에서 다시 로드하여 반환 (메모리 파편화 방지)
    reloaded_data = torch.load(cache_path)
    return reloaded_data["img_coh"].to(device), reloaded_data["fringes_coh"].to(device)