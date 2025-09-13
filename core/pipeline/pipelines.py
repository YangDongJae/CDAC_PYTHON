import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

class OctProcessingPipeline:
    """
    '완성된' info 객체를 받아 볼륨 처리만 수행하는 단순화된 파이프라인.
    GPU 사용을 우선 시도하고, CUDA OOM 발생 시 자동으로 CPU로 폴백합니다.
    """
    def __init__(self, dfile_path_no_ext: str, processed_info: dict, device: str):
        from core.frame_process.processor import FrameProcessor  # 로컬 import 권장
        self.FrameProcessor = FrameProcessor

        self.dfile_path = Path(dfile_path_no_ext)
        self.info = processed_info
        # 요청한 device가 cuda여도 가용성 없으면 cpu로
        self.device = ("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"Simplified Pipeline initialized with processed info. (device={self.device})")

    def _make_processor(self, device: str):
        return self.FrameProcessor(self.info, device)

    def _is_cuda_oom(self, err: Exception) -> bool:
        msg = str(err)
        return isinstance(err, RuntimeError) and ("CUDA out of memory" in msg or "cuda runtime error" in msg)

    def process_volume(self, vol_idx: int) -> torch.Tensor:
        """단일 볼륨 전체를 FrameProcessor를 사용하여 처리합니다.
        - 우선 GPU로 처리 시도
        - 프레임 처리 중 CUDA OOM 발생 시 CPU로 전환하여 해당 프레임부터 계속
        - 누적 스택은 항상 CPU에 저장해 GPU 메모리 폭증 방지
        """
        print(f"\nProcessing volume {vol_idx} using pre-loaded info...")
        num_img_frames = int(self.info['numImgFrames'])
        all_fringes = []

        processor = self._make_processor(self.device)

        with torch.inference_mode(), open(str(self.dfile_path) + ".data", "rb") as fid:
            for i in range(1, num_img_frames + 1):
                if (i % 50 == 0) or (i == num_img_frames):
                    print(f"  Processing frame {i}/{num_img_frames}...")

                try:
                    fr = processor.process(fid, vol_idx, i)  # GPU 우선
                except Exception as e:
                    if self._is_cuda_oom(e) and self.device == "cuda":
                        print(f"[WARN] CUDA OOM at frame {i}. Switching to CPU...")
                        # GPU 메모리 정리
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        # CPU로 전환 후 동일 프레임 재처리
                        self.device = "cpu"
                        processor = self._make_processor(self.device)
                        fr = processor.process(fid, vol_idx, i)
                    else:
                        # 다른 에러는 그대로 전파
                        raise

                # 누적은 항상 CPU에 (GPU 메모리 폭증 방지)
                all_fringes.append(fr.cpu())
                # 중간 GPU 메모리 청소 (가능 시)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 스택도 CPU에서 수행
        fringes_vol = torch.stack(all_fringes, dim=2)  # CPU tensor
        print(f"Volume processing complete. Final shape: {fringes_vol.shape} (device=CPU)")
        return fringes_vol