# core/io/saver.py
from pathlib import Path
import shutil
import torch
import numpy as np
from typing import Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# 단일 이미지 저장 백엔드
_iio = None
_PIL = None
try:
    import imageio.v2 as _iio
except Exception:
    try:
        from PIL import Image as _PIL
    except Exception:
        pass
class Saver:
    def __init__(self, info: Dict, out_dir: Path):
        self.info = info
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

    # --- 내부 유틸 ---
    def _get_db_params(self) -> tuple[float, float]:
        # dict 접근은 [] 형태로!
        dBRange = float(self.info["dBRange"])
        noiseFloor = float(self.info["noiseFloor"])
        return dBRange, noiseFloor

    def _to_uint16(self, img_db: torch.Tensor) -> np.ndarray:
        dBRange, _ = self._get_db_params()
        img_db = torch.clamp(img_db, 0, dBRange)
        return (img_db / dBRange * 65535).to(torch.uint16).cpu().numpy()

    # --- 기존 메서드 (참고: dict 접근 수정) ---
    def save_volume(
        self,
        vol: int,
        img_cube: torch.Tensor,          # (Nz, L, F) complex
        layers: Tuple[np.ndarray, ...],  # ILM, NFL, ISOS, RPE
        meta: Dict,
    ):
        vol_tag = f"{vol:02d}"
        int_path = self.out_dir / f"eye1_Int_{vol_tag}.tif"
        ph_path  = self.out_dir / f"eye1_Ph_{vol_tag}.tif"

        Nz, L, F = img_cube.shape
        dBRange, noise_floor = self._get_db_params()

        # 새로 쓰기 위해 기존 파일 삭제(선택)
        if int_path.exists(): int_path.unlink()
        if ph_path.exists():  ph_path.unlink()

        with tiff.TiffWriter(str(int_path), bigtiff=True) as int_tw, \
             tiff.TiffWriter(str(ph_path),  bigtiff=True) as ph_tw:

            for f in range(F):
                # intensity(dB)
                intensity = (img_cube[:, :, f].real**2 + img_cube[:, :, f].imag**2).float()
                eps = torch.finfo(torch.float32).tiny
                intensity = torch.clamp(intensity, min=eps)
                int_db = 10 * torch.log10(intensity) - noise_floor
                int_tw.write(self._to_uint16(int_db), contiguous=True)

                # phase → uint16 ( -pi..pi → 0..65535 )
                phase = torch.angle(img_cube[:, :, f])
                phase_u16 = ((phase / (2 * torch.pi) + 0.5) * 65535).to(torch.uint16).cpu().numpy()
                ph_tw.write(phase_u16, contiguous=True)

        # TODO: layers 저장 등 필요 시 여기에 추가

    # --- 추가: MATLAB 'saveOriginal' 블록 포팅(+디버그) ---
    #OUTPUT .tif
    def save_original_stack(
        self,
        vol: int,
        img: torch.Tensor,                # (L, W, F) complex, IFFT 완료 상태
        dfile_stem: str = "eye1",
        *,
        preview_every: int = 10,          # MATLAB: mod(frame,10)==0
        show_preview: bool = False,       # True면 10프레임마다 figure(2)
        delete_existing: bool = True,
        debug_stats: bool = True,
    ) -> Optional[Path]:
        # MATLAB: if Info.saveOriginal && ~Info.segmentOnly
        if not self.info.get("saveOriginal", False) or bool(self.info.get("segmentOnly", False)):
            print("[save_original] skipped (Info.saveOriginal==False or Info.segmentOnly==True)")
            return None

        assert torch.is_complex(img), f"img must be complex, got {img.dtype}"
        L, W, F = img.shape
        Lcrop = int(self.info["numImgPixels"])
        assert L >= Lcrop, "img depth(L) < Info.numImgPixels"
        assert F == int(self.info["numImgFrames"]), "img frames(F) != Info.numImgFrames"
        numVolumes = int(self.info.get("numVolumes", 1))

        dBRange, noiseFloor = self._get_db_params()
        out_path = self.out_dir / f"{dfile_stem}_Orig_{vol:02d}.tif"
        if delete_existing and out_path.exists():
            out_path.unlink()

        if show_preview and plt is None:
            print("[save_original] matplotlib not available → preview disabled.")
            show_preview = False
        if show_preview:
            plt.figure(num=2, figsize=(6, 4))

        print(f"[save_original] → {out_path}")
        with tiff.TiffWriter(str(out_path), bigtiff=True) as tw:
            last_msg = ""
            for frame in range(1, F + 1):
                sl = img[:Lcrop, :, frame - 1]                 # (Lcrop, W)
                intensity = (sl.real**2 + sl.imag**2).float()  # |img|^2
                eps = torch.finfo(torch.float32).tiny
                intensity = torch.clamp(intensity, min=eps)
                int_db = 10.0 * torch.log10(intensity) - noiseFloor
                u16 = self._to_uint16(int_db)
                tw.write(u16, contiguous=True)

                if frame % preview_every == 0:
                    # 진행 로그
                    msg = f"OCT original saving vol {vol}/{numVolumes}, frame {frame}/{F}..."
                    print("\r" + " " * len(last_msg), end="\r")
                    print(msg, end="", flush=True)
                    last_msg = msg

                    # 미리보기
                    if show_preview:
                        disp = u16.astype(np.float32) * (dBRange / 65535.0)
                        plt.clf()
                        plt.imshow(disp, cmap="gray", vmin=0, vmax=dBRange)
                        plt.title(f"OCT Original frame {frame}")
                        plt.axis("off")
                        plt.pause(0.01)

                    # 디버그 통계
                    if debug_stats:
                        print("\n[debug]")
                        print(f"  |img|^2 : min={float(intensity.min()):.6e}, max={float(intensity.max()):.6e}, any NaN={bool(torch.isnan(intensity).any())}")
                        print(f"  dB      : min={float(int_db.min()):.3f}, max={float(int_db.max()):.3f}")
                        print(f"  uint16  : min={int(u16.min())}, max={int(u16.max())}, shape={u16.shape}")

            print()  # 줄바꿈
        print("[save_original] done.")
        return out_path

    def save_original_frames_dir(
        self,
        vol: int,
        img: torch.Tensor,                 # (L, W, F) complex, IFFT 완료 상태
        dfile_stem: str = "eye1",
        *,
        medium: str = "png",               # "png"|"jpg"|"jpeg"|"tif" 등
        folder_tag: str = "Orig",
        delete_existing: bool = True,
        preview_every: int = 10,
        show_preview: bool = False,
        debug_stats: bool = True,
    ) -> Optional[Path]:
        """
        MATLAB 블록을 프레임 단위 '개별 파일' 저장으로 변환.
        - 폴더명: {stem}_{folder_tag}_{vol:02d}_{medium}
        - 파일명: {stem}_{vol:02d}_{frame:04d}.{medium}
        - PNG인 경우 16-bit PNG로 저장(권장). JPEG 선택 시 8-bit로 다운스케일.
        """
        # governing flags (MATLAB: if Info.saveOriginal && ~Info.segmentOnly)
        if not self.info.get("saveOriginal", False) or bool(self.info.get("segmentOnly", False)):
            print("[save_original_frames_dir] skipped (flags)")
            return None

        assert torch.is_complex(img), f"img must be complex, got {img.dtype}"
        L, W, F = img.shape
        Lcrop = int(self.info["numImgPixels"])
        assert L >= Lcrop, "img depth(L) < Info.numImgPixels"
        assert F == int(self.info["numImgFrames"]), "img frames(F) != Info.numImgFrames"

        dBRange, noiseFloor = self._get_db_params()

        # 폴더 경로 구성 (매체명을 폴더명에 반영)
        folder_name = f"{dfile_stem}_{folder_tag}_{vol:02d}_{medium.lower()}"
        out_dir = (self.out_dir / folder_name)
        if delete_existing and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 미리보기 준비
        if show_preview and plt is None:
            print("[save_original_frames_dir] matplotlib not available → preview disabled.")
            show_preview = False
        if show_preview:
            plt.figure(num=2, figsize=(6, 4))

        print(f"[save_original_frames_dir] → {out_dir}  (F={F}, medium={medium})")

        for frame in range(1, F + 1):
            sl = img[:Lcrop, :, frame - 1]                # (Lcrop, W)
            intensity = (sl.real**2 + sl.imag**2).float() # |img|^2
            eps = torch.finfo(torch.float32).tiny
            intensity = torch.clamp(intensity, min=eps)

            int_db = 10.0 * torch.log10(intensity) - noiseFloor
            u16 = self._to_uint16(int_db)                 # (Lcrop, W) uint16

            # 저장 파일 경로
            fname = f"{dfile_stem}_{vol:02d}_{frame:04d}.{medium.lower()}"
            fpath = out_dir / fname

            # 포맷별 저장
            if medium.lower() in ("png", "tif", "tiff"):
                if _iio is not None and medium.lower() == "png":
                    _iio.imwrite(fpath, u16)  # 16-bit PNG 지원
                elif _iio is not None and medium.lower() in ("tif", "tiff"):
                    _iio.imwrite(fpath, u16)
                else:
                    # PIL fallback: 16-bit PNG 저장
                    if _PIL is None:
                        raise RuntimeError("Neither imageio nor PIL available to save images.")
                    imgP = _PIL.fromarray(u16, mode="I;16")
                    imgP.save(fpath)
            elif medium.lower() in ("jpg", "jpeg"):
                # JPEG는 8-bit만 안전 → 다운스케일
                u8 = (u16 >> 8).astype(np.uint8)
                if _iio is not None:
                    _iio.imwrite(fpath, u8, quality=95)
                else:
                    if _PIL is None:
                        raise RuntimeError("Neither imageio nor PIL available to save images.")
                    imgP = _PIL.fromarray(u8, mode="L")
                    imgP.save(fpath, quality=95, subsampling=0)
            else:
                raise ValueError(f"Unsupported medium: {medium}")

            # 진행/미리보기/통계
            if frame % preview_every == 0:
                msg = f"OCT original saving (dir) vol {vol}/{self.info.get('numVolumes', 1)}, frame {frame}/{F}..."
                print(msg)

                if show_preview:
                    disp = (u16.astype(np.float32) * (dBRange / 65535.0))
                    plt.clf()
                    plt.imshow(disp, cmap="gray", vmin=0, vmax=dBRange)
                    plt.title(f"OCT Original frame {frame}")
                    plt.axis("off")
                    plt.pause(0.01)

                if debug_stats:
                    print("[debug]")
                    print(f"  |img|^2 : min={float(intensity.min()):.6e}, max={float(intensity.max()):.6e}, any NaN={bool(torch.isnan(intensity).any())}")
                    print(f"  dB      : min={float(int_db.min()):.3f}, max={float(int_db.max()):.3f}")
                    print(f"  u16     : min={int(u16.min())}, max={int(u16.max())}, shape={u16.shape}")

        print("[save_original_frames_dir] done.")
        return out_dir        