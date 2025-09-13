# config/context.py


from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import torch
from config.schema import OCTConfig
from config.utils import compute_derived_fields, load_motion_info
from core.bgcal.bg import BackgroundCorrector
import numpy as np

@dataclass
class OCTContext:
    config: OCTConfig
    bgMean: Optional[torch.Tensor] = None
    bgOsc: Optional[torch.Tensor] = None
    trigDelay: Optional[int] = None
    noiseProfile: Optional[torch.Tensor] = None
    refVol: Optional[np.ndarray] = None
    data_path: Optional[Path] = None # data_path 속성 추가

    # ✨ 2. 수정된 initialize 메서드
    def initialize(self, data_path: Union[str, Path], device="cuda"):
        """
        BackgroundCorrector를 사용하여 BG 및 Noise 프로필을 계산하고
        Context 객체를 초기화합니다.
        """
        print(f"[INFO] Initializing OCTContext for data: {data_path}")
        self.data_path = Path(data_path)
        
        # .data 파일 경로를 확인합니다.
        bin_path = self.data_path
        if bin_path.suffix != '.data':
            bin_path = bin_path.with_suffix('.data')
        
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary data file not found: {bin_path}")
            
        # (1) BackgroundCorrector 객체를 생성하고 실행합니다.
        #    config 객체를 info로 전달합니다.
        corrector = BackgroundCorrector(info=self.config, bin_path=bin_path, device=device)
        updated_info = corrector.run()

        # (2) run() 메서드가 반환한 업데이트된 info 객체로 config를 갱신합니다.
        self.config.update(updated_info) # config가 dict처럼 동작한다고 가정

        # (3) 계산된 결과를 OCTContext의 속성으로 할당합니다.
        #     BackgroundCorrector가 numpy 배열을 반환하므로 torch 텐서로 변환합니다.
        self.bgMean = torch.as_tensor(self.config['bgMean'], device=device)
        self.bgOsc = torch.as_tensor(self.config['bgOsc'], device=device)
        self.trigDelay = self.config['trigDelay']
        self.noiseProfile = torch.as_tensor(self.config['noiseProfile'], device=device)
        
        print("[INFO] OCTContext initialization complete.")

    def set_derived_fields(self):
        derived = compute_derived_fields(self.config)
        for k, v in derived.items():
            setattr(self.config, k, v)

    def load_motion(self):
        if self.config.correctSaccades and self.config.Motion.motionFile:
            print(f"[INFO] Loading motion data from: {self.config.Motion.motionFile}")
            motion_dict = load_motion_info(Path(self.config.Motion.motionFile))
            for k, v in motion_dict.items():
                setattr(self.config.Motion, k, v)
        else:
            # fallback defaults
            self.config.Motion.motionFrames = np.zeros(self.config.numScanFrames, dtype=bool)
            self.config.Motion.cumPhaseX = np.zeros(self.config.numScanFrames)
            self.config.Motion.cumPhaseY = np.zeros(self.config.numScanFrames)
            self.config.Motion.motionLineShift = np.zeros(self.config.numScanFrames)

            print("[WARN] Motion correction disabled or file missing.")            