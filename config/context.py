# config/context.py


from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import torch
from config.schema import OCTConfig
from core.bgcal import calc_bg, calc_noise
from config.utils import compute_derived_fields, load_motion_info
import numpy as np

@dataclass
class OCTContext:
    config: OCTConfig
    bgMean: Optional[torch.Tensor] = None
    bgOsc: Optional[torch.Tensor] = None
    trigDelay: Optional[int] = None
    noiseProfile: Optional[torch.Tensor] = None
    refVol: Optional[np.ndarray] = None      

    def initialize(self, data_path: Union[str, Path], device="cuda"):
        self.bgMean, self.bgOsc, self.trigDelay = calc_bg(data_path, self.config, device)
        self.noiseProfile                      = calc_noise(data_path, self.config, device)
        self.data_path = data_path


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