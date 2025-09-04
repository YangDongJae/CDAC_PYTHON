# config/utils.py
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import List, Optional, Union
import numpy as np 
import torch

class MotionConfig(BaseModel):
    enable: bool = True
    motionFile: Optional[Union[str, Path]] = None
    motionLineShift: Optional[np.ndarray] = None
    motionFrames: Optional[np.ndarray] = None
    cumPhaseX: Optional[np.ndarray] = None
    cumPhaseY: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class CAOConfig(BaseModel):
    enable: bool = True
    useCUDA: bool = False
    zMaxOrder: int = 4
    zCoeffsLB: int = -20
    zCoeffsUB: int = 20
    zeroPadFactor: int = 2
    saveProgress: bool = True

    zCoeffsN: Optional[int] = None
    imSize: Optional[List[int]] = None
    FDsize: Optional[List[int]] = None

class ISAMConfig(BaseModel):
    enable: bool = True
    autoOpt: bool = False
    useCUDA: bool = False
    scanFactor: float = 1.35
    scanFactorLB: float = 1.2
    scanFactorUB: float = 1.8
    saveProgress: bool = True

class SavingOptions(BaseModel):
    saveOriginal: bool = False
    saveIntensity: bool = True
    saveFlattened: bool = False
    savePhase: bool = False
    RPEflatLoc: int = 384

class OCTConfig(BaseModel):
    root: Path
    matDir: Path
    
    noiseProfile: Optional[np.ndarray] = None
    
    # ONH-related 
    ONHmask: Optional[np.ndarray] = None          # (H, W) bool
    validVols: Optional[torch.Tensor] = None      # (m, n) bool
    partialVols: Optional[torch.Tensor] = None    # (m, n) bool
    zfMap: Optional[np.ndarray] = None            # (m, n) float

    # Scan Pattern
    numScanLines: int
    numFlybackLines: int
    numScanFrames: int
    scanSize: List[float]
    subdivFactors: List[int]
    numImgLines: int
    numImgFrames: int

    # Shift
    initLineShift: int
    bgShift: int
    bgLines: int
    trigDelayRange: int

    # Processing
    imgPixelsStart: int
    numImgPixels: int
    numEnFacePixels: int
    bgBW: int
    adaptiveBG: bool
    adaptiveBGOsc: bool
    correctSaccades: bool
    excludeONH: bool

    # Signal Window
    noiseFloor: int
    dBRange: int

    # Dispersion
    dispCompParam: List[float]
    autoDisp: bool

    # Phase Stabilization
    stabilizePhaseRepeats: int
    stabilizePhaseRevert: bool
    stabilizePhaseWrap: bool

    # Segmentation
    segmentFigureOn: bool
    segmentSigma: List[float]
    ILMshift: int
    NFLminRange: int
    NFLmaxRange: int
    GCLprct: List[int]
    GCLmaxRange: int
    segmentOnly: bool

    # Submodules
    CAO: CAOConfig
    ISAM: ISAMConfig
    Save: SavingOptions
    Motion : MotionConfig

    numResampLines: Optional[int] = None
    usedSamples: Optional[List[int]] = None
    resampTraceA: Optional[np.ndarray] = None
    dispComp: List[Union[float, complex]]
    spectralWindow: Optional[np.ndarray] = None
    depthPerPixel: Optional[float] = None    

    #Derived Fields
    numVolumes: Optional[int] = None
    centerX: Optional[torch.Tensor] = None
    centerY: Optional[torch.Tensor] = None
    sigmaX: Optional[float] = None
    sigmaY: Optional[float] = None
    sigmaZ: Optional[float] = None

    #FROM info.mat
    trigDelay: Optional[int] = None
    numSamples: Optional[int] = None
    usedSamples: Optional[List[int]] = None
    numUsedSamples: Optional[int] = None
    numFTSamples: Optional[int] = None
    bgLineShift: Optional[int] = None
    bgLines: Optional[int] = None
    bgRef: Optional[List[float]] = None
    noiseFloor: Optional[float] = None
    bgBW: Optional[int] = None
    depthPerPixel: Optional[float] = None
    radPerPixel: Optional[float] = None
    spectralWindow: Optional[List[float]] = None
    resampTraceA: Optional[List[float]] = None
    dispComp: Optional[List[float]] = None
    bgMean: Optional[List[float]] = None
    bgOsc: Optional[List[float]] = None
    numRawLines: Optional[int] = None
    numResampLines: Optional[int] = None
    resampTraceB: Optional[List[float]] = None
    FDFlip: Optional[bool] = None
    wlLow: Optional[float] = None
    wlHigh: Optional[float] = None
    kcenter: Optional[float] = None
    kbandwidth: Optional[float] = None
    n: Optional[float] = None    
    
    class Config:
        arbitrary_types_allowed = True  # allows custom types in derived usage


        