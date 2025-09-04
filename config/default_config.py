# config/default_config.py
from pathlib import Path
from .schema import OCTConfig, CAOConfig, ISAMConfig, SavingOptions, MotionConfig
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAT_DIR = PROJECT_ROOT / "mat"

INFO = OCTConfig(
    root=PROJECT_ROOT,
    matDir=MAT_DIR,

    # Laser Parameters
    wlLow=1014e-6,
    wlHigh=1098e-6,
    kcenter=3.141592653589793*(1/1014e-6 + 1/1098e-6),
    kbandwidth=2*3.141592653589793*(1/1014e-6 - 1/1098e-6),
    FDFlip=True,
    n=1.328,

    # Fringe Processing Parameters
    numSamples=832,
    usedSamples=list(range(1, 324)),
    numUsedSamples=323,
    numFTSamples=1024,
    bgLineShift=4352*1,
    # bgLines=4352*3,

    # BG Subtraction Parameters
    adaptiveBG=True,
    adaptiveBGOsc=True,
    bgRef=[0.0]*323,

    # Image Processing Parameters
    # noiseFloor=8,
    # bgBW=30,

    # Scaling Factors
    depthPerPixel=(2*3.141592653589793/(2*3.141592653589793*(1/1014e-6 - 1/1098e-6))) * (323/1024) / 1.328 / 2,
    radPerPixel=(2*3.141592653589793) * ((3.141592653589793*(1/1014e-6 + 1/1098e-6)) / (2*3.141592653589793*(1/1014e-6 - 1/1098e-6))) * (323/1024),

    # Calibration Defaults
    # trigDelay=[402, 402, 402, 402],
    # trigDelayRange=10,
    numCalibLines=4352,
    
    # Calibration Windows (to be initialized later)
    spectralWindow=None,  # Replace at runtime with np.kaiser(...)
    mirrorBW=[50],
    noiseBW=list(range(125, 151)),

    # Figure positions
    figurePos=None,  # Replace at runtime with getFigurePositions(...)

    # Optional ONH placeholders
    ONHmask=None,
    cumPhaseX=None,
    cumPhaseY=None,
    # subdivFactors=[1, 1],
    centerX=torch.tensor([0]),
    centerY=torch.tensor([0]),    

    numScanLines=2864,
    numFlybackLines=1488,
    numScanFrames=2500,
    scanSize=[3, 3],
    subdivFactors=[6, 6],
    numImgLines=500,
    numImgFrames=500,

    initLineShift=4352 * 5 + 790,
    bgShift=4352 * 0 + 790,
    bgLines=4352 * 2,
    trigDelayRange=10,

    imgPixelsStart=1,
    numImgPixels=512,
    numEnFacePixels=512,
    bgBW=10,
    correctSaccades=True,
    excludeONH=False,

    noiseFloor=12,
    dBRange=40,

    dispCompParam=[-0.54, -0.26],
    autoDisp=False,

    stabilizePhaseRepeats=2,
    stabilizePhaseRevert=False,
    stabilizePhaseWrap=False,

    segmentFigureOn=False,
    segmentSigma=[20e-3, 20e-3, 10e-3],
    ILMshift=20,
    NFLminRange=10,
    NFLmaxRange=40,
    GCLprct=[75, 85],
    GCLmaxRange=25,
    segmentOnly=False,

    CAO=CAOConfig(),
    ISAM=ISAMConfig(),
    Save=SavingOptions(),
    Motion=MotionConfig()   

)