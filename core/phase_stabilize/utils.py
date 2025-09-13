import torch

def revert_fringes(fringes: torch.Tensor, phase_per_frame: torch.Tensor) -> torch.Tensor:
    """
    주어진 위상을 적용하여 프린지를 되돌립니다.
    ... (함수 내용) ...
    """
    # phase_per_frame을 (1, 1, F) 형태로 브로드캐스팅 가능하게 만듭니다.
    phase_reshaped = phase_per_frame.view(1, 1, -1)

    # 위상 보정 항 생성: exp(-1j * phase)
    # 여기서 부호는 위상을 '더하는' 것인지 '빼는' 것인지에 따라 달라집니다.
    # 일반적으로 위상 오차를 '제거'하므로 exp(-1j * error)를 곱합니다.
    correction = torch.exp(-1j * phase_reshaped)

    # 프린지에 위상 보정 적용
    fringes_corrected = fringes * correction

    return fringes_corrected