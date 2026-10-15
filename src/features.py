"""Step 2 공통 전처리·특징 (2주차 단계 1·3·4). 3주차도 이 모듈을 그대로 쓴다.

 preprocess   60 Hz 노치(Q 30) -> 20~450 Hz 4차 Butterworth, filtfilt(axis=0)
 cut          mark 기준 세그먼트 (기본 1.0~3.0 s = 2,000샘플, 1주차 활성 구간 측정 근거)
 make_windows 200 ms 윈도우, 100 ms 증분 -> 19개
 td_features  RMS·MAV·WL·ZC·SSC x 5채널 = 25차원 (논문 식 (1)~(5))
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

FS = 1000
SEG_START, SEG_LEN = 1000, 2000        # mark + 1.0 s 부터 2 s
WIN, HOP = 200, 100
ZC_TH = 0.01
FEATS = ["RMS", "MAV", "WL", "ZC", "SSC"]
CHANNELS = ["ECR", "ED", "FDS", "BB", "TB"]
FEATURE_NAMES = [f"{f}_{c}" for f in FEATS for c in CHANNELS]      # 순서: 특징 블록 x 채널


def preprocess(x, fs=FS):
    """(..., T, 5) 시간축 = axis -2. 시행 배열 (N, T, 5) 도 한 번에 처리한다."""
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=-2)
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-2)


def cut(x, start=SEG_START, n=SEG_LEN):
    return x[..., start:start + n, :]


def make_windows(x, win=WIN, hop=HOP):
    """(..., T, C) -> (..., n_win, win, C)"""
    n = (x.shape[-2] - win) // hop + 1
    return np.stack([x[..., i * hop:i * hop + win, :] for i in range(n)], axis=-3)


def td_features(w, th=ZC_TH):
    """w: (..., win, C) -> (..., 5*C). 마지막 축 순서 = [RMS x C, MAV x C, WL x C, ZC x C, SSC x C]"""
    d = np.diff(w, axis=-2)
    rms = np.sqrt(np.mean(w ** 2, axis=-2))
    mav = np.mean(np.abs(w), axis=-2)
    wl = np.sum(np.abs(d), axis=-2)
    zc = np.sum((w[..., :-1, :] * w[..., 1:, :] < 0) & (np.abs(d) >= th), axis=-2)            # 식 (4): 임계값 th
    ssc = np.sum((w[..., 1:-1, :] - w[..., :-2, :]) * (w[..., 1:-1, :] - w[..., 2:, :]) > 0, axis=-2)  # 식 (5): 임계값 없음
    return np.concatenate([rms, mav, wl, zc, ssc], axis=-1).astype(np.float32)


def trial_features(trials_filtered, start=SEG_START, n=SEG_LEN, win=WIN, hop=HOP, th=ZC_TH):
    """(N, 5000, 5) 필터된 시행 -> (N, n_win, 25)"""
    return td_features(make_windows(cut(trials_filtered, start, n), win, hop), th)
