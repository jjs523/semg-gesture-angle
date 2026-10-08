"""Step 2 공통: 파일 이름 해석, 시행 추출(mark 기준), 캐시, 무결성 해시.

데이터: TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset (논문 저자 공개)
  data/raw/data/<p>_<angle> <gesture>.csv  — 64개 (8명 x 2각도 x 4동작), 파일 하나 = 30시행 연속 녹음(약 151 s)
  열: bandpass 1~5 (이미 60 Hz 노치 + 20~450 Hz 필터), time, onset(파일당 1), mark(시행 시작, 파일당 30, 간격 5.0 s)
"""
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("SEMG2_DATA", os.path.join(ROOT, "data", "raw", "data"))
CACHE = os.path.join(ROOT, "data", "cache")
FS = 1000
TRIAL_LEN = 5000                      # mark 이후 5 s (휴식 2 + 동작 1 + 휴식 2)
CHANNELS = ["ECR", "ED", "FDS", "BB", "TB"]
CH_KO = ["요측수근신근", "총지신근", "천지굴근", "상완이두근", "상완삼두근"]
GESTURES = {"fist": "fist", "open": "open", "pro": "pronation", "extor": "supination"}   # 파일 표기 -> 논문 이름
GESTURE_ORDER = ["fist", "open", "pronation", "supination"]
ANGLES = [30, 60]
CLASSES = [f"{a}_{g}" for a in ANGLES for g in GESTURE_ORDER]                         # 8클래스, 논문 표기 30_fist ...
PARTICIPANTS = list("abcdefgh")
_NAME = re.compile(r"^([a-h])_(30|60) (fist|open|pro|extor)\.csv$")


def parse_name(path):
    m = _NAME.match(os.path.basename(path))
    if not m:
        raise ValueError(f"이름 규칙에 맞지 않음: {path}")
    p, a, g = m.groups()
    return dict(participant=p, angle=int(a), gesture=GESTURES[g], label=f"{a}_{GESTURES[g]}")


def list_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"CSV를 찾지 못했습니다: {DATA_DIR}  (bash setup.sh 로 데이터를 받으세요)")
    return files


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_file(path):
    """CSV -> (신호 (N,5) float32, time (N,), mark 인덱스 (30,), onset 인덱스)"""
    df = pd.read_csv(path)
    x = df[[f"bandpass {i}" for i in range(1, 6)]].to_numpy(dtype=np.float32)
    marks = np.flatnonzero(df["mark"].to_numpy() == 1)
    onset = np.flatnonzero(df["onset"].to_numpy() == 1)
    return x, df["time"].to_numpy(), marks, onset


def extract_trials(x, marks, n=TRIAL_LEN):
    """mark 마다 n 샘플을 잘라 (30, n, 5). 파일 끝을 넘는 mark 는 없음을 호출 쪽에서 확인한다."""
    return np.stack([x[m:m + n] for m in marks])


def load_all(use_cache=True):
    """전체 1,920 시행 -> X (1920, 5000, 5), meta DataFrame(file, participant, angle, gesture, label, trial, mark_sample)"""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, "trials.npz")
    if use_cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return z["X"], pd.DataFrame(list(z["meta"]))
    X, meta = [], []
    for f in list_files():
        info = parse_name(f)
        x, t, marks, _ = load_file(f)
        assert len(marks) == 30 and marks[-1] + TRIAL_LEN <= len(x), f
        X.append(extract_trials(x, marks))
        for k, m in enumerate(marks):
            meta.append(dict(file=os.path.basename(f), trial=k, mark_sample=int(m), **info))
    X = np.concatenate(X)
    meta = pd.DataFrame(meta)
    np.savez(cp, X=X, meta=np.array(meta.to_dict("records"), dtype=object))
    return X, meta


def rms_envelope(trials, bin_ms=50):
    """(T, n, 5) -> (T, n/bin, 5) 구간별 RMS"""
    T, n, c = trials.shape
    b = n // bin_ms
    return np.sqrt((trials[:, :b * bin_ms] ** 2).reshape(T, b, bin_ms, c).mean(2))
