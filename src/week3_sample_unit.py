"""Step 2 · 3주차 추가 실험 — 표본 구성 방식에 따른 세 모델 (논문이 명시하지 않은 부분)

논문은 "2초 세그먼트마다 25차원 벡터 19개를 만든다"고만 쓰고 이를 모델에 어떻게 넣었는지 밝히지 않는다.
같은 5-fold(시행 단위, seed 42)에서 세 가지 구성을 비교한다.
  vote    윈도우 하나 = 표본 하나로 학습, 세그먼트의 19개 윈도우 점수를 평균해 판정 (본 실험의 기본)
  concat  세그먼트 하나 = 19 x 25 = 475차원 벡터 하나
  mean    세그먼트 하나 = 19개 윈도우 특징의 평균 25차원
결과: results/week3/sample_unit_cv.csv
"""
import os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import common as C
from models import MODEL_NAMES, make_model, scores
from week3_run import load_features, OUT

FT, lab, part = load_features(); y = LabelEncoder().fit(C.CLASSES).transform(lab)
rows = []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(len(FT)), lab)):
    for unit, (Za, Zb) in (("concat", (FT[a].reshape(len(a), -1), FT[b].reshape(len(b), -1))), ("mean", (FT[a].mean(1), FT[b].mean(1)))):
        sc = StandardScaler().fit(Za); Za_, Zb_ = sc.transform(Za), sc.transform(Zb)
        for name in MODEL_NAMES:
            m = make_model(name); t0 = time.perf_counter(); m.fit(Za_, y[a]); tt = time.perf_counter() - t0
            acc = float((scores(m, Zb_).argmax(1) == y[b]).mean())
            rows.append(dict(fold=k, unit=unit, model=name, acc=acc, train_time_s=tt))
            print(f"fold {k} {unit:6s} {name:7s} {acc*100:.2f}  ({tt:.2f}s)", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sample_unit_cv.csv"), index=False)
