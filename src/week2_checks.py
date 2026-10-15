"""Step 2 · 2주차 추가 검증 — 표본 단위와 녹음 파일 교란 (학습 1,536시행 안 5-fold, 테스트 384시행은 보지 않음)

 sample_unit.csv     윈도우 1개 / 세그먼트(19윈도우 확률 평균) / 세그먼트(475차원 연결) / 세그먼트(윈도우 평균 25차원)
 rest_confound.csv   동작 구간(1~3 s) vs 휴식 구간(4~5 s) 특징으로 8클래스·손동작 4종·각도 2종을 맞히는 정확도
 baseline_var.csv    휴식 구간 RMS 의 파일 간 / 파일 내 표준편차
 confound.png
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import common as C, features as F

OUT = os.path.join(C.ROOT, "results", "week2")
X, meta = C.load_all(); Xf = F.preprocess(X.astype(np.float64)).astype(np.float32)
strat = (meta.label + "_" + meta.participant).values
tr, _ = train_test_split(np.arange(len(meta)), test_size=0.2, stratify=strat, random_state=42); tr = np.sort(tr)   # week2_features.py 와 같은 분할
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, meta.label.values[tr]))
rf = lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def cv(FT, y, unit):
    acc = []
    for a, b in folds:
        A, B = FT[tr[a]], FT[tr[b]]; nw = A.shape[1]
        if unit in ("window", "vote"):
            s = StandardScaler().fit(A.reshape(-1, 25)); m = rf().fit(s.transform(A.reshape(-1, 25)), np.repeat(y[tr[a]], nw))
            if unit == "window": acc.append(m.score(s.transform(B.reshape(-1, 25)), np.repeat(y[tr[b]], nw)))
            else:
                p = m.predict_proba(s.transform(B.reshape(-1, 25))).reshape(len(b), nw, -1).mean(1); acc.append((m.classes_[p.argmax(1)] == y[tr[b]]).mean())
        else:
            ZA, ZB = (A.reshape(len(a), -1), B.reshape(len(b), -1)) if unit == "concat" else (A.mean(1), B.mean(1))
            s = StandardScaler().fit(ZA); acc.append(rf().fit(s.transform(ZA), y[tr[a]]).score(s.transform(ZB), y[tr[b]]))
    return float(np.mean(acc)), float(np.std(acc))

FT = F.trial_features(Xf); y8 = meta.label.values
rows = []
for unit, name in (("window", "윈도우 1개 (200 ms)"), ("vote", "세그먼트: 19윈도우 확률 평균"), ("concat", "세그먼트: 19×25 = 475차원 연결"), ("mean", "세그먼트: 윈도우 평균 25차원")):
    m_, s_ = cv(FT, y8, unit); rows.append(dict(unit=name, n_test_per_fold=f"{'5,837' if unit == 'window' else '307'}", acc=m_, std=s_)); print(name, f"{m_*100:.2f} ± {s_*100:.2f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sample_unit.csv"), index=False)

rows = []
for seg, st, n in (("동작 1.0~3.0 s", 1000, 2000), ("휴식 4.0~5.0 s", 4000, 1000)):
    R = F.trial_features(Xf, start=st, n=n)
    for tname, y, chance in (("8클래스", y8, 12.5), ("손동작 4종", meta.gesture.values, 25), ("각도 2종", meta.angle.values.astype(str), 50)):
        m_, s_ = cv(R, y, "vote"); rows.append(dict(segment=seg, target=tname, chance=chance, acc=m_, std=s_)); print(seg, tname, f"{m_*100:.2f}")
rc = pd.DataFrame(rows); rc.to_csv(os.path.join(OUT, "rest_confound.csv"), index=False)

base = np.sqrt((Xf[:, 4000:5000] ** 2).mean(1)); df = meta.assign(**{c: base[:, i] for i, c in enumerate(C.CHANNELS)})
bv = pd.DataFrame([dict(channel=c, between_files=df.groupby("file")[c].mean().std(), within_file=df.groupby("file")[c].std().mean()) for c in C.CHANNELS])
bv["ratio"] = bv.between_files / bv.within_file; bv.to_csv(os.path.join(OUT, "baseline_var.csv"), index=False); print(bv.round(5).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
tg = ["8클래스", "손동작 4종", "각도 2종"]; xs = np.arange(3)
for k, (seg, col) in enumerate((("동작 1.0~3.0 s", "C0"), ("휴식 4.0~5.0 s", "C3"))):
    v = rc[rc.segment == seg].set_index("target").loc[tg]
    ax[0].bar(xs + (k - 0.5) * 0.38, v.acc * 100, 0.38, color=col, label=["motion 1-3 s", "rest 4-5 s (hand relaxed)"][k])
    for x_, a_ in zip(xs + (k - 0.5) * 0.38, v.acc * 100): ax[0].text(x_, a_ + 1, f"{a_:.1f}", ha="center", fontsize=8)
ax[0].scatter(xs, [12.5, 25, 50], marker="_", s=900, color="k", label="chance", zorder=3)
ax[0].set_xticks(xs, ["8 classes", "gesture (4)", "angle (2)"]); ax[0].set_ylim(0, 108); ax[0].set_ylabel("segment accuracy (%)"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("RF, inner 5-fold on training trials", fontsize=10)
lab = df.groupby("file").FDS.mean().sort_values()
for i, (f_, v_) in enumerate(lab.items()):
    vals = df[df.file == f_].FDS.values; ax[1].plot([i] * len(vals), vals, ".", ms=2, color="0.6"); ax[1].plot(i, v_, "o", ms=3, color="C3")
ax[1].set_xlabel("64 recording files (sorted)"); ax[1].set_ylabel("FDS RMS at rest 4-5 s (mV)"); ax[1].set_xticks([])
ax[1].set_title(f"rest baseline differs between files ({bv.ratio.min():.1f}-{bv.ratio.max():.1f}x within-file SD)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "confound.png"), dpi=120); plt.close(fig)
