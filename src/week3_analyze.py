"""Step 2 · 3주차 — 결과 집계 (강의 3주차 단계 2·3·4·5·6)

 results/week3/*.json -> 표·그림
   table_holdout.csv        세 모델 hold-out 4지표 (논문 재현 설정 475D / 강의 실습 설정 윈도우·세그먼트)
   table_cv.csv             5-fold 폴드별 값과 평균 ± 표준편차
   table_ttest.csv          대응표본 t-검정 3쌍
   table_loso.csv           피험자별 LOSO 정확도
   table_cost.csv           학습 시간·추론 시간·모델 크기
   table_participant.csv    hold-out 예측의 피험자별 정확도 (논문 Table 4 대응)
   error_types.csv          오분류를 '같은 동작·다른 각도 / 다른 동작·같은 각도 / 둘 다'로 분해
   confusion8.png           세 모델 8클래스 혼동행렬 (hold-out, 475D)  + confusion8_<model>.png
   confusion8_loso.png      LOSO 합산 혼동행렬 (새 사람)
   summary.json
"""
import glob, json, os
from itertools import combinations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import ttest_rel
import common as C
from models import MODEL_NAMES

R = os.path.join(C.ROOT, "results", "week3")
SHORT = ["30F", "30O", "30P", "30S", "60F", "60O", "60P", "60S"]
J = lambda n: json.load(open(os.path.join(R, f"{n}.json")))
folds = lambda pre, m, k: [J(f"{pre}_{m}_fold{i}") for i in range(k)]
P = lambda v: f"{v*100:.2f}"
ms = lambda v: f"{np.mean(v)*100:.2f} ± {np.std(v)*100:.2f}"

# ------------------------------------------------------------ hold-out 4지표
rows = []
for m in MODEL_NAMES:
    c = J(f"concat_holdout_{m}")["segment"]; w = J(f"holdout_{m}")
    rows.append(dict(model=m, setting="논문 재현 (세그먼트 475D)", accuracy=c["accuracy"], precision=c["precision"], recall=c["recall"], f1=c["f1"]))
    rows.append(dict(model=m, setting="강의 실습 (윈도우 25D, 윈도우 단위)", **{k: w["window"][k] for k in ("accuracy", "precision", "recall", "f1")}))
    rows.append(dict(model=m, setting="강의 실습 (윈도우 25D, 19윈도우 평균 판정)", **{k: w["segment"][k] for k in ("accuracy", "precision", "recall", "f1")}))
ho = pd.DataFrame(rows); ho.to_csv(os.path.join(R, "table_holdout.csv"), index=False)
print(ho.assign(**{k: ho[k] * 100 for k in ("accuracy", "precision", "recall", "f1")}).round(2).to_string(index=False))

# ------------------------------------------------------------ 5-fold + t-test
cv = {}
for m in MODEL_NAMES:
    cv[("concat", m)] = [f["segment"]["accuracy"] for f in folds("concat_cv", m, 5)]
    W = folds("cv", m, 5)
    cv[("window", m)] = [f["window"]["accuracy"] for f in W]; cv[("vote", m)] = [f["segment"]["accuracy"] for f in W]
    cv[("concat_f1", m)] = [f["segment"]["f1"] for f in folds("concat_cv", m, 5)]
crow = []
for setting, key in (("논문 재현 (세그먼트 475D)", "concat"), ("윈도우 단위", "window"), ("19윈도우 평균 판정", "vote")):
    for m in MODEL_NAMES:
        v = cv[(key, m)]; crow.append(dict(setting=setting, model=m, **{f"fold{i+1}": v[i] for i in range(5)}, mean=np.mean(v), std=np.std(v)))
cvt = pd.DataFrame(crow); cvt.to_csv(os.path.join(R, "table_cv.csv"), index=False)
print(cvt.assign(**{c: cvt[c] * 100 for c in cvt.columns if c.startswith("fold") or c in ("mean", "std")}).round(2).to_string(index=False))
trow = []
for setting, key in (("논문 재현 (세그먼트 475D)", "concat"), ("윈도우 단위", "window"), ("19윈도우 평균 판정", "vote")):
    for a, b in combinations(MODEL_NAMES, 2):
        t, p = ttest_rel(cv[(key, a)], cv[(key, b)])
        trow.append(dict(setting=setting, pair=f"{a} vs {b}", diff_mean=np.mean(np.array(cv[(key, a)]) - np.array(cv[(key, b)])), t=t, p=p, significant=bool(p < 0.05)))
tt = pd.DataFrame(trow); tt.to_csv(os.path.join(R, "table_ttest.csv"), index=False); print(tt.round(4).to_string(index=False))

# ------------------------------------------------------------ LOSO
lrow = []
for p_i, p in enumerate(C.PARTICIPANTS):
    r = {"participant": p}
    for m in MODEL_NAMES:
        r[f"{m} (475D)"] = J(f"concat_loso_{m}_fold{p_i}")["segment"]["accuracy"]
        r[f"{m} (윈도우)"] = J(f"loso_{m}_fold{p_i}")["window"]["accuracy"]
    lrow.append(r)
lo = pd.DataFrame(lrow); lo.loc["평균"] = ["평균"] + lo.iloc[:, 1:].mean().tolist(); lo.loc["표준편차"] = ["표준편차"] + lo.iloc[:8, 1:].std(ddof=0).tolist()
lo.to_csv(os.path.join(R, "table_loso.csv"), index=False); print(lo.round(4).to_string(index=False))
ltt = []
for a, b in combinations(MODEL_NAMES, 2):
    va = [J(f"concat_loso_{a}_fold{i}")["segment"]["accuracy"] for i in range(8)]; vb = [J(f"concat_loso_{b}_fold{i}")["segment"]["accuracy"] for i in range(8)]
    t, p = ttest_rel(va, vb); ltt.append(dict(setting="LOSO (475D, 8명)", pair=f"{a} vs {b}", diff_mean=np.mean(np.array(va) - np.array(vb)), t=t, p=p, significant=bool(p < 0.05)))
pd.concat([tt, pd.DataFrame(ltt)]).to_csv(os.path.join(R, "table_ttest.csv"), index=False); print(pd.DataFrame(ltt).round(4).to_string(index=False))

# ------------------------------------------------------------ 계산 비용
crow = []
for m in MODEL_NAMES:
    c = J(f"concat_holdout_{m}"); w = J(f"holdout_{m}")
    crow.append(dict(model=m, train_475=c["train_time_s"], latency_475=c["latency_ms_per_segment"], batch_475=c["batch_ms_per_segment"], size_475=c["model_size_mb"],
                     train_win=w["train_time_s"], latency_win=w["latency_ms_per_window"], size_win=w["model_size_mb"]))
cost = pd.DataFrame(crow); cost.to_csv(os.path.join(R, "table_cost.csv"), index=False); print(cost.round(3).to_string(index=False))

# ------------------------------------------------------------ 피험자별 (hold-out 475D)
prow = []
for m in MODEL_NAMES:
    c = J(f"concat_holdout_{m}"); df = pd.DataFrame(dict(p=c["seg_participant"], ok=np.array(c["seg_pred"]) == np.array(c["seg_true"])))
    prow.append(df.groupby("p").ok.mean().rename(m))
pt = pd.concat(prow, axis=1); pt.to_csv(os.path.join(R, "table_participant.csv")); print(pt.mul(100).round(1).T.to_string())

# ------------------------------------------------------------ 오분류 유형
def err_types(cm):
    cm = np.array(cm); same_g = diff_g = both = 0
    for i in range(8):
        for j in range(8):
            if i == j or cm[i, j] == 0: continue
            gi, gj, ai, aj = i % 4, j % 4, i // 4, j // 4
            if gi == gj: same_g += cm[i, j]
            elif ai == aj: diff_g += cm[i, j]
            else: both += cm[i, j]
    tot = cm.sum() - np.trace(cm)
    return dict(total_errors=int(tot), angle_only=int(same_g), gesture_only=int(diff_g), both=int(both))
erow = []
for m in MODEL_NAMES:
    erow.append(dict(eval="hold-out (475D)", model=m, **err_types(J(f"concat_holdout_{m}")["segment"]["confusion"])))
    cm_lo = sum(np.array(J(f"concat_loso_{m}_fold{i}")["segment"]["confusion"]) for i in range(8))
    erow.append(dict(eval="LOSO 합산 (475D)", model=m, **err_types(cm_lo)))
et = pd.DataFrame(erow); et.to_csv(os.path.join(R, "error_types.csv"), index=False); print(et.to_string(index=False))

# ------------------------------------------------------------ 혼동행렬 그림
def max_off(cm):
    c = np.array(cm).copy(); np.fill_diagonal(c, 0); i, j = np.unravel_index(c.argmax(), c.shape); return int(i), int(j), int(c[i, j])
def draw(ax, cm, title):
    cm = np.array(cm); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(8), SHORT, rotation=45, fontsize=8); ax.set_yticks(range(8), SHORT, fontsize=8)
    for i in range(8):
        for j in range(8):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    i, j, n = max_off(cm); ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red", lw=2))
    ax.axhline(3.5, color="0.5", lw=0.6); ax.axvline(3.5, color="0.5", lw=0.6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title}\nmax error {SHORT[i]}->{SHORT[j]} = {n}", fontsize=9)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, m in zip(axes, MODEL_NAMES):
    c = J(f"concat_holdout_{m}")["segment"]; draw(ax, c["confusion"], f"{m}  hold-out acc {P(c['accuracy'])} %")
    f1, a1 = plt.subplots(figsize=(5.6, 5)); draw(a1, c["confusion"], f"{m}  hold-out acc {P(c['accuracy'])} %"); f1.tight_layout(); f1.savefig(os.path.join(R, f"confusion8_{m}.png"), dpi=120); plt.close(f1)
fig.suptitle("8-class confusion matrices, hold-out 384 test segments (paper setting, 475-D). F=fist O=open P=pronation S=supination", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(R, "confusion8.png"), dpi=120); plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, m in zip(axes, MODEL_NAMES):
    cm = sum(np.array(J(f"concat_loso_{m}_fold{i}")["segment"]["confusion"]) for i in range(8))
    draw(ax, cm, f"{m}  LOSO (8 unseen people) acc {np.trace(cm)/cm.sum()*100:.2f} %")
fig.suptitle("LOSO, summed over 8 held-out participants (1,920 segments)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(R, "confusion8_loso.png"), dpi=120); plt.close(fig)

# 정확도 비교 그림: trial-wise vs LOSO
fig, ax = plt.subplots(figsize=(8.5, 3.8)); x = np.arange(3); wbar = 0.26
for k, (lab_, vals, col) in enumerate((("5-fold trial-wise (paper setting)", [np.mean(cv[("concat", m)]) for m in MODEL_NAMES], "C0"),
                                      ("hold-out trial-wise", [J(f"concat_holdout_{m}")["segment"]["accuracy"] for m in MODEL_NAMES], "C2"),
                                      ("LOSO (new person)", [lo.loc["평균", f"{m} (475D)"] for m in MODEL_NAMES], "C3"))):
    ax.bar(x + (k - 1) * wbar, np.array(vals) * 100, wbar, color=col, label=lab_)
    for xi, v in zip(x + (k - 1) * wbar, vals): ax.text(xi, v * 100 + 1, f"{v*100:.1f}", ha="center", fontsize=8)
ax.axhline(12.5, color="k", ls=":", lw=1); ax.text(2.45, 14, "chance 12.5", fontsize=8, ha="right")
ax.set_xticks(x, MODEL_NAMES); ax.set_ylim(0, 122); ax.set_ylabel("segment accuracy (%)"); ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
ax.set_title("same models, same features: seen people vs unseen people", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(R, "trialwise_vs_loso.png"), dpi=120); plt.close(fig)

json.dump(dict(cv={f"{k[0]}:{k[1]}": v for k, v in cv.items()}), open(os.path.join(R, "summary.json"), "w"), indent=1)
print("저장:", R)
