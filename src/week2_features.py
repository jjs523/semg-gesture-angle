"""Step 2 · 2주차 — 전처리와 특징 추출 (단계 1~6 + 설정 검증)

 단계 1 필터            60 Hz 노치 + 20~450 Hz BPF, 전후 Welch 스펙트럼      -> filter_check.png (제출물 ①)
 단계 2 세그먼트        mark + 1.0~3.0 s, (2000, 5)                            -> segment_windows.png
 단계 3 윈도우          200 ms / 100 ms -> (19, 200, 5)                         (제출물 ②)
 단계 4 특징            td_features -> (25,)                                     (제출물 ③, src/features.py)
 단계 5 데이터셋        시행 단위로 먼저 8:2 분할 -> 각 집합 안에서만 윈도우·특징   (제출물 ④)
 단계 6 표준화·저장     StandardScaler 를 학습셋에만 fit -> data/dataset.npz      (제출물 ⑤)
 설정 검증 (제출물 ⑥ 근거, 학습 1,536시행 안에서만 — 테스트 384시행은 보지 않음)
   segment_start.csv / window_length.csv  RF 5-fold(시행 단위)로 세그먼트 시작점·윈도우 길이·오버랩 비교
   leakage.csv                            윈도우 단위로 섞어 나누면 정확도가 얼마나 부풀려지는가
실행: python src/week2_features.py
"""
import json, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import common as C
import features as F

OUT = os.path.join(C.ROOT, "results", "week2"); os.makedirs(OUT, exist_ok=True)
DS = os.path.join(C.ROOT, "data"); SEED = 42
t_all = time.time()
X, meta = C.load_all()                                     # (1920, 5000, 5) 원본(저장소 배포본, 이미 필터됨)
facts = {}

# ------------------------------------------------------------ 단계 1: 필터
Xf = F.preprocess(X.astype(np.float64)).astype(np.float32)
i0 = meta.index[(meta.participant == "a") & (meta.label == "30_fist") & (meta.trial == 0)][0]
print(f"[단계 1] 표준편차 (a, 30_fist, 시행 1, 채널별) 전 {X[i0].std(0).round(4)} -> 후 {Xf[i0].std(0).round(4)}")
facts["std_before"] = X[i0].std(0).round(5).tolist(); facts["std_after"] = Xf[i0].std(0).round(5).tolist()
f, Pb = welch(X, fs=F.FS, nperseg=1024, axis=1); _, Pa = welch(Xf, fs=F.FS, nperseg=1024, axis=1)
Pb_m, Pa_m = Pb.mean(0), Pa.mean(0)                        # (freq, 5) 1,920시행 평균
k60 = np.argmin(abs(f - 60)); k10 = f < 15; k470 = f > 460
facts["p60_ratio"] = (Pa_m[k60] / Pb_m[k60]).round(3).tolist()
facts["below15_ratio"] = (Pa_m[k10].sum(0) / Pb_m[k10].sum(0)).round(3).tolist()
facts["above460_ratio"] = (Pa_m[k470].sum(0) / Pb_m[k470].sum(0)).round(3).tolist()
facts["power_kept"] = (Pa_m.sum(0) / Pb_m.sum(0)).round(4).tolist()
print(f"         60 Hz 전력 후/전 {facts['p60_ratio']}  15 Hz 미만 {facts['below15_ratio']}  460 Hz 초과 {facts['above460_ratio']}  전체 보존 {facts['power_kept']}")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for ch in range(5):
    ax[0].semilogy(f, Pb_m[:, ch], lw=0.9, label=f"ch{ch+1} {C.CHANNELS[ch]}")
    ax[1].semilogy(f, Pa_m[:, ch], lw=0.9)
for a, t in ((ax[0], "before (dataset as distributed)"), (ax[1], "after notch 60 + BPF 20-450 Hz")):
    a.set_title(f"{t}, mean of 1,920 trials", fontsize=10); a.set_xlabel("Hz"); a.set_xlim(0, 500)
    for v in (20, 60, 450): a.axvline(v, color="r" if v == 60 else "0.5", ls="--", lw=0.8)
ax[0].set_ylabel("PSD (mV²/Hz)"); ax[0].legend(fontsize=7)
ax[2].plot(f, Pa_m.mean(1) / Pb_m.mean(1), color="k", lw=1)
ax[2].set_title("after / before (5-channel mean)", fontsize=10); ax[2].set_xlabel("Hz"); ax[2].set_xlim(0, 500); ax[2].set_ylim(0, 1.2)
for v in (20, 60, 450): ax[2].axvline(v, color="r" if v == 60 else "0.5", ls="--", lw=0.8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "filter_check.png"), dpi=120); plt.close(fig)

# ------------------------------------------------------------ 단계 2·3: 세그먼트·윈도우
seg = F.cut(Xf[i0]); w = F.make_windows(seg)
print(f"[단계 2] 세그먼트 모양 {seg.shape}  (mark + {F.SEG_START/1000:.1f}~{(F.SEG_START+F.SEG_LEN)/1000:.1f} s)")
print(f"[단계 3] 윈도우 배열 {w.shape}  ((2000-200)/100+1 = {(F.SEG_LEN - F.WIN)//F.HOP + 1})")
facts.update(segment_shape=list(seg.shape), window_shape=list(w.shape))
t = np.arange(5000) / 1000
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
for ch in range(5): ax[0].plot(t, Xf[i0, :, ch] + ch * 0.6, lw=0.5, label=f"ch{ch+1} {C.CHANNELS[ch]}")
ax[0].axvspan(F.SEG_START/1000, (F.SEG_START+F.SEG_LEN)/1000, color="green", alpha=0.08)
ax[0].set_yticks([]); ax[0].set_title("filtered trial (a, 30_fist, 1) — green = 2 s segment, mark + 1.0~3.0 s", fontsize=10)
for i in range(19):
    s = (F.SEG_START + i * F.HOP) / 1000
    ax[1].add_patch(plt.Rectangle((s, i % 2), F.WIN / 1000, 0.8, color=f"C{i%10}", alpha=0.6))
ax[1].set_ylim(0, 2); ax[1].set_yticks([]); ax[1].set_xlabel("s from mark"); ax[1].set_title("19 windows of 200 ms, step 100 ms (50 % overlap)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "segment_windows.png"), dpi=120); plt.close(fig)

# ------------------------------------------------------------ 단계 4: 특징
fv = F.td_features(w[0])
print(f"[단계 4] 특징 벡터 {fv.shape}  첫 윈도우: " + " ".join(f"{n}={v:.4g}" for n, v in zip(F.FEATURE_NAMES[::5], fv[::5])))
facts["feature_shape"] = list(fv.shape)
FT = F.trial_features(Xf)                                    # (1920, 19, 25) — 시행별로 따로 계산되므로 분할 순서와 무관
zc = FT[..., 15:20]; ssc = FT[..., 20:25]
facts["zc_mean_per_ch"] = zc.mean((0, 1)).round(2).tolist(); facts["ssc_mean_per_ch"] = ssc.mean((0, 1)).round(2).tolist()
zc0 = F.td_features(F.make_windows(F.cut(Xf[:200])), th=0.0)[..., 15:20]
facts["zc_th0_mean_per_ch"] = zc0.mean((0, 1)).round(2).tolist()
facts["zc_zero_frac"] = float((zc == 0).mean().round(4))
print(f"         ZC 평균(th 0.01) {facts['zc_mean_per_ch']}  vs th 0 {facts['zc_th0_mean_per_ch']}  ZC=0 비율 {facts['zc_zero_frac']}")
print(f"         SSC 평균 {facts['ssc_mean_per_ch']}")

# 클래스별 특징 평균 (z-score) 히트맵
mz = pd.DataFrame(FT.reshape(-1, 25), columns=F.FEATURE_NAMES).assign(label=np.repeat(meta.label.values, 19)).groupby("label").mean().reindex(C.CLASSES)
mz = (mz - mz.mean()) / mz.std()
fig, ax = plt.subplots(figsize=(12, 3.8)); im = ax.imshow(mz.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_yticks(range(8), mz.index); ax.set_xticks(range(25), F.FEATURE_NAMES, rotation=90, fontsize=8)
for b in (4.5, 9.5, 14.5, 19.5): ax.axvline(b, color="k", lw=0.8)
ax.set_title("class mean of each feature (z-score across classes)", fontsize=10); fig.colorbar(im, fraction=0.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_class_heatmap.png"), dpi=120); plt.close(fig)

# ------------------------------------------------------------ 단계 5: 시행 단위 분할 -> 각 집합 안에서만 특징
strat = (meta.label + "_" + meta.participant).values           # 클래스 x 피험자 층화 (클래스 층화를 포함)
tr_idx, te_idx = train_test_split(np.arange(len(meta)), test_size=0.2, stratify=strat, random_state=SEED)
tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
assert not set(tr_idx) & set(te_idx), "같은 시행이 학습·테스트 양쪽에 있음"
def build(idx):
    Xw = F.trial_features(Xf[idx])                               # 이 집합의 시행에서만 윈도우·특징 생성
    y = np.repeat(meta.label.values[idx], Xw.shape[1])
    tid = np.repeat(idx, Xw.shape[1]); pos = np.tile(np.arange(Xw.shape[1]), len(idx))
    return Xw.reshape(-1, 25), y, tid, pos
Xtr, ytr, tid_tr, pos_tr = build(tr_idx); Xte, yte, tid_te, pos_te = build(te_idx)
assert not set(tid_tr) & set(tid_te)
print(f"[단계 5] 시행 학습 {len(tr_idx)} / 테스트 {len(te_idx)}  ->  윈도우 {Xtr.shape} / {Xte.shape}  (비율 {len(Xtr)/len(Xte):.2f})")
summ = pd.DataFrame({"학습 시행": pd.Series(meta.label.values[tr_idx]).value_counts(), "테스트 시행": pd.Series(meta.label.values[te_idx]).value_counts()}).reindex(C.CLASSES)
summ["학습 윈도우"] = summ["학습 시행"] * 19; summ["테스트 윈도우"] = summ["테스트 시행"] * 19
summ.loc["합계"] = summ.sum(); summ.to_csv(os.path.join(OUT, "dataset_summary.csv"))
pp = pd.DataFrame({"학습": pd.Series(meta.participant.values[tr_idx]).value_counts(), "테스트": pd.Series(meta.participant.values[te_idx]).value_counts()}).sort_index()
pp.to_csv(os.path.join(OUT, "dataset_by_participant.csv"))
print(summ.to_string())
json.dump(dict(seed=SEED, test_size=0.2, stratify="label x participant",
               train_trials=meta.iloc[tr_idx][["file", "trial", "mark_sample", "participant", "label"]].to_dict("records"),
               test_trials=meta.iloc[te_idx][["file", "trial", "mark_sample", "participant", "label"]].to_dict("records")),
          open(os.path.join(OUT, "split.json"), "w"), ensure_ascii=False)

# ------------------------------------------------------------ 단계 6: 표준화(학습셋에만 fit) + 저장
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
facts.update(train_mean_absmax=float(abs(Xtr_s.mean(0)).max()), train_std_min=float(Xtr_s.std(0).min()), train_std_max=float(Xtr_s.std(0).max()),
             test_mean_absmax=float(abs(Xte_s.mean(0)).max()), test_std_range=[float(Xte_s.std(0).min()), float(Xte_s.std(0).max())])
print(f"[단계 6] 학습셋 평균 |최대| {facts['train_mean_absmax']:.2e}, 표준편차 {facts['train_std_min']:.4f}~{facts['train_std_max']:.4f}  |  "
      f"테스트셋 평균 |최대| {facts['test_mean_absmax']:.3f}, 표준편차 {facts['test_std_range'][0]:.3f}~{facts['test_std_range'][1]:.3f}")
np.savez_compressed(os.path.join(DS, "dataset.npz"), Xtr=Xtr_s, ytr=ytr, Xte=Xte_s, yte=yte, trial_tr=tid_tr, trial_te=tid_te, pos_tr=pos_tr, pos_te=pos_te,
                    feature_names=np.array(F.FEATURE_NAMES), classes=np.array(C.CLASSES), scaler_mean=sc.mean_, scaler_scale=sc.scale_,
                    participant_tr=meta.participant.values[tid_tr], participant_te=meta.participant.values[tid_te])
np.savez_compressed(os.path.join(DS, "features_all.npz"), F=FT, label=meta.label.values, participant=meta.participant.values,
                    feature_names=np.array(F.FEATURE_NAMES))      # 3주차 5-fold 용 (표준화 전, 폴드마다 학습 폴드에 fit)
print(f"         저장: data/dataset.npz ({os.path.getsize(os.path.join(DS, 'dataset.npz'))//1024} KB), data/features_all.npz")

# ------------------------------------------------------------ 설정 검증 (학습 1,536시행 안에서만)
def rf(): return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
def inner_cv(feat_fn, idx=tr_idx, k=5):
    """시행 단위 StratifiedKFold. feat_fn(시행 인덱스) -> (n, n_win, 25). 폴드마다 학습 폴드에 scaler fit."""
    accs = []
    for a, b in StratifiedKFold(k, shuffle=True, random_state=0).split(idx, meta.label.values[idx]):
        A, B = feat_fn(idx[a]), feat_fn(idx[b])
        ya, yb = np.repeat(meta.label.values[idx[a]], A.shape[1]), np.repeat(meta.label.values[idx[b]], B.shape[1])
        s = StandardScaler().fit(A.reshape(-1, 25)); m = rf().fit(s.transform(A.reshape(-1, 25)), ya)
        accs.append(m.score(s.transform(B.reshape(-1, 25)), yb))
    return float(np.mean(accs)), float(np.std(accs))

rows = []
for st in (0, 500, 1000, 1500, 2000, 2500):
    m_, s_ = inner_cv(lambda ii: F.trial_features(Xf[ii], start=st)); rows.append(dict(start_s=st / 1000, end_s=(st + 2000) / 1000, acc=m_, std=s_))
    print(f"   세그먼트 {st/1000:.1f}~{(st+2000)/1000:.1f} s: {m_*100:.2f} ± {s_*100:.2f}")
seg_df = pd.DataFrame(rows); seg_df.to_csv(os.path.join(OUT, "segment_start.csv"), index=False)

rows = []
for win, hop in ((100, 50), (200, 50), (200, 100), (200, 200), (300, 150), (400, 200)):
    m_, s_ = inner_cv(lambda ii: F.trial_features(Xf[ii], win=win, hop=hop))
    rows.append(dict(win_ms=win, hop_ms=hop, overlap=f"{(1-hop/win)*100:.0f} %", n_windows=(2000 - win)//hop + 1, acc=m_, std=s_))
    print(f"   윈도우 {win} ms / 증분 {hop} ms ({rows[-1]['n_windows']}개): {m_*100:.2f} ± {s_*100:.2f}")
win_df = pd.DataFrame(rows); win_df.to_csv(os.path.join(OUT, "window_length.csv"), index=False)

# 누수 실험: 같은 학습 1,536시행을 (a) 시행 단위 (b) 윈도우 단위로 섞어 5-fold
FTtr = FT[tr_idx]; ytr_w = np.repeat(meta.label.values[tr_idx], 19)
acc_trial = inner_cv(lambda ii: FT[ii])
acc_win = []
flat = FTtr.reshape(-1, 25)
for a, b in KFold(5, shuffle=True, random_state=0).split(flat):
    s = StandardScaler().fit(flat[a]); acc_win.append(rf().fit(s.transform(flat[a]), ytr_w[a]).score(s.transform(flat[b]), ytr_w[b]))
leak = pd.DataFrame([dict(split="시행 단위 (올바름)", acc=acc_trial[0], std=acc_trial[1]), dict(split="윈도우 단위 (누수)", acc=float(np.mean(acc_win)), std=float(np.std(acc_win)))])
leak.to_csv(os.path.join(OUT, "leakage.csv"), index=False); print(leak.to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].errorbar(seg_df.start_s, seg_df.acc * 100, yerr=seg_df["std"] * 100, marker="o", capsize=3)
ax[0].axvline(1.0, color="green", ls="--", lw=1); ax[0].set_xlabel("segment start (s from mark), length 2 s"); ax[0].set_ylabel("RF accuracy (%)")
ax[0].set_title("segment start (inner 5-fold, training trials only)", fontsize=9)
lab = [f"{r.win_ms}/{r.hop_ms}\n({r.n_windows})" for r in win_df.itertuples()]
ax[1].bar(range(len(win_df)), win_df.acc * 100, yerr=win_df["std"] * 100, capsize=3, color=["C0" if (r.win_ms, r.hop_ms) != (200, 100) else "C2" for r in win_df.itertuples()])
ax[1].set_xticks(range(len(win_df)), lab, fontsize=8); ax[1].set_ylim(min(win_df.acc * 100) - 5, 100); ax[1].set_xlabel("window / step ms (windows per segment)")
ax[1].set_title("window length & overlap (green = paper)", fontsize=9)
ax[2].bar([0, 1], leak.acc * 100, yerr=leak["std"] * 100, capsize=3, color=["C2", "C3"]); ax[2].set_xticks([0, 1], ["trial-wise split", "window-wise split\n(leakage)"])
ax[2].set_ylim(min(leak.acc * 100) - 5, 100); ax[2].set_title("same 1,536 trials, RF 5-fold", fontsize=9)
for i, v in enumerate(leak.acc * 100): ax[2].text(i, v + 0.5, f"{v:.2f}", ha="center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "settings_check.png"), dpi=120); plt.close(fig)

facts.update(n_train_trials=int(len(tr_idx)), n_test_trials=int(len(te_idx)), Xtr=list(Xtr.shape), Xte=list(Xte.shape), elapsed_s=round(time.time() - t_all, 1))
json.dump(facts, open(os.path.join(OUT, "week2_facts.json"), "w"), ensure_ascii=False, indent=1)
print(f"완료 {facts['elapsed_s']} s  ->  {OUT}")
