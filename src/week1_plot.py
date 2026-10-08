"""1주차 단계 4 — 채널별 신호 그리기 + 동작 구간 찾기 + 30° vs 60° 비교 (제출물 ③·④)

 출력 (results/week1/):
   ch5_example.png        한 시행의 5채널 파형 (강의 가정 2~3 s 와 실제 활성 구간 표시)
   envelope_by_class.png  8클래스 평균 RMS 포락선 (피험자 a) — 동작 구간이 어디인지
   envelope_all.png       8명 평균 포락선 (클래스별) — 피험자에 무관한 동작 시점
   motion_window.csv      클래스별 활성 시작·끝 (시행별 임계값 기준의 중앙값) -> 2주차 세그먼트 시작점 근거
   angle_compare.png      동작 4종 x 채널 5개, 30° vs 60° 평균 포락선 (8명 평균)
   angle_rms_table.csv    동작·채널별 30°/60° 활성 구간 RMS, 차이(%), 피험자 8명 대응표본 t-검정
   angle_per_participant.csv  피험자별 30°/60° RMS (BB·TB)
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import common as C

OUT = os.path.join(C.ROOT, "results", "week1"); os.makedirs(OUT, exist_ok=True)
X, meta = C.load_all()
BIN = 50                                   # ms
env = C.rms_envelope(X, BIN)               # (1920, 100, 5)
tt = (np.arange(env.shape[1]) + 0.5) * BIN / 1000
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ------------------------------------------------------------ 1) 단일 시행 5채널
sel = meta.index[(meta.participant == "a") & (meta.label == "30_fist") & (meta.trial == 0)][0]
x = X[sel]; t = np.arange(C.TRIAL_LEN) / C.FS
fig, ax = plt.subplots(5, 1, figsize=(10, 7.5), sharex=True)
for ch in range(5):
    ax[ch].plot(t, x[:, ch], lw=0.5, color=COL[ch])
    ax[ch].set_ylabel(f"ch{ch + 1} {C.CHANNELS[ch]}\n(mV)", fontsize=8)
    ax[ch].axvspan(2, 3, color="red", alpha=0.08)
    for b in (2, 3): ax[ch].axvline(b, color="r", ls="--", lw=0.8)
    ax[ch].axvspan(1, 3, color="green", alpha=0.06)
ax[0].set_title(f"participant a, 30_fist, trial 1  (red: lecture motion 2-3 s, green: observed activity 1-3 s)", fontsize=10)
ax[4].set_xlabel("time from mark (s)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ch5_example.png"), dpi=120); plt.close(fig)

# ------------------------------------------------------------ 2) 클래스별 평균 포락선 (피험자 a / 8명 평균)
def env_grid(mask_fn, title, fname):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for k, c in enumerate(C.CLASSES):
        a = axes[k // 4, k % 4]
        m = mask_fn(c)
        e = env[m].mean(0)
        for ch in range(5): a.plot(tt, e[:, ch], color=COL[ch], lw=1.2, label=f"ch{ch + 1} {C.CHANNELS[ch]}")
        for b in (2, 3): a.axvline(b, color="r", ls="--", lw=0.8)
        a.set_title(f"{c}  (n={int(m.sum())})", fontsize=9); a.set_ylim(0, None)
        if k % 4 == 0: a.set_ylabel("RMS (mV, 50 ms bins)")
        if k >= 4: a.set_xlabel("s from mark")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=120); plt.close(fig)
env_grid(lambda c: ((meta.label == c) & (meta.participant == "a")).values, "mean RMS envelope per class, participant a (red = lecture 2-3 s)", "envelope_by_class.png")
env_grid(lambda c: (meta.label == c).values, "mean RMS envelope per class, all 8 participants (red = lecture 2-3 s)", "envelope_all.png")

# ------------------------------------------------------------ 3) 활성 구간: 시행별 임계값 (기저선 = 4~5 s 휴식)
tot = env.sum(2)                                              # (1920, 100) 5채널 합
base_m, base_s = tot[:, 80:].mean(1, keepdims=True), tot[:, 80:].std(1, keepdims=True)
active = tot > base_m + 3 * base_s
on, off, peak = [], [], []
for i in range(len(tot)):
    idx = np.flatnonzero(active[i, :80])
    on.append(tt[idx[0]] if len(idx) else np.nan); off.append(tt[idx[-1]] if len(idx) else np.nan); peak.append(tt[tot[i].argmax()])
meta2 = meta.assign(onset_s=on, offset_s=off, peak_s=peak)
mw = meta2.groupby("label")[["onset_s", "peak_s", "offset_s"]].median().reindex(C.CLASSES).round(2)
mw.loc["전체"] = meta2[["onset_s", "peak_s", "offset_s"]].median().round(2)
q = meta2[["onset_s", "offset_s"]].quantile([0.1, 0.9]).round(2)
mw.to_csv(os.path.join(OUT, "motion_window.csv"))
print("활성 구간 중앙값 (s from mark):\n", mw.to_string())
print(f"10~90 % 분위: 시작 {q.loc[0.1, 'onset_s']}~{q.loc[0.9, 'onset_s']}, 끝 {q.loc[0.1, 'offset_s']}~{q.loc[0.9, 'offset_s']}")
seg = dict(start_s=1.0, end_s=3.0, reason="활성 시작 중앙값·끝 중앙값을 덮는 2 s 구간")

# ------------------------------------------------------------ 4) 30° vs 60° 비교 (제출물 ④)
win = slice(int(seg["start_s"] * 1000), int(seg["end_s"] * 1000))
rms_tr = np.sqrt((X[:, win] ** 2).mean(1))                   # (1920, 5) 활성 구간 RMS
m3 = meta.assign(**{C.CHANNELS[ch]: rms_tr[:, ch] for ch in range(5)})
pp = m3.groupby(["gesture", "angle", "participant"])[C.CHANNELS].mean().reset_index()   # 피험자별 평균 (각 30시행)
rows = []
for g in C.GESTURE_ORDER:
    for ch in C.CHANNELS:
        a30 = pp[(pp.gesture == g) & (pp.angle == 30)].sort_values("participant")[ch].values
        a60 = pp[(pp.gesture == g) & (pp.angle == 60)].sort_values("participant")[ch].values
        t_, p_ = stats.ttest_rel(a60, a30)
        n_up = int((a60 > a30).sum())
        rows.append(dict(gesture=g, channel=ch, rms30=a30.mean(), rms60=a60.mean(), diff_pct=(a60.mean() / a30.mean() - 1) * 100,
                         n_participants_60_higher=n_up, t=t_, p=p_))
at = pd.DataFrame(rows)
at.to_csv(os.path.join(OUT, "angle_rms_table.csv"), index=False)
print("30° vs 60° (활성 구간 1~3 s RMS, 8명 평균):\n", at.round(4).to_string(index=False))
pp.to_csv(os.path.join(OUT, "angle_per_participant.csv"), index=False)

fig, axes = plt.subplots(4, 5, figsize=(17, 10), sharex=True)
for r, g in enumerate(C.GESTURE_ORDER):
    for ch in range(5):
        a = axes[r, ch]
        for ang, ls, col in ((30, "-", "#1f77b4"), (60, "-", "#d62728")):
            m = ((meta.gesture == g) & (meta.angle == ang)).values
            a.plot(tt, env[m][:, :, ch].mean(0), color=col, lw=1.3, label=f"{ang}°")
        a.axvspan(seg["start_s"], seg["end_s"], color="green", alpha=0.06)
        row = at[(at.gesture == g) & (at.channel == C.CHANNELS[ch])].iloc[0]
        a.set_title(f"{g} · ch{ch + 1} {C.CHANNELS[ch]}   60/30 {row.diff_pct:+.0f} %  p={row.p:.3f}", fontsize=8)
        if ch == 0: a.set_ylabel("RMS (mV)")
        if r == 3: a.set_xlabel("s from mark")
axes[0, 0].legend(fontsize=8)
fig.suptitle("30° vs 60° — mean RMS envelope, all 8 participants (green: 1-3 s window used for RMS/t-test)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "angle_compare.png"), dpi=110); plt.close(fig)

# 피험자별로 각도 차이가 일관적인지 (BB, TB)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for k, g in enumerate(C.GESTURE_ORDER):
    a = axes[k]
    for ch, col in (("BB", "#d62728"), ("TB", "#9467bd")):
        for i, p in enumerate(C.PARTICIPANTS):
            v30 = pp[(pp.gesture == g) & (pp.angle == 30) & (pp.participant == p)][ch].values[0]
            v60 = pp[(pp.gesture == g) & (pp.angle == 60) & (pp.participant == p)][ch].values[0]
            a.plot([0, 1], [v30, v60], "o-", color=col, alpha=0.6, ms=3, lw=1, label=ch if i == 0 else None)
    a.set_xticks([0, 1], ["30°", "60°"]); a.set_title(f"{g}: BB / TB RMS per participant", fontsize=9)
    if k == 0: a.set_ylabel("RMS 1-3 s (mV)"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "angle_per_participant.png"), dpi=120); plt.close(fig)

json.dump(dict(segment=seg, motion_window=mw.to_dict(), onset_q=q.to_dict(), bin_ms=BIN), open(os.path.join(OUT, "plot_facts.json"), "w"), ensure_ascii=False, indent=1)
print("저장:", OUT)
