"""1주차 단계 2·3 — 데이터 확보 확인과 구조 파악 (제출물 ②)

 출력 (results/week1/):
   file_summary.csv     파일별 샘플 수·길이·mark 수·mark 간격·마지막 mark 뒤 여유·SHA-256
   class_summary.csv    8클래스 x 시행 수 (피험자별 포함)
   trial_manifest.csv   시행 1,920개 목록 (파일, mark 위치, 라벨) — 2주차 분할의 기준
   channel_stats.csv    클래스·채널별 신호 표준편차 (시행 전체 5 s 기준)
   data_facts.json      보고서용 숫자
"""
import json, os
import numpy as np, pandas as pd
import common as C

OUT = os.path.join(C.ROOT, "results", "week1"); os.makedirs(OUT, exist_ok=True)
files = C.list_files()
print(f"[단계 2] 파일 수: {len(files)}  (기대 64)   예시: {os.path.basename(files[0])}")

rows = []
for f in files:
    x, t, marks, onset = C.load_file(f)
    info = C.parse_name(f)
    gaps = np.diff(t[marks])
    rows.append(dict(file=os.path.basename(f), **info, n_samples=len(x), duration_s=round(float(t[-1] - t[0]), 3),
                     n_marks=len(marks), n_onset=len(onset), mark_gap_min_s=round(float(gaps.min()), 3), mark_gap_max_s=round(float(gaps.max()), 3),
                     first_mark_s=round(float(t[marks[0]] - t[0]), 3), tail_after_last_mark_s=round(float(t[-1] - t[marks[-1]]), 3),
                     dt_ms=round(float(np.median(np.diff(t)) * 1000), 3), sha256=C.sha256(f)))
fs = pd.DataFrame(rows)
fs.to_csv(os.path.join(OUT, "file_summary.csv"), index=False)
print(f"[단계 3] 샘플 수 {fs.n_samples.min()}~{fs.n_samples.max()} ({fs.duration_s.min()}~{fs.duration_s.max()} s), "
      f"mark 수 {fs.n_marks.unique().tolist()}, mark 간격 {fs.mark_gap_min_s.min()}~{fs.mark_gap_max_s.max()} s, "
      f"마지막 mark 뒤 여유 {fs.tail_after_last_mark_s.min()} s 이상, 샘플 간격 {fs.dt_ms.unique().tolist()} ms")

X, meta = C.load_all(use_cache=False)
print(f"         시행 배열: {X.shape}  -> (시행, 샘플, 채널)   NaN: {int(np.isnan(X).sum())}")
meta.to_csv(os.path.join(OUT, "trial_manifest.csv"), index=False)

cs = meta.pivot_table(index="label", columns="participant", values="trial", aggfunc="count").reindex(C.CLASSES)
cs["합계"] = cs.sum(1)
cs.to_csv(os.path.join(OUT, "class_summary.csv"))
print("         클래스별 시행 수:\n", cs.to_string())

std = np.array([[X[(meta.label == c).values][:, :, ch].std() for ch in range(5)] for c in C.CLASSES])
ch = pd.DataFrame(std, index=C.CLASSES, columns=C.CHANNELS).round(5)
ch.to_csv(os.path.join(OUT, "channel_stats.csv"))
print("         클래스·채널별 표준편차 (mV):\n", ch.to_string())

facts = dict(n_files=len(files), n_trials=int(len(meta)), trials_per_class=int(cs["합계"].iloc[0]), n_participants=meta.participant.nunique(),
             n_channels=5, fs=C.FS, trial_len=C.TRIAL_LEN, samples_min=int(fs.n_samples.min()), samples_max=int(fs.n_samples.max()),
             duration_min=float(fs.duration_s.min()), duration_max=float(fs.duration_s.max()), marks_per_file=int(fs.n_marks.iloc[0]),
             mark_gap_s=[float(fs.mark_gap_min_s.min()), float(fs.mark_gap_max_s.max())], tail_min_s=float(fs.tail_after_last_mark_s.min()),
             value_abs_max=float(np.abs(X).max()), value_std=float(X.std()), nan=int(np.isnan(X).sum()))
json.dump(facts, open(os.path.join(OUT, "data_facts.json"), "w"), indent=1, ensure_ascii=False)
print("저장:", OUT)
