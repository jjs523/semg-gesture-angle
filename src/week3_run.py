"""Step 2 · 3주차 — 학습·평가 실행기 (강의 3주차 단계 1·2·4·6 + 심화 LOSO)

 holdout  2주차 dataset.npz (시행 단위 8:2) 로 세 모델 학습 -> 테스트 1회 평가, 계산 비용 측정
 cv       1,920시행을 시행 단위 StratifiedKFold(5, shuffle, seed 42) -> 폴드마다 학습 폴드에만 scaler fit
 loso     피험자 1명을 통째로 빼고 7명으로 학습 (8회) -> 녹음 세션 교란이 없는 새 사람 성능

 두 단위로 평가한다.
   윈도우 단위   200 ms 윈도우 하나 = 표본 하나
   세그먼트 단위 한 시행의 19개 윈도우 점수를 평균해 판정 (논문 테스트 384개 = 세그먼트 수, 2주차 7.1절)
 결과: results/week3/<mode>_<model>[_fold k].json (지표·혼동행렬·예측)
 논문 재현 설정(concat): 세그먼트 하나 = 19개 윈도우 특징을 이어 붙인 475차원 — 5-fold RF 96.72 % 로 논문과 일치 (sample_unit_cv.csv)
 실행: python src/week3_run.py [holdout|cv|loso|concat|all]
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import common as C
import features as F
from models import MODEL_NAMES, make_model, scores

OUT = os.path.join(C.ROOT, "results", "week3"); os.makedirs(OUT, exist_ok=True)
DS = os.path.join(C.ROOT, "data")
le = LabelEncoder().fit(C.CLASSES)          # 0..7, C.CLASSES 순서(30_fist ... 60_supination)와 같음
assert list(le.classes_) == sorted(C.CLASSES)
ORDER = [list(le.classes_).index(c) for c in C.CLASSES]   # 혼동행렬을 논문 순서로 다시 배열할 인덱스


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(8), zero_division=0)
    cm = confusion_matrix(y, p, labels=range(8))[np.ix_(ORDER, ORDER)]
    return dict(accuracy=float(accuracy_score(y, p)), precision=float(pr.mean()), recall=float(rc.mean()), f1=float(f1.mean()),
                per_class_recall=rc[ORDER].tolist(), per_class_precision=pr[ORDER].tolist(), per_class_f1=f1[ORDER].tolist(), confusion=cm.tolist())


def evaluate_concat(model, Xte, yte):
    return dict(segment=metrics(yte, scores(model, Xte).argmax(1)), n_segment=int(len(yte)), seg_pred=scores(model, Xte).argmax(1).tolist(), seg_true=yte.tolist())


def evaluate(model, Xte, yte, tid):
    s = scores(model, Xte)
    pw = s.argmax(1)
    trials = np.unique(tid)
    seg_s = np.stack([s[tid == t].mean(0) for t in trials]); seg_y = np.array([yte[tid == t][0] for t in trials])
    ps = seg_s.argmax(1)
    return dict(window=metrics(yte, pw), segment=metrics(seg_y, ps), n_window=int(len(yte)), n_segment=int(len(trials)),
                seg_trials=trials.tolist(), seg_pred=ps.tolist(), seg_true=seg_y.tolist())


def fit_eval(name, Xa, ya, Xb, yb, tid_b, unit="window"):
    m = make_model(name); t0 = time.perf_counter(); m.fit(Xa, ya); tt = time.perf_counter() - t0
    r = evaluate(m, Xb, yb, tid_b) if unit == "window" else evaluate_concat(m, Xb, yb); r["train_time_s"] = tt
    return m, r


def latency_ms(model, X, reps=200):
    """샘플(윈도우) 1개 추론 시간. 병렬 스레드 생성 오버헤드를 빼기 위해 1스레드로 잰다."""
    if hasattr(model, "n_jobs"): model.set_params(n_jobs=1)
    x1 = X[:1]
    for _ in range(20): model.predict(x1)                     # 예열
    t0 = time.perf_counter()
    for _ in range(reps): model.predict(x1)
    return (time.perf_counter() - t0) / reps * 1000


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, f"{name}.json"), "w"), ensure_ascii=False)


def run_holdout():
    d = np.load(os.path.join(DS, "dataset.npz"), allow_pickle=True)
    Xtr, Xte = d["Xtr"], d["Xte"]; ytr, yte = le.transform(d["ytr"]), le.transform(d["yte"])
    for name in MODEL_NAMES:
        m, r = fit_eval(name, Xtr, ytr, Xte, yte, d["trial_te"])
        r["latency_ms_per_window"] = latency_ms(m, Xte)
        t0 = time.perf_counter(); m.predict(Xte); r["batch_ms_per_window"] = (time.perf_counter() - t0) / len(Xte) * 1000
        r["model_size_mb"] = len(pickle.dumps(m)) / 1e6
        r["participant_te"] = d["participant_te"].tolist()
        pid = {t: p for t, p in zip(d["trial_te"], d["participant_te"])}
        r["seg_participant"] = [pid[t] for t in r["seg_trials"]]
        save(f"holdout_{name}", r)
        print(f"[holdout {name}] window acc {r['window']['accuracy']*100:.2f}  segment acc {r['segment']['accuracy']*100:.2f}  "
              f"train {r['train_time_s']:.2f}s  latency {r['latency_ms_per_window']:.3f} ms  size {r['model_size_mb']:.2f} MB", flush=True)


def run_holdout_concat():
    """논문 재현 설정: 세그먼트 하나 = 19 x 25 = 475차원. 2주차와 같은 시행 단위 분할(dataset.npz 의 시행 번호)."""
    d = np.load(os.path.join(DS, "dataset.npz"), allow_pickle=True)
    FT, lab, part = load_features(); y = le.transform(lab)
    a, b = np.unique(d["trial_tr"]), np.unique(d["trial_te"]); assert not set(a) & set(b)
    sc = StandardScaler().fit(FT[a].reshape(len(a), -1))
    Xa, Xb = sc.transform(FT[a].reshape(len(a), -1)), sc.transform(FT[b].reshape(len(b), -1))
    for name in MODEL_NAMES:
        m, r = fit_eval(name, Xa, y[a], Xb, y[b], None, unit="concat")
        r["latency_ms_per_segment"] = latency_ms(m, Xb)
        t0 = time.perf_counter(); m.predict(Xb); r["batch_ms_per_segment"] = (time.perf_counter() - t0) / len(Xb) * 1000
        r["model_size_mb"] = len(pickle.dumps(m)) / 1e6; r["seg_trials"] = b.tolist(); r["seg_participant"] = part[b].tolist()
        save(f"concat_holdout_{name}", r)
        print(f"[concat holdout {name}] segment acc {r['segment']['accuracy']*100:.2f}  train {r['train_time_s']:.2f}s  "
              f"latency {r['latency_ms_per_segment']:.3f} ms  size {r['model_size_mb']:.2f} MB", flush=True)


def load_features():
    p = os.path.join(DS, "features_all.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z["F"], z["label"], z["participant"]
    X, meta = C.load_all()                                    # features_all.npz 가 없으면 다시 계산 (약 1분)
    FT = F.trial_features(F.preprocess(X.astype(np.float64)).astype(np.float32))
    return FT, meta.label.values, meta.participant.values


def run_split(mode, unit="window"):
    FT, lab, part = load_features(); y = le.transform(lab); n = len(FT)
    if mode == "cv":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(n), lab))
    else:
        splits = [(np.flatnonzero(part != p), np.flatnonzero(part == p)) for p in C.PARTICIPANTS]
    for k, (a, b) in enumerate(splits):
        assert not set(a) & set(b)
        if unit == "window":
            Xa, Xb = FT[a].reshape(-1, 25), FT[b].reshape(-1, 25); ya, yb = np.repeat(y[a], 19), np.repeat(y[b], 19); tid = np.repeat(b, 19)
        else:
            Xa, Xb = FT[a].reshape(len(a), -1), FT[b].reshape(len(b), -1); ya, yb = y[a], y[b]; tid = None
        sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)    # 학습 폴드에만 fit
        for name in MODEL_NAMES:
            _, r = fit_eval(name, Xa, ya, Xb, yb, tid, unit=unit)
            r.update(fold=k, held_out=C.PARTICIPANTS[k] if mode == "loso" else None, n_train_trials=int(len(a)), n_eval_trials=int(len(b)))
            if unit == "concat": r["seg_participant"] = part[b].tolist()
            save(f"{'concat_' if unit == 'concat' else ''}{mode}_{name}_fold{k}", r)
            print(f"[{unit} {mode} fold {k}{' (' + C.PARTICIPANTS[k] + ')' if mode == 'loso' else ''} {name}] "
                  + (f"window {r['window']['accuracy']*100:.2f}  " if unit == "window" else "")
                  + f"segment {r['segment']['accuracy']*100:.2f}  ({r['train_time_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("holdout", "all"): run_holdout()
    if what in ("cv", "all"): run_split("cv")
    if what in ("loso", "all"): run_split("loso")
    if what in ("concat", "all"):            # 논문 재현 설정 (475차원 세그먼트)
        run_holdout_concat(); run_split("cv", "concat"); run_split("loso", "concat")
