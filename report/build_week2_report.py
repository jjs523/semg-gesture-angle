"""results/week2/ 산출물로 Step 2 2주차 보고서 HTML/PDF 생성.  실행: python report/build_week2_report.py --pdf"""
import base64, html as H, inspect, json, os, shutil, subprocess, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src"))
import features as F
R = os.path.join(ROOT, "results", "week2"); OUT_DIR = os.path.join(ROOT, "report")
WIN_TMP = "/mnt/c/Users/user/semg_report_tmp"; CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
SUBMIT_DATE = "2026년 10월 15일"
def img(n): return "data:image/png;base64," + base64.b64encode(open(os.path.join(R, n), "rb").read()).decode()
def table(df, caption, fmt=None, index=False, cls="tbl"):
    h = f"<div class='{cls}'><table><thead><tr>" + ("<th></th>" if index else "") + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        h += "<tr>" + (f"<th>{i}</th>" if index else "") + "".join(f"<td>{fmt[c].format(row[c]) if fmt and c in fmt else row[c]}</td>" for c in df.columns) + "</tr>"
    return h + f"</tbody></table><p class='cap'>{caption}</p></div>"
fx = json.load(open(os.path.join(R, "week2_facts.json"), encoding="utf-8"))
ds = pd.read_csv(os.path.join(R, "dataset_summary.csv"), index_col=0); dp = pd.read_csv(os.path.join(R, "dataset_by_participant.csv"), index_col=0)
seg = pd.read_csv(os.path.join(R, "segment_start.csv")); wl = pd.read_csv(os.path.join(R, "window_length.csv")); lk = pd.read_csv(os.path.join(R, "leakage.csv"))
su = pd.read_csv(os.path.join(R, "sample_unit.csv")); rc = pd.read_csv(os.path.join(R, "rest_confound.csv")); bv = pd.read_csv(os.path.join(R, "baseline_var.csv"))
pct = lambda v: f"{v*100:.2f}"
filt = pd.DataFrame({"채널": [f"ch{i+1} {c}" for i, c in enumerate(F.CHANNELS)], "60 Hz 전력 (후/전)": fx["p60_ratio"], "15 Hz 미만 (후/전)": fx["below15_ratio"],
                     "460 Hz 초과 (후/전)": fx["above460_ratio"], "전체 전력 보존": fx["power_kept"], "표준편차 전 (mV)": fx["std_before"], "표준편차 후 (mV)": fx["std_after"]})
ffmt = {c: "{:.3f}" for c in filt.columns[1:5]} | {c: "{:.4f}" for c in filt.columns[5:]}
zc = pd.DataFrame({"채널": F.CHANNELS, "ZC 평균 (th 0.01)": fx["zc_mean_per_ch"], "ZC 평균 (th 0)": fx["zc_th0_mean_per_ch"], "SSC 평균": fx["ssc_mean_per_ch"]})
seg2 = seg.assign(구간=lambda d: d.start_s.map(lambda v: f"{v:.1f}~{v+2:.1f} s"), 정확도=lambda d: [f"{a*100:.2f} ± {s*100:.2f}" for a, s in zip(d.acc, d["std"])])[["구간", "정확도"]]
wl2 = wl.assign(설정=lambda d: [f"{w} ms / {h} ms" for w, h in zip(d.win_ms, d.hop_ms)], 정확도=lambda d: [f"{a*100:.2f} ± {s*100:.2f}" for a, s in zip(d.acc, d["std"])]).rename(columns={"overlap": "오버랩", "n_windows": "세그먼트당 윈도우"})[["설정", "오버랩", "세그먼트당 윈도우", "정확도"]]
lk2 = lk.assign(정확도=lambda d: [f"{a*100:.2f} ± {s*100:.2f}" for a, s in zip(d.acc, d["std"])]).rename(columns={"split": "분할 방식"})[["분할 방식", "정확도"]]
su2 = su.assign(정확도=lambda d: [f"{a*100:.2f} ± {s*100:.2f}" for a, s in zip(d.acc, d["std"])]).rename(columns={"unit": "표본 단위"})[["표본 단위", "정확도"]]
rc2 = rc.pivot(index="target", columns="segment", values="acc").loc[["8클래스", "손동작 4종", "각도 2종"]].map(lambda v: f"{v*100:.2f}")
rc2.insert(0, "우연 수준", ["12.5", "25.0", "50.0"])
bv2 = bv.rename(columns={"channel": "채널", "between_files": "파일 간 SD (mV)", "within_file": "파일 내 SD (mV)", "ratio": "비율"})
lk_gap = (lk.acc[1] - lk.acc[0]) * 100
rest_g = rc[(rc.segment.str.startswith("휴식")) & (rc.target == "손동작 4종")].acc.iloc[0] * 100
rest_a = rc[(rc.segment.str.startswith("휴식")) & (rc.target == "각도 2종")].acc.iloc[0] * 100
rest_8 = rc[(rc.segment.str.startswith("휴식")) & (rc.target == "8클래스")].acc.iloc[0] * 100
w_acc, v_acc, c_acc, m_acc = (su.acc * 100).tolist()
td_src = H.escape(inspect.getsource(F.td_features))
split_code = H.escape('''strat = (meta.label + "_" + meta.participant).values        # 클래스 x 피험자 층화
# ① 시행(=mark 1개) 단위로 먼저 8:2 분할  ← 순서가 핵심
tr_idx, te_idx = train_test_split(np.arange(len(meta)), test_size=0.2,
                                  stratify=strat, random_state=42)
assert not set(tr_idx) & set(te_idx)                       # 같은 시행이 양쪽에 없음
# ② 각 집합의 시행 안에서만 세그먼트 -> 윈도우 -> 특징 생성
def build(idx):
    Xw = F.trial_features(Xf[idx])                         # (n시행, 19, 25)
    y  = np.repeat(meta.label.values[idx], Xw.shape[1])
    tid = np.repeat(idx, Xw.shape[1])                      # 윈도우가 속한 시행 번호
    return Xw.reshape(-1, 25), y, tid
Xtr, ytr, tid_tr = build(tr_idx)
Xte, yte, tid_te = build(te_idx)
assert not set(tid_tr) & set(tid_te)
# ③ 표준화 통계는 학습셋에서만
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)''')
split_code = split_code.replace("# ① 시행(=mark 1개) 단위로 먼저 8:2 분할  ← 순서가 핵심", "<mark># ① 시행(=mark 1개) 단위로 먼저 8:2 분할  ← 순서가 핵심</mark>")
for ln in ("tr_idx, te_idx = train_test_split(np.arange(len(meta)), test_size=0.2,", "                                  stratify=strat, random_state=42)"):
    split_code = split_code.replace(ln, f"<mark>{ln}</mark>")

html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>Step 2 2주차 보고서</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 20mm 20mm; }}
body {{ font-family: "Noto Serif KR", "Batang", "바탕", serif; font-size: 10.5pt; line-height: 1.65; color: #000; margin: 0; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 4px; }} .sub {{ text-align: center; font-size: 11pt; margin: 0 0 14px; }}
h2 {{ font-size: 12.5pt; margin: 20px 0 6px; break-after: avoid; }} h3 {{ font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }}
p {{ margin: 4px 0 6px; text-align: justify; }}
table {{ border-collapse: collapse; margin: 8px auto 2px; font-size: 9.3pt; }} .tbl {{ break-inside: avoid; }}
th, td {{ border: 1px solid #333; padding: 3px 7px; text-align: center; }} th {{ background: #f0f0f0; font-weight: 700; }}
.cap {{ text-align: center; font-size: 9.5pt; margin: 2px 0 10px; }}
figure {{ margin: 8px 0 10px; text-align: center; break-inside: avoid; }} figcaption {{ font-size: 9.5pt; margin-top: 2px; }}
code {{ font-family: "D2Coding", "Consolas", monospace; font-size: 9.3pt; }}
pre {{ font-family: "D2Coding", "Consolas", monospace; font-size: 8.8pt; background: #f6f6f6; border: 1px solid #ccc; padding: 8px 10px; line-height: 1.4; break-inside: avoid; white-space: pre-wrap; }}
pre mark {{ background: #ffe08a; }}
.hdr td {{ text-align: left; }} .hdr th {{ width: 60px; }} .ref p {{ text-align: left; margin: 2px 0; }}
.box {{ border: 1px solid #888; background: #fafafa; padding: 6px 10px; margin: 8px 0; break-inside: avoid; }}
ol li {{ margin: 3px 0; }}
</style></head><body>
<h1>2주차 과제 보고서 — 전처리와 특징 추출</h1>
<p class="sub">Step 2 · sEMG 팔동작 예측 프로젝트 (손동작 × 팔꿈치 각도 동시 분류)</p>
<table class="hdr" style="width:100%"><tr><th>학 과</th><td>컴퓨터소프트웨어학부</td><th>학 번</th><td>22114553</td></tr>
<tr><th>이 름</th><td>전진성</td><th>제출일</th><td>{SUBMIT_DATE}</td></tr>
<tr><th>저장소</th><td colspan="3">https://github.com/jjs523/semg-gesture-angle &nbsp;(실행 <code>python main.py week2</code>, 노트북 <code>notebooks/week2.ipynb</code>, 데이터셋 <code>data/dataset.npz</code>)</td></tr></table>

<div class="box"><b>요약.</b> 1,920시행을 60 Hz 노치 + 20~450 Hz 필터 → mark + 1.0~3.0 s 세그먼트 → 200 ms·100 ms 윈도우 19개 → 시간영역 특징 25차원으로 바꾸고, <b>시행 단위로 먼저 8:2 분할</b>한 뒤 학습셋 통계로 표준화하여 학습 29,184 / 테스트 7,296 윈도우의 <code>dataset.npz</code>를 만들었다. 학습셋 안의 5-fold 검증에서 세 가지를 확인하였다. ① 윈도우 하나의 정확도는 {w_acc:.1f} %이고 논문 수준(96 %)은 세그먼트(19윈도우) 단위에서 나온다({v_acc:.1f} %). ② 윈도우를 먼저 만들고 나누면 정확도가 {lk_gap:.1f} %p 부풀려진다. ③ <b>손을 쉬고 있는 4~5 s 구간만으로도 손동작을 {rest_g:.1f} % 맞힌다</b>(우연 25 %). 클래스마다 별도 파일로 녹음된 데이터라 trial-wise 평가에 녹음 세션 교란이 섞여 있다는 뜻이며, 3주차 해석과 LOSO 검증의 근거가 된다.</div>

<h2>1. 단계 1 — 필터 적용과 검증 (제출물 ①)</h2>
<p>배포본은 이미 같은 필터를 거친 신호이지만 논문 절차를 재현하기 위해 다시 적용하였다. <code>iirnotch(60, 30)</code> 뒤에 4차 Butterworth 20~450 Hz 대역통과를 이어 걸고, 둘 다 <code>filtfilt</code>로 시간축(axis=0)에 영위상으로 적용해 신호가 밀리지 않게 하였다. 시행 배열 (1920, 5000, 5) 전체를 한 번에 처리한다(<code>src/features.py</code>의 <code>preprocess</code>).</p>
<figure><img src="{img('filter_check.png')}" style="width:100%"><figcaption>그림 1. 1,920시행 평균 Welch 스펙트럼 — 필터 전(왼쪽), 후(가운데), 후/전 비(오른쪽). 빨간 점선 60 Hz, 회색 점선 20·450 Hz</figcaption></figure>
{table(filt, "표 1. 필터 효과 (1,920시행 평균 스펙트럼 기준, 표준편차는 피험자 a·30_fist 첫 시행)", fmt=ffmt)}
<p>60 Hz 성분은 20~24 %로 줄고, 15 Hz 미만과 460 Hz 초과 성분은 1 % 남짓만 남는다. 전체 전력은 90~95 %가 보존되어 근활성 대역은 손대지 않았다. 다만 그림 1 왼쪽에서 보듯 배포본에는 이미 60 Hz와 200·400 Hz 골이 있고, <b>120·180·300 Hz 같은 전원 고조파 봉우리는 노치 하나로 지워지지 않고 남는다</b>. 논문 조건(60 Hz 노치 하나)을 그대로 따랐으므로 이 고조파는 특징에 그대로 들어가며, 고조파 노치 추가는 3주차 이후 어블레이션 후보로 남긴다.</p>

<h2>2. 단계 2·3 — 분석 구간과 슬라이딩 윈도우 (제출물 ②)</h2>
<p>논문은 2 s 세그먼트의 시작점을 명시하지 않고 실습 자료는 1.5 s를 가정하지만, 1주차에 1,920시행의 RMS 포락선으로 측정한 활성 구간은 mark 뒤 0.72 s 시작 · 1.68 s 최대 · 3.22 s 종료(중앙값)였다. 1.5~3.5 s로 자르면 상승부 절반을 잃으므로 <b>mark + 1.0~3.0 s</b>를 세그먼트로 잡고 재현 가정으로 기록한다. 이 2,000샘플을 200샘플 윈도우, 100샘플 증분으로 자르면 (2000 − 200) / 100 + 1 = 19개가 나온다.</p>
<pre>[단계 2] 세그먼트 모양 {tuple(fx['segment_shape'])}  (mark + 1.0~3.0 s)
[단계 3] 윈도우 배열 {tuple(fx['window_shape'])}  ((2000-200)/100+1 = 19)</pre>
<figure><img src="{img('segment_windows.png')}" style="width:92%"><figcaption>그림 2. 필터된 시행 하나(5채널)와 2 s 세그먼트(녹색), 그 안의 200 ms 윈도우 19개</figcaption></figure>

<h2>3. 단계 4 — 시간영역 특징 구현 (제출물 ③)</h2>
<p>윈도우 하나에서 채널마다 RMS·MAV·WL·ZC·SSC를 numpy만으로 계산해 25차원 벡터를 만든다. 논문 식 (4)에 따라 <b>ZC에만 임계값 0.01</b>을 두고, 식 (5)의 SSC는 (x<sub>i</sub> − x<sub>i−1</sub>)(x<sub>i</sub> − x<sub>i+1</sub>) &gt; 0인 점의 개수로 임계값 없이 센다. 순서는 [RMS×5채널, MAV×5, WL×5, ZC×5, SSC×5]로 고정하였다. 함수는 (…, 200, 5) 모양을 받아 시행 배열 전체에 한 번에 적용된다.</p>
<pre>{td_src}</pre>
<pre>[단계 4] 특징 벡터 {tuple(fx['feature_shape'])}</pre>
{table(zc, "표 2. 채널별 ZC·SSC 평균 (윈도우 200샘플당 횟수, 전체 36,480윈도우)", fmt={c: "{:.2f}" for c in zc.columns[1:]})}
<p>임계값 0.01은 이 데이터의 단위(mV)에서 상당히 크다. 휴식 수준 진폭이 0.005 mV 안팎이라 임계값 0에서는 윈도우당 21~31회인 영교차가 0.01에서는 3~13회로 줄고, 전체 윈도우·채널의 {fx['zc_zero_frac']*100:.1f} %에서 ZC가 0이다. 즉 이 설정의 ZC는 "주파수"보다 "충분히 큰 진동이 있었는가"를 세는 특징에 가깝다. 논문 원자료의 단위가 확인되지 않아 식대로 0.01을 유지하였다.</p>
<figure><img src="{img('feature_class_heatmap.png')}" style="width:100%"><figcaption>그림 3. 25개 특징의 클래스별 평균 (클래스 간 z-score)</figcaption></figure>
<p>그림 3에서 동작 축은 전완 채널 블록이 가른다(주먹은 ECR, 펴기는 ED·FDS가 높음). 각도 축은 BB 열에서 갈려 60° 회외가 RMS·MAV·WL·ZC·SSC 모두 극단값이다. 1주차의 진폭 비교(회외 BB +55 %)가 특징 공간에서도 그대로 보인다.</p>

<h2>4. 단계 5 — 데이터셋 만들기, 누수 막기 (제출물 ④·⑤)</h2>
<p>한 시행의 19개 윈도우는 절반씩 겹치므로, 윈도우를 먼저 만들고 나누면 이웃 윈도우가 학습·테스트 양쪽에 들어간다. 그래서 <b>시행(mark 1개) 단위로 먼저 8:2로 나누고, 각 집합의 시행 안에서만 윈도우와 특징을 만들었다</b>(노란 줄). 층화는 클래스 × 피험자(64칸, 칸마다 30시행 → 24 / 6)로 걸어 8클래스와 8명이 양쪽에 똑같이 들어가게 했고, 시드 42로 고정하였다. 분할된 시행 목록은 <code>results/week2/split.json</code>에 있다.</p>
<pre>{split_code}</pre>
{table(ds, "표 3. 클래스별 학습·테스트 시행 수와 윈도우 수 (제출물 ⑤)", index=True)}
{table(dp.T, "표 4. 피험자별 시행 수", index=True)}
<p>학습 1,536시행 × 19 = 29,184윈도우, 테스트 384시행 × 19 = 7,296윈도우로 정확히 4 : 1이고 8클래스·8명이 모두 균등하다. 논문의 1,536 / 384 세그먼트 분할과 같은 규모다.</p>

<h2>5. 단계 6 — 표준화와 저장</h2>
<p><code>StandardScaler</code>를 학습셋에만 <code>fit</code>하고 테스트셋에는 <code>transform</code>만 적용했다. 학습셋은 25개 특징 모두 평균 |최대| {fx['train_mean_absmax']:.1e}, 표준편차 {fx['train_std_min']:.4f}~{fx['train_std_max']:.4f}로 0 · 1이 되었고, 테스트셋은 평균 |최대| {fx['test_mean_absmax']:.3f}, 표준편차 {fx['test_std_range'][0]:.3f}~{fx['test_std_range'][1]:.3f}로 0 · 1에 가깝지만 정확히 같지 않다(정상). <code>data/dataset.npz</code>(2.5 MB, 저장소 포함)에는 <code>Xtr, ytr, Xte, yte</code> 외에 윈도우가 속한 시행 번호·윈도우 위치·피험자, 특징 이름, 스케일러 통계를 함께 저장해 3주차에 세그먼트 단위 집계와 피험자별 분석을 할 수 있게 했다. 5-fold용으로는 표준화 전 특징 (1920, 19, 25)을 <code>data/features_all.npz</code>에 따로 저장했다(폴드마다 학습 폴드에 다시 fit).</p>

<h2>6. 설정 근거 — 윈도우 200 ms · 오버랩 50 % (제출물 ⑥)</h2>
<p>근거를 말로만 쓰지 않으려고, <b>학습 1,536시행 안에서만</b> 시행 단위 5-fold로 RF(트리 100개)를 돌려 설정을 비교했다. 테스트 384시행은 3주차 최종 평가까지 보지 않는다.</p>
<figure><img src="{img('settings_check.png')}" style="width:100%"><figcaption>그림 4. 세그먼트 시작점(왼쪽), 윈도우 길이·증분(가운데), 분할 방식(오른쪽)에 따른 윈도우 정확도. 학습셋 안 5-fold</figcaption></figure>
{table(wl2, "표 5. 윈도우 길이·증분 비교 (윈도우 단위 정확도 %, 세그먼트 1.0~3.0 s)")}
<p><b>윈도우 200 ms.</b> 근전도는 수백 ms 안에서 통계적 성질이 거의 변하지 않는다고 볼 수 있어 RMS·ZC 같은 통계량이 안정적으로 잡히고, 의수·재활 기기 제어에서 허용되는 판정 지연(약 300 ms) 안에 들어오는 길이다. 실제로 100 ms로 줄이면 정확도가 {wl.acc[0]*100:.1f} %로 크게 떨어지고 300·400 ms는 {wl.acc[4]*100:.1f}·{wl.acc[5]*100:.1f} %로 오르지만, 지연이 커지고 세그먼트당 표본이 12·9개로 줄어드므로 논문 설정 200 ms를 유지했다.</p>
<p><b>오버랩 50 %(증분 100 ms).</b> 증분 100 ms는 판정을 0.1 s마다 갱신하게 하고, 같은 2 s에서 표본을 10개(오버랩 0 %)에서 19개로 두 배 가까이 늘린다. 오버랩 0 %는 {wl.acc[3]*100:.1f} %, 75 %(증분 50 ms)는 {wl.acc[1]*100:.1f} %로 오버랩이 늘수록 조금씩 오르지만, 75 %는 이웃 윈도우가 거의 같은 정보라 이득 {(wl.acc[1]-wl.acc[2])*100:.1f} %p에 비해 계산량과 표본 간 상관이 두 배가 되므로 50 %로 두었다.</p>

<h2>7. 추가 검증 — 숫자를 믿기 전에 확인한 것</h2>
<h3>7.1 표본 단위: 논문 96 %는 세그먼트 단위다</h3>
{table(su2, "표 6. 같은 특징, 다른 표본 단위의 정확도 (RF, 학습셋 안 5-fold, %)")}
<p>윈도우 하나(200 ms)로 8클래스를 맞히면 {w_acc:.1f} %에 그치지만, 세그먼트의 19윈도우 확률을 평균하면 {v_acc:.1f} %, 19 × 25 = 475차원으로 이어 붙이면 {c_acc:.1f} %가 되어 논문의 RF 96.72 %와 같은 수준이다. 논문의 테스트 384개도 세그먼트 수와 정확히 같으므로, <b>논문 수치는 세그먼트 단위 평가로 보는 것이 맞다</b>. 3주차에는 윈도우 단위와 세그먼트 단위를 둘 다 보고하고, 논문과는 세그먼트 단위로 비교한다.</p>
<h3>7.2 누수의 크기: 윈도우부터 나누면 {lk_gap:.1f} %p 부풀려진다</h3>
{table(lk2, "표 7. 같은 학습 1,536시행을 두 방식으로 5-fold (윈도우 정확도 %)")}
<p>윈도우를 먼저 만들고 섞어서 나누면, 50 % 겹치는 이웃 윈도우가 학습·평가 양쪽에 들어가 정확도가 {lk.acc[0]*100:.2f} → {lk.acc[1]*100:.2f} %로 오른다. 4절의 순서(시행 먼저)가 이 차이를 막는다.</p>
<h3>7.3 녹음 세션 교란: 쉬고 있는 손으로도 동작을 맞힌다</h3>
<p>그림 4 왼쪽에서 이상한 점이 보였다. 세그먼트를 동작 구간(1.0~3.0 s)에서 휴식 구간(2.5~4.5 s) 쪽으로 옮길수록 정확도가 <b>올라간다</b>(59.5 → 71.8 %). 동작 정보가 줄어드는데 정확도가 오른다면 동작이 아닌 다른 단서가 있다는 뜻이라 직접 확인했다.</p>
<figure><img src="{img('confound.png')}" style="width:100%"><figcaption>그림 5. 왼쪽: 동작 구간과 휴식 구간(4~5 s) 특징으로 각 목표를 맞힌 세그먼트 정확도(검은 선 = 우연 수준). 오른쪽: 64개 녹음 파일의 휴식 구간 FDS RMS(회색 = 시행, 빨강 = 파일 평균)</figcaption></figure>
{table(rc2, "표 8. 동작 구간 vs 휴식 구간으로 맞힌 정확도 (세그먼트 단위, 학습셋 안 5-fold, %)", index=True)}
{table(bv2, "표 9. 휴식 구간(4~5 s) RMS의 파일 간 / 파일 내 표준편차", fmt={"파일 간 SD (mV)": "{:.5f}", "파일 내 SD (mV)": "{:.5f}", "비율": "{:.2f}"})}
<p><b>손을 쉬고 있는 4~5 s 구간만으로 손동작 4종을 {rest_g:.1f} % 맞힌다</b>(우연 25 %). 각도({rest_a:.1f} %)는 휴식 중에도 팔꿈치 자세를 유지하므로 정당한 단서가 일부 있지만, 손동작은 휴식 구간에 정보가 없어야 한다. 원인은 데이터 구조에 있다. 이 데이터는 <b>클래스 × 피험자마다 별도 파일로 30시행을 연속 녹음</b>했고, 표 9처럼 휴식 기저선 크기가 파일마다 2.0~2.6배 더 크게 달라진다(전극 접촉·피부 상태가 녹음마다 다름). 시행 단위로 무작위 분할하면 같은 파일의 시행이 학습과 평가에 함께 들어가므로, 모델은 동작 패턴과 함께 "이 녹음 세션의 기저선"을 단서로 쓸 수 있다.</p>
<p>이 결과가 3주차에 주는 의미는 세 가지다. ① 논문과 같은 trial-wise 정확도(96 % 수준)에는 녹음 세션 단서가 섞여 있으므로 "동작을 그만큼 잘 읽는다"로 해석하면 안 된다. ② 피험자를 통째로 빼는 LOSO 검증은 평가 피험자의 녹음 파일을 전혀 보지 않으므로 이 교란을 제거한 성능을 준다. ③ 세그먼트 시작점을 정확도로 고르면 교란을 최대화하는 쪽(휴식 구간)을 고르게 되므로, 1.0~3.0 s는 정확도가 아니라 1주차의 생리적 활성 구간으로 정한 것이 맞다.</p>

<h2>8. 체크포인트·제출물 대조</h2>
{table(pd.DataFrame([
 ["① 필터 검증", "그림 1, 표 1 — results/week2/filter_check.png"],
 ["② 윈도우 확인", "2절 출력 (19, 200, 5) — 노트북 단계 3"],
 ["③ 특징 구현 코드", "3절 td_features 전체 — src/features.py"],
 ["④ 분할 코드 (시행 먼저)", "4절 노란 표시 — src/week2_features.py 단계 5, results/week2/split.json"],
 ["⑤ 데이터셋 요약", "표 3·4 — results/week2/dataset_summary.csv"],
 ["⑥ 설정 근거 (200 ms · 50 %)", "6절 두 문단, 표 5 — results/week2/window_length.csv"],
 ["제출 형식", "이 PDF + data/dataset.npz + notebooks/week2.ipynb + 저장소 링크"],
], columns=["슬라이드 17쪽 제출물", "산출물"]), "표 10. 제출물 대응표")}
<p>강의 자료 15쪽 체크포인트 7개 — 필터 후 5채널 표준편차 출력, 세그먼트 (2000, 5), 윈도우 (19, 200, 5), 특징 (25,), 시행 단위 분할 후 특징 생성, 학습셋 평균 0 · 분산 1, <code>dataset.npz</code> 저장 — 는 모두 노트북 실행본(<code>notebooks/week2_executed.ipynb</code>)에서 확인된다.</p>

</body></html>"""
os.makedirs(OUT_DIR, exist_ok=True)
hp = os.path.join(OUT_DIR, "week2_report.html"); open(hp, "w", encoding="utf-8").write(html); print("HTML", len(html) // 1024, "KB")
if "--pdf" in sys.argv:
    os.makedirs(WIN_TMP, exist_ok=True); shutil.copy(hp, os.path.join(WIN_TMP, "week2_report.html"))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-margins", r"--print-to-pdf=C:\Users\user\semg_report_tmp\week2_report.pdf",
                    "--virtual-time-budget=10000", "file:///C:/Users/user/semg_report_tmp/week2_report.html"], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = os.path.join(OUT_DIR, "Step2_2주차과제_보고서_22114553_전진성.pdf"); shutil.copy(os.path.join(WIN_TMP, "week2_report.pdf"), out)
    shutil.rmtree(WIN_TMP, ignore_errors=True); print("PDF", out, os.path.getsize(out) // 1024, "KB")
