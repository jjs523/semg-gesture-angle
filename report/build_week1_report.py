"""results/week1/ 산출물로 Step 2 1주차 보고서 HTML 을 만들고 Windows Chrome 으로 PDF 를 찍는다.  실행: python report/build_week1_report.py --pdf"""
import base64, json, os, shutil, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "week1"); OUT_DIR = os.path.join(ROOT, "report")
WIN_TMP = "/mnt/c/Users/user/semg_report_tmp"; CHROME = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
SUBMIT_DATE = "2026년 10월 8일"

def img(n): return "data:image/png;base64," + base64.b64encode(open(os.path.join(R, n), "rb").read()).decode()
def table(df, caption, fmt=None, index=False, allow_break=False):
    h = f"<div class='{'tbl-long' if allow_break else 'tbl'}'><table><thead><tr>" + ("<th></th>" if index else "") + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        h += "<tr>" + (f"<th>{i}</th>" if index else "") + "".join(f"<td>{fmt[c].format(row[c]) if fmt and c in fmt else row[c]}</td>" for c in df.columns) + "</tr>"
    return h + f"</tbody></table><p class='cap'>{caption}</p></div>"

facts = json.load(open(os.path.join(R, "data_facts.json"), encoding="utf-8"))
fs = pd.read_csv(os.path.join(R, "file_summary.csv")); cs = pd.read_csv(os.path.join(R, "class_summary.csv"), index_col=0)
ch = pd.read_csv(os.path.join(R, "channel_stats.csv"), index_col=0); mw = pd.read_csv(os.path.join(R, "motion_window.csv"), index_col=0)
at = pd.read_csv(os.path.join(R, "angle_rms_table.csv")); env = open(os.path.join(R, "env.txt"), encoding="utf-8").read().strip().splitlines()
pf = json.load(open(os.path.join(R, "plot_facts.json"), encoding="utf-8"))
design = open(os.path.join(ROOT, "docs", "class_design.md"), encoding="utf-8").read()

env_df = pd.DataFrame([l.split(None, 1) for l in env], columns=["패키지", "버전"])
ex = fs.iloc[0]
struct = pd.DataFrame([
    ["파일 수", f"{facts['n_files']}개 = 피험자 {facts['n_participants']}명 × 각도 2 × 동작 4", "`a_30 fist.csv` 형식, extor = 회외"],
    ["파일 길이", f"{facts['samples_min']:,}~{facts['samples_max']:,}샘플 ({facts['duration_min']}~{facts['duration_max']} s)", "30시행 연속 녹음"],
    ["열", "bandpass 1~5, time, onset, mark", "신호 5채널 + 시각 + 표식 2개"],
    ["샘플 간격", "1.0 ms (1 kHz)", "time 열 차분으로 확인"],
    ["시행 표식 mark", f"파일당 {facts['marks_per_file']}개, 간격 {facts['mark_gap_s'][0]}~{facts['mark_gap_s'][1]} s", "마지막 mark 뒤 여유 ≥ " + f"{facts['tail_min_s']} s"],
    ["시행 배열", f"({facts['n_trials']}, {facts['trial_len']}, {facts['n_channels']})", "mark부터 5,000샘플, NaN 0개"],
    ["클래스 분포", f"8클래스 × {facts['trials_per_class']}시행 (피험자당 30)", "완전 균등"],
    ["값 범위", f"표준편차 {facts['value_std']:.3f} mV, 절댓값 최대 {facts['value_abs_max']:.1f} mV", "이미 60 Hz 노치 + 20~450 Hz 필터 적용"],
], columns=["항목", "값", "비고"])
cs2 = cs.copy(); cs2.columns = [c if c == "합계" else f"피험자 {c}" for c in cs2.columns]
mw2 = mw.rename(columns={"onset_s": "활성 시작 (s)", "peak_s": "최대 (s)", "offset_s": "활성 끝 (s)"})
at2 = at.copy(); at2["60/30"] = at2.diff_pct.map(lambda v: f"{v:+.0f} %"); at2["p"] = at2.p.map(lambda v: f"{v:.3f}" + (" *" if v < 0.05 else ""))
at2 = at2.rename(columns={"gesture": "동작", "channel": "채널", "rms30": "30° RMS (mV)", "rms60": "60° RMS (mV)", "n_participants_60_higher": "60°가 큰 피험자 수 (8명 중)"})[["동작", "채널", "30° RMS (mV)", "60° RMS (mV)", "60/30", "60°가 큰 피험자 수 (8명 중)", "p"]]
sig = at[at.p < 0.05]

import re
def md_to_html(md):
    out, in_tbl = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "): continue
            tag = "th" if not in_tbl else "td"
            if not in_tbl: out.append("<div class='tbl'><table>"); in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl: out.append("</table></div>"); in_tbl = False
        if line.startswith("# "): continue
        elif line.startswith("## "): out.append(f"<h3>{line[3:]}</h3>")
        elif line.startswith("- "): out.append(f"<p class='li'>• {line[2:]}</p>")
        elif re.match(r"^\d+\. ", line): out.append(f"<p class='li'>{line}</p>")
        elif line.strip(): out.append(f"<p>{line}</p>")
    if in_tbl: out.append("</table></div>")
    h = "\n".join(out)
    h = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", h); h = re.sub(r"`(.+?)`", r"<code>\1</code>", h)
    return h

html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>Step 2 1주차 보고서</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 20mm 20mm; }}
body {{ font-family: "Noto Serif KR", "Batang", "바탕", serif; font-size: 10.5pt; line-height: 1.65; color: #000; margin: 0; }}
h1 {{ font-size: 17pt; text-align: center; margin: 0 0 4px; }} .sub {{ text-align: center; font-size: 11pt; margin: 0 0 14px; }}
h2 {{ font-size: 12.5pt; margin: 20px 0 6px; break-after: avoid; }} h3 {{ font-size: 11pt; margin: 14px 0 4px; break-after: avoid; }}
p {{ margin: 4px 0 6px; text-align: justify; }} p.li {{ margin: 2px 0 2px 14px; }}
table {{ border-collapse: collapse; margin: 8px auto 2px; font-size: 9.3pt; }} .tbl {{ break-inside: avoid; }}
th, td {{ border: 1px solid #333; padding: 3px 7px; text-align: center; }} th {{ background: #f0f0f0; font-weight: 700; }}
.cap {{ text-align: center; font-size: 9.5pt; margin: 2px 0 10px; }}
figure {{ margin: 8px 0 10px; text-align: center; break-inside: avoid; }} figcaption {{ font-size: 9.5pt; margin-top: 2px; }}
code {{ font-family: "D2Coding", "Consolas", monospace; font-size: 9.3pt; }}
.hdr td {{ text-align: left; }} .hdr th {{ width: 60px; }} .ref p {{ text-align: left; margin: 2px 0; }}
</style></head><body>
<h1>1주차 과제 보고서 — 문제 정의와 데이터 이해</h1>
<p class="sub">Step 2 · sEMG 팔동작 예측 프로젝트 (손동작 × 팔꿈치 각도 동시 분류)</p>
<table class="hdr" style="width:100%"><tr><th>학 과</th><td>컴퓨터소프트웨어학부</td><th>학 번</th><td>22114553</td></tr>
<tr><th>이 름</th><td>전진성</td><th>제출일</th><td>{SUBMIT_DATE}</td></tr>
<tr><th>저장소</th><td colspan="3">https://github.com/jjs523/semg-gesture-angle &nbsp;(노트북 <code>notebooks/week1.ipynb</code>, 실행본 <code>week1_executed.ipynb</code>, 설계안 <code>docs/class_design.md</code>)</td></tr></table>

<h2>1. 환경 확인 (제출물 ①)</h2>
<p>Step 1 환경(연구실 GPU 서버, Anaconda Python 3.12)에 이번 주 추가 패키지 xgboost만 설치하였다. 1주차는 GPU를 쓰지 않는다. 확인 출력은 <code>results/week1/env.txt</code>와 노트북 첫 셀에 있다.</p>
{table(env_df, "표 1. 패키지 버전")}

<h2>2. 데이터 확보와 구조 파악 (제출물 ②)</h2>
<p>강의 자료는 논문 데이터가 요청 기반이라고 안내하지만, 논문 저자가 GitHub 저장소 <code>TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset</code>에 공개해 두었다(1.2 GB). <code>setup.sh</code>가 <code>data/raw/</code>에 clone하고, 64개 파일의 SHA-256을 <code>results/week1/file_summary.csv</code>에 기록해 두 재현 시 같은 데이터인지 확인할 수 있게 하였다. Step 1 데이터와 가장 다른 점은 <b>파일 하나가 시행 30개의 연속 녹음</b>이라는 것이다. 파일을 그대로 쓰면 시행 단위 분할을 할 수 없으므로, <code>mark</code> 열(파일당 정확히 30개, 간격 5.0 s)을 시행 시작으로 삼아 5,000샘플씩 잘라 시행 1,920개의 배열을 만들었다(<code>src/common.py</code>).</p>
{table(struct, "표 2. 데이터 구조 (64개 파일 전체 집계)")}
{table(cs2, "표 3. 클래스 × 피험자 시행 수", index=True)}
{table(ch, "표 4. 클래스·채널별 신호 표준편차 (mV, 시행 5 s 전체)", fmt={c: "{:.4f}" for c in ch.columns}, index=True)}
<p>표 4에서 손동작 축의 단서가 보인다. 주먹·펴기는 전완 채널(ED, FDS)이 0.03~0.07 mV로 크고, 회외는 상완이두근(BB)이 0.025~0.032 mV로 네 동작 중 가장 크다. 회내는 모든 채널이 0.012~0.02 mV로 작아 논문이 "미세한 근협응"이라고 설명한 것과 맞는다.</p>

<h2>3. 채널별 신호 그리기 (제출물 ③)</h2>
<figure><img src="{img('ch5_example.png')}" style="width:92%"><figcaption>그림 1. 피험자 a, 30_fist 첫 시행의 5채널 파형. 붉은 띠는 강의 가정의 동작 구간(2~3 s), 녹색 띠는 실제 활성 구간(1~3 s)</figcaption></figure>
<p>그림 1에서 세 가지가 보인다. 첫째, 주먹 쥐기에서는 ch2(총지신근)와 ch3(천지굴근)이 가장 크게 반응하고 ch4(이두근)도 작지만 뚜렷하게 활성된다. 둘째, ch5(삼두근)는 0.02 mV 이하로 작고 0.5 s 간격의 규칙적인 파동이 보이는데, 이는 근활성이 아니라 메트로놈이나 심전도 같은 주기 잡음으로 보인다. 셋째, 활성은 강의 가정인 2 s보다 약 1 s 앞서 시작한다. 한 시행만으로는 우연일 수 있어 1,920시행 전체로 확인하였다.</p>
<figure><img src="{img('envelope_all.png')}" style="width:100%"><figcaption>그림 2. 8클래스의 평균 RMS 포락선 (50 ms 구간, 8명 × 30시행 = 240시행 평균). 붉은 점선이 강의 가정 2~3 s</figcaption></figure>
{table(mw2, "표 5. 활성 구간 (시행별 기저선 4~5 s 평균 + 3σ 임계값, 클래스별 중앙값, mark 기준 초)", index=True)}
<p><b>동작 구간은 강의 가정과 다르다.</b> 8클래스 모두 활성이 mark 뒤 약 0.7 s에 시작해 1.6~1.9 s에 최대가 되고 3.1~3.4 s에 끝난다(표 5). 메트로놈 박자에 맞춰 피험자가 신호보다 먼저 움직이기 시작한 것으로 보인다. 논문은 2 s 분석 세그먼트의 시작점을 명시하지 않았고 2주차 실습 자료는 1.5 s를 가정하는데, 1.5~3.5 s로 자르면 활성의 상승부 절반을 잃는다. 따라서 <b>2주차 세그먼트는 mark + 1.0~3.0 s</b>로 잡고 재현 가정으로 기록한다. 이 구간은 최대점(1.68 s)을 가운데에 두고 시작(0.72 s)과 끝(3.22 s)의 대부분을 덮는다.</p>
<p>그림 2는 채널 역할도 보여 준다. 펴기는 ch3(FDS)과 ch2(ED)가 0.12~0.15 mV로 가장 크고, 주먹은 ch2·ch3·ch1 순이며, 회외는 ch4(BB)가 유일하게 두드러지고, 회내는 다섯 채널이 모두 0.03 mV 아래에서 비슷하다. 즉 동작 축은 전완 3채널의 진폭 비율로, 회외는 이두근으로 갈리고, 회내는 뚜렷한 단서가 없어 다른 동작과 가장 헷갈릴 후보다.</p>

<h2>4. 같은 동작의 30°와 60° 비교 (제출물 ④)</h2>
<figure><img src="{img('angle_compare.png')}" style="width:90%"><figcaption>그림 3. 동작 4종 × 채널 5개의 30°(파랑)·60°(빨강) 평균 포락선 (8명 평균). 제목의 비율과 p값은 1~3 s RMS의 8명 대응표본 t-검정</figcaption></figure>
{table(at2, "표 6. 활성 구간(1~3 s) RMS의 30° vs 60° 비교 — 피험자별 평균 뒤 8명 대응표본 t-검정 (* p < 0.05)", fmt={"30° RMS (mV)": "{:.4f}", "60° RMS (mV)": "{:.4f}"}, allow_break=True)}
<figure><img src="{img('angle_per_participant.png')}" style="width:100%"><figcaption>그림 4. 상완 두 채널(BB 빨강, TB 보라)의 피험자별 30°→60° RMS 변화</figcaption></figure>
<p><b>관찰 1 — 각도 단서는 상완이두근(BB)에 있고, 동작마다 크기가 다르다.</b> 60°에서 BB는 회외 +55 %(p = 0.011), 펴기 +28 %(p = 0.035)로 뚜렷이 커지지만, 주먹은 +7 %(p = 0.62), 회내는 +12 %(p = 0.21)로 유의하지 않다. 팔꿈치를 60°로 굽혀 유지하면 이두근의 기저 긴장이 커지고, 회외는 이두근이 직접 관여하는 동작이라 차이가 가장 증폭된다. 반면 주먹·회내는 전완 근육이 주도해서 상완 채널에 각도 흔적이 적다. 논문 혼동행렬의 최대 오류가 60° 회내 → 30° 회내였던 것과 일치하는 결과다.</p>
<p><b>관찰 2 — 60°에서는 활성이 0.1~0.2 s 일찍 시작한다.</b> 그림 3의 붉은 선은 20개 칸 거의 모두에서 파란 선보다 왼쪽에 있고, 표 5에서도 60° 클래스의 활성 시작이 0.72~0.78 s로 30°(0.48~0.92 s)보다 대체로 이르다. 팔꿈치를 더 굽힌 자세에서 동작 준비가 빨라지는 것으로 보이며, 진폭이 같은 주먹에서도 이 시간 차이는 남는다. 각도 정보가 진폭뿐 아니라 <b>시간 구조</b>에 있다는 뜻이므로, 2주차에 2 s 세그먼트 안 윈도우의 순서를 유지하면 각도 구분에 도움이 될 수 있다.</p>
<p><b>관찰 3 — 각도 효과는 피험자에 따라 방향이 다르다.</b> 그림 4에서 회외 BB는 8명 중 7명이 60°에서 크지만, 주먹 BB는 4명이 커지고 4명이 작아진다. 삼두근(TB)은 모든 동작에서 60°가 큰 피험자가 5~6명에 그쳐 유의하지 않다. 각도 축은 동작 축보다 피험자 간 일관성이 낮은 축이고, 이것이 피험자 독립(LOSO) 검증에서 각도 정확도가 먼저 무너질 것이라고 예상하는 근거다.</p>

<h2>5. 클래스 설계안 (제출물 ⑤)</h2>
{md_to_html(design)}

<h2>6. 제출물 대조</h2>
{table(pd.DataFrame([
 ["① 환경 확인", "results/week1/env.txt, 표 1, 노트북 첫 셀"], ["② 데이터 요약", "results/week1/file_summary.csv · class_summary.csv · channel_stats.csv, 표 2~4"],
 ["③ 신호 그림 (동작 구간 표시)", "results/week1/ch5_example.png · envelope_all.png, 그림 1·2"], ["④ 비교 관찰 (30° vs 60°)", "results/week1/angle_compare.png · angle_rms_table.csv, 4절 관찰 1~3"],
 ["⑤ 클래스 설계안 (5항목)", "docs/class_design.md, 5절"], ["제출 형식", "notebooks/week1.ipynb + 이 PDF + 저장소 링크"],
], columns=["슬라이드 16쪽 제출물", "산출물"]), "표 7. 제출물 대응표")}
<p>강의 자료 14쪽 체크포인트 6개(xgboost import, 파일 수·클래스 분포, 신호 길이·5채널, ch5_example.png, 동작 구간 관찰, 설계안)는 모두 노트북 실행본과 2~5절에서 확인하였다.</p>

</body></html>"""
os.makedirs(OUT_DIR, exist_ok=True)
hp = os.path.join(OUT_DIR, "week1_report.html"); open(hp, "w", encoding="utf-8").write(html); print("HTML", len(html) // 1024, "KB")
if "--pdf" in sys.argv:
    os.makedirs(WIN_TMP, exist_ok=True); shutil.copy(hp, os.path.join(WIN_TMP, "week1_report.html"))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-margins", r"--print-to-pdf=C:\Users\user\semg_report_tmp\week1_report.pdf",
                    "--virtual-time-budget=10000", "file:///C:/Users/user/semg_report_tmp/week1_report.html"], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = os.path.join(OUT_DIR, "Step2_1주차과제_보고서_22114553_전진성.pdf"); shutil.copy(os.path.join(WIN_TMP, "week1_report.pdf"), out)
    print("PDF", out, os.path.getsize(out) // 1024, "KB")
