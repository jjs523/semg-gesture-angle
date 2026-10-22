"""Step 2 단일 진입점.   python main.py            -> 1~3주차 전체
                       python main.py week3      -> 3주차만 (env|week1|week2|week3|all)"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {"env": ["week1_env.py"], "week1": ["week1_env.py", "week1_data.py", "week1_plot.py"],
         "week2": ["week2_features.py", "week2_checks.py"],
         "week3": ["week3_run.py", "week3_sample_unit.py", "week3_analyze.py"],
         "all": ["week1_env.py", "week1_data.py", "week1_plot.py", "week2_features.py", "week2_checks.py", "week3_run.py", "week3_sample_unit.py", "week3_analyze.py"]}
step = sys.argv[1] if len(sys.argv) > 1 else "all"           # 인자 없이 실행하면 1~3주차 전체
if step not in STEPS:
    raise SystemExit(f"사용: python main.py [{'|'.join(STEPS)}]   (인자 없으면 all)")
for s in STEPS[step]:
    print(f"\n===== {s}")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", s)], check=True, cwd=ROOT)
