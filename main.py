"""Step 2 단일 진입점.   python main.py {env|week1|week2}"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {"env": ["week1_env.py"], "week1": ["week1_env.py", "week1_data.py", "week1_plot.py"],
         "week2": ["week2_features.py", "week2_checks.py"]}
if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
    raise SystemExit(f"사용: python main.py {{{'|'.join(STEPS)}}}")
for s in STEPS[sys.argv[1]]:
    print(f"\n===== {s}")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", s)], check=True, cwd=ROOT)
