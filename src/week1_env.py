"""1주차 단계 1 — 환경 점검 (제출물 ①). 버전을 results/week1/env.txt 에 남긴다."""
import os, platform, sys
import numpy, scipy, sklearn, pandas, matplotlib
try:
    import xgboost
    xgb = xgboost.__version__
except ImportError:
    xgb = "미설치 (pip install xgboost)"
lines = [f"python  {sys.version.split()[0]} ({platform.system()} {platform.release()})",
         f"numpy   {numpy.__version__}", f"scipy   {scipy.__version__}", f"sklearn {sklearn.__version__}",
         f"pandas  {pandas.__version__}", f"matplotlib {matplotlib.__version__}", f"xgboost {xgb}"]
print("\n".join(lines))
os.makedirs("results/week1", exist_ok=True)
open("results/week1/env.txt", "w").write("\n".join(lines) + "\n")
