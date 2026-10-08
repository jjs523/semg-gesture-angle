#!/bin/bash
# Step 2 환경 준비: 가상환경 + 패키지 + 데이터 clone (논문 저자 공개 저장소)
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
if [ ! -d data/raw/data ]; then
  git clone --depth 1 https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset data/raw
fi
echo "데이터 파일 수: $(ls data/raw/data/*.csv | wc -l)  (기대 64)"
