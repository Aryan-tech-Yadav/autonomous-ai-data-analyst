#!/usr/bin/env bash

set -Eeuo pipefail

cd "$(dirname "$0")/.."

echo "======================================"
echo " AI DATA ANALYST - VALIDATION"
echo "======================================"

echo
echo "[1/3] Checking Python syntax..."
python -m compileall -q app.py src tests
echo "PASS: Python syntax"

echo
echo "[2/3] Running automated tests..."
PYTHONPATH=. pytest -q
echo "PASS: Automated tests"

echo
echo "[3/3] Checking Git patch..."
git diff --check
echo "PASS: Git diff check"

echo
echo "======================================"
echo " ALL VALIDATION CHECKS PASSED"
echo "======================================"
