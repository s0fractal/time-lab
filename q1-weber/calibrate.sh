#!/bin/bash
# Calibration for q1-weber: the full n grid, 8 lists per n, one model at a time (memory).
set -eu
G=8,11,16,23,32,45,64,91,128,181,256,362,512
for m in gemma4:26b-a4b-it-q4_K_M qwen3.8:27b-mlx gemma3:27b; do
  python3 -I q1-weber/probe.py "$m" "$G" 8 "q1-weber/calibration/grid-$(echo $m | tr ':/' '--').json" > /dev/null
  echo "done $m"
done
