#!/usr/bin/env bash
# Runs the paper's XnYGZip measurements: 7 relative bounds x {fast, default, exhaustive} on every dataset found in
# DATA_DIR, first the strict verification (compression ratio, PSNR, error bound), then the throughput benchmark.
#
#   DATA_DIR=/path/to/data OUT=results scripts/run_all.sh
#
# Datasets (float32 x y z triples, no header) are looked up as DATA_DIR/<name>.bin:
#   usgs2 hacc warpx usgs vesicles lammps
# Missing files are skipped. Environment: GPU (device index, default 0), DATASETS, BOUNDS, SEARCH, WARMUP, REPS,
# SKIP_VERIFY=1, SKIP_BENCH=1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the directory with the .bin datasets}"
OUT="${OUT:-$HERE/results}"
DATASETS="${DATASETS:-usgs2 hacc warpx usgs vesicles lammps}"
BOUNDS="${BOUNDS:-1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2}"
SEARCH="${SEARCH:-fast,default,exhaustive}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv > "$OUT/gpu.txt"

for ds in $DATASETS; do
  f="$DATA_DIR/$ds.bin"
  if [ ! -f "$f" ]; then echo "skip $ds (no $f)"; continue; fi
  if [ "${SKIP_VERIFY:-0}" != 1 ]; then
    "$HERE/bin/xnygzip_verify" "$f" "$ds" "$OUT/verify.csv" --R="$BOUNDS" --search="$SEARCH" || echo "verification FAILED on $ds"
  fi
  if [ "${SKIP_BENCH:-0}" != 1 ]; then
    "$HERE/bin/xnygzip_bench" "$f" "$ds" "$OUT/bench.csv" --R="$BOUNDS" --search="$SEARCH" \
      --warmup="${WARMUP:-10}" --reps="${REPS:-30}"
  fi
done
python3 "$HERE/scripts/summarize.py" "$OUT"
