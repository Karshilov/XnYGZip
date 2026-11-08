# XnYGZip artifact

GPU-native, error-bounded compressor for float32 point clouds. This artifact reproduces the paper's two key
measurements of XnYGZip: **compression ratio** (with a strict per-point Euclidean error check and PSNR) and
**GPU-resident compression / decompression throughput**.

## Contents

| Path | Purpose |
|---|---|
| `xnygzip.cu` | The codec (single file): library API and a command-line compressor / decompressor |
| `bench/bench.cu` | Compression ratio and throughput (CUDA events, median of 30 after 10 warm-up runs) |
| `bench/verify.cu` | Strict error-bound verification, compression ratio and PSNR |
| `bench/common.h` | Input loading and error-bound helpers shared by both programs |
| `scripts/run_all.sh` | Runs both programs over all datasets, bounds and search budgets |
| `scripts/summarize.py` | Summary tables (standard library only) |
| `expected/paper_xnygzip.csv` | The paper's XnYGZip values (RTX PRO 6000 and A100), for comparison |
| `Makefile` | Build |

## Requirements

- NVIDIA GPU with compute capability 8.0 or newer. The paper used an RTX PRO 6000 Blackwell Max-Q (96 GB, `sm_120`)
  and an A100-SXM4 (40 GB, `sm_80`).
- CUDA 12.8 (nvcc), a C++20 host compiler with OpenMP (paper: GCC 13.3), Python 3 for the summary.
- CUB from CCCL. The paper used CCCL 3.5.0 (`make CCCL=<path to cccl-3.5.0>`); without `CCCL=` the toolkit's
  bundled CUB is used. Compressed streams do not depend on the CUB version.
- Device memory: about 37 B per point for the benchmark (input, worst-case chunk slots and output stream), e.g.
  39 GB for the 1.07 billion-point HACC snapshot. Host memory: about 50 B per point for the verifier.

## Build

```
make ARCH=sm_120 CCCL=/path/to/cccl-3.5.0     # RTX PRO 6000 (Blackwell)
make ARCH=sm_80  CCCL=/path/to/cccl-3.5.0     # A100
```

This builds `bin/xnygzip` (command-line tool), `bin/xnygzip_bench` and `bin/xnygzip_verify`.

## Datasets

Every input is a raw file of float32 `x y z` triples (12 bytes per point, no header). `scripts/run_all.sh` looks
for `DATA_DIR/<name>.bin` with the names below and skips missing files.

| Name | Domain | Points | Size |
|---|---|---|---|
| `usgs2` | LiDAR | 734,077,453 | 8.81 GB |
| `hacc` | Cosmology | 1,073,726,487 | 12.88 GB |
| `warpx` | Plasma simulation | 240,934,232 | 2.89 GB |
| `usgs` | LiDAR | 15,291,339 | 183.5 MB |
| `vesicles` | Biological imaging | 1,748,952 | 21.0 MB |
| `lammps` | Molecular dynamics | 67,108,864 | 805.3 MB |

<!-- TODO: dataset download / preparation instructions -->

## Running

```
DATA_DIR=/path/to/data OUT=results scripts/run_all.sh
```

For every dataset, bound `R` in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} and search budget in
{fast, default, exhaustive}, the script runs the verifier (`results/verify.csv`), then the benchmark
(`results/bench.csv`), and finally writes `results/summary.txt`. Options (environment variables): `GPU` (device
index), `DATASETS`, `BOUNDS`, `SEARCH`, `WARMUP`, `REPS`, `SKIP_VERIFY=1`, `SKIP_BENCH=1`. Throughput
measurements need an otherwise idle GPU.

The programs can also be run directly:

```
bin/xnygzip_verify <in.bin> <name> <out.csv> [--R=1e-5,1e-3] [--search=fast,default,exhaustive]
bin/xnygzip_bench  <in.bin> <name> <out.csv> [--R=1e-5,1e-3] [--search=default] [--warmup=10] [--reps=30]
```

Rows are appended to `out.csv`.

## What is measured

- **Error bound.** The relative bound `R` gives the absolute Euclidean bound `eps = R x diag`, where `diag` is the
  diagonal of the input's bounding box. A cell passes only if every reconstructed point lies within `eps` of its
  original (double precision, no tolerance).
- **Compression ratio** = input bytes / compressed file bytes. The file bytes include the 56-byte file header and
  the 2-byte size of every 1,024-point chunk. The bitstream is deterministic, so compression ratios match the paper
  exactly on any GPU (`summarize.py` compares them with `expected/paper_xnygzip.csv`). `bench.csv` also records a
  hash of every stream.
- **Verification** (`verify.cu`). The decoder reorders points within each chunk, so the check runs per chunk:
  - Lossless chunks must decode to exactly the original points.
  - Every decoded exception must equal a distinct original point exactly.
  - Every other original point is re-quantized on the CPU. The multiset of these lattice points must equal the
    multiset of decoded lattice points, and every decoded coordinate must be bit-identical to its lattice point
    scaled and offset by the chunk origin.
  - The Euclidean error of every point is computed from this reconstruction.

  PSNR = 20 log10(diag / RMSE), with the RMSE over the pointwise Euclidean errors.
- **Throughput** (`bench.cu`) = input bytes / time, with 1 GB = 1e9 B. Input and output stay in device memory;
  file I/O and host-device copies are excluded.
  - Compression covers the sampling pass, chunk encoding, the device-wide prefix scan and the stream assembly.
  - Decompression covers the prefix scan of the chunk size table and the decoding kernel.
  - Times are CUDA-event medians of 30 runs after 10 warm-up runs. All search budgets of one dataset are timed in
    the same process.
- **Search budgets.**
  - `fast`: one mode, chosen by sampling.
  - `default`: a shortlist of up to three modes, each chosen by at least 10% of the sampled chunks.
  - `exhaustive`: every mode for every chunk.

  Inputs that fit in one resident wave of the GPU always use the exhaustive search.

## Mapping to the paper

| Paper | Output |
|---|---|
| Rate-distortion comparison (compression ratio / bpp and PSNR of XnYGZip) | `verify.csv`, *compression ratio* table |
| Compression and decompression throughput (default and fast configurations) | `bench.csv`, *throughput* tables |
| Table II (fast / default / exhaustive compression throughput, size within 0.33% of exhaustive) | *search budget* table |

Throughput depends on the GPU model, clocks and driver; on the paper's GPUs the values should be close to
`expected/paper_xnygzip.csv`.

## Command-line tool

```
bin/xnygzip compress   <in.bin> <eps | rel=R> <out.xng> [--search=fast|default|exhaustive]
bin/xnygzip decompress <in.xng> <out.bin>
```

`rel=R` sets `eps = R x` bounding-box diagonal. The decompressed file uses the same raw float32 format; points
keep their chunk but may be reordered within it.
