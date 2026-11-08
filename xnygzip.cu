/**
 * xnygzip.cu - XnYGZip GPU point-cloud compressor.
 *
 * Error-bounded lossy compression of float32 xyz point clouds on the GPU: every reconstructed point lies within a
 * Euclidean distance eps of its original.
 *
 * Format:
 *   - Fine BCC quantization (the quantizer of XnYZip); a rigorous per-point L2 check stores every point that could
 *     violate the bound (float precision) as an exact exception.
 *   - Independent 1024-point blocks (one CUDA block of 128 threads, 8 points per thread, each). A block is coded as
 *       flat     linearized fine-lattice cell keys (delta coded) + per-cell multiplicities;
 *       TO k     the same over the coarse lattice 2^k BCC (truncated-octahedron cells; translation-equivariant
 *                integer nearest-point rule, so each cell holds one point of each of the 8^k cosets) + a 3k-bit
 *                coset index per point, k = 1..6;
 *       RAWQ     raw quantized coordinates;   LOSSLESS  the float32 originals.
 *     Deltas and counts use per-group bit widths (groups of 32, 64, 128 or 1024, chosen per block).
 *   - Mode selection: a sampling pass runs the exhaustive search (every mode) on sampled blocks of each chunk and
 *     keeps a shortlist of the most frequent modes; every block then evaluates the shortlist exactly and keeps the
 *     smallest. Search budgets: fast (shortlist of 1), default (up to 3), exhaustive (every mode for every block).
 *     Inputs that fit in one resident wave of the GPU always use the exhaustive search.
 *
 * Build: nvcc -O3 -arch=sm_XX -std=c++17 [-I<cccl>/cub ...] xnygzip.cu
 *   XNYGZIP_MINBLOCKS   resident blocks per SM requested from the compiler (default 4; GPU-dependent tuning)
 *   XNYGZIP_NO_MAIN     omit the command-line tool (when included as a library)
 */
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#define XNYG_CHECK(expr)                                                                                     \
    do {                                                                                                     \
        cudaError_t _e = (expr);                                                                             \
        if (_e != cudaSuccess) {                                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                                                         \
        }                                                                                                    \
    } while (0)

#ifndef XNYGZIP_MINBLOCKS
#define XNYGZIP_MINBLOCKS 4   // 4 x 128 threads per SM: needs <= 128 registers and <= ~24 KB shared memory per block
#endif

namespace xnyg {

constexpr int BT = 128;                 // threads per CUDA block
constexpr int IPT = 8;                  // points per thread
constexpr int NB = BT * IPT;            // points per compression block
constexpr int MAX_BLOCK_BYTES = 12544;  // >= lossless block (12288 B) + header
constexpr int SLOT_BYTES = MAX_BLOCK_BYTES;
constexpr int STAGE_WORDS = MAX_BLOCK_BYTES / 4;
constexpr int KMAX = 6;
constexpr int LIMIT_BITS = 12288 * 8;   // larger payloads fall back to lossless

// block modes (4-bit field; 7..13 unused)
enum : int { M_FLAT = 0, M_RAWQ = 14, M_LOSSLESS = 15 };
__host__ __device__ inline bool is_to(int m) { return m >= 1 && m <= KMAX; }
__host__ __device__ inline bool is_lattice(int m) { return m >= 0 && m <= KMAX; }

// mode selection: exhaustive search over every mode, or the shortlist chosen by the sampling pass
enum Select { SEL_COST = 3, SEL_LIST = 5 };

// file header word: a fixed format identifier plus the (informational) search settings select << 8, adapt << 28
constexpr uint32_t FORMAT_WORD = 0x801310D5u;
constexpr uint32_t SETTINGS_MASK = 0x7u << 8 | 0x7u << 28;

struct Config {
    int select = SEL_LIST;  // SEL_LIST: shortlist from the sampling pass; SEL_COST: exhaustive search for every block
    int adapt = 3;          // shortlist length (1: fast, 3: default)
    int fth = 10;           // shortlist modes after the first must have won >= fth % of the sampled blocks
    int sample = 0;         // internal: > 0 -> sampling launch, CUDA block b encodes block b * sample (size only)
    int direct = 0;         // internal: device-chosen output path of the main launch (1 = single-pass compact stream)
    int list = 0;           // internal: SEL_LIST candidates, evaluated in order: bits 0-1 count (1..3), then 3 x 3-bit
                            // mode codes (0 flat, k = TO level k), least frequent first. Written by k_pickmask.
    uint32_t pack() const { return FORMAT_WORD | (uint32_t)select << 8 | (uint32_t)adapt << 28; }
};

// ---------------------------------------------------------------- bit helpers
__host__ __device__ __forceinline__ uint32_t bitw(uint32_t v) {
#ifdef __CUDA_ARCH__
    return v ? 32u - __clz(v) : 0u;
#else
    return v ? 32u - (uint32_t)__builtin_clz(v) : 0u;
#endif
}
__host__ __device__ __forceinline__ uint32_t bitw64(uint64_t v) {
#ifdef __CUDA_ARCH__
    return v ? 64u - (uint32_t)__clzll((long long)v) : 0u;
#else
    return v ? 64u - (uint32_t)__builtin_clzll(v) : 0u;
#endif
}
__host__ __device__ __forceinline__ uint32_t lowmask(uint32_t n) { return n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u); }
__host__ __device__ __forceinline__ uint32_t zigzag(int32_t v) { return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31); }
__host__ __device__ __forceinline__ int32_t unzigzag(uint32_t u) { return (int32_t)(u >> 1) ^ -(int32_t)(u & 1); }

__device__ __forceinline__ void put_bits(uint32_t* w, uint32_t pos, uint32_t val, uint32_t n) {
    if (!n) return;
    val &= lowmask(n);
    uint32_t wi = pos >> 5, bo = pos & 31u;
    atomicOr(&w[wi], val << bo);
    if (bo + n > 32) atomicOr(&w[wi + 1], val >> (32 - bo));
}
__host__ __device__ __forceinline__ uint32_t get_bits(const uint32_t* w, uint64_t pos, uint32_t n) {
    if (!n) return 0;
    uint64_t wi = pos >> 5;
    uint32_t bo = (uint32_t)(pos & 31u);
    uint64_t v = (uint64_t)w[wi] >> bo;
    if (bo + n > 32) v |= (uint64_t)w[wi + 1] << (32 - bo);
    return (uint32_t)v & lowmask(n);
}
// up to 64-bit fields (cell keys / deltas): low word first
__device__ __forceinline__ void put_bits64(uint32_t* w, uint32_t pos, uint64_t val, uint32_t n) {
    if (n <= 32) { put_bits(w, pos, (uint32_t)val, n); return; }
    put_bits(w, pos, (uint32_t)val, 32);
    put_bits(w, pos + 32, (uint32_t)(val >> 32), n - 32);
}
__host__ __device__ __forceinline__ uint64_t get_bits64(const uint32_t* w, uint64_t pos, uint32_t n) {
    if (n <= 32) return get_bits(w, pos, n);
    return (uint64_t)get_bits(w, pos, 32) | (uint64_t)get_bits(w, pos + 32, n - 32) << 32;
}

// ---------------------------------------------------------- fine BCC quantizer
__host__ __device__ __forceinline__ void bcc_nearest_f(float px, float py, float pz, int32_t& qx, int32_t& qy, int32_t& qz) {
#ifdef __CUDA_ARCH__
    float ex = rintf(px * 0.5f) * 2.0f, ey = rintf(py * 0.5f) * 2.0f, ez = rintf(pz * 0.5f) * 2.0f;
#else
    float ex = std::nearbyint(px * 0.5f) * 2.0f, ey = std::nearbyint(py * 0.5f) * 2.0f, ez = std::nearbyint(pz * 0.5f) * 2.0f;
#endif
    float dx = px - ex, dy = py - ey, dz = pz - ez;
    bool use_even = (fabsf(dx) + fabsf(dy) + fabsf(dz) <= 1.5f);
    float ox = ex + copysignf(1.0f, dx), oy = ey + copysignf(1.0f, dy), oz = ez + copysignf(1.0f, dz);
    qx = (int32_t)(use_even ? ex : ox);
    qy = (int32_t)(use_even ? ey : oy);
    qz = (int32_t)(use_even ? ez : oz);
}

// ------------------------------------- coarse TO: equivariant integer nearest point
// Nearest point of m*BCC (m = 2^k) to the BCC point q, returned as the center c (in fine units). Per-axis rounding
// uses half-open residuals [-m, m); E/O ties are broken by the lexicographically larger residual. The rule depends
// only on the residual, so nearest(q + m t) = nearest(q) + m t for every t in BCC, and each coarse cell is a
// fundamental domain: it contains exactly m^3 fine points, one per coset of BCC / (m BCC).
__host__ __device__ __forceinline__ void to_center_int(int32_t qx, int32_t qy, int32_t qz, int k,
                                                       int32_t& cx, int32_t& cy, int32_t& cz) {
    const int32_t m = 1 << k, s = k + 1;
    int32_t ex = ((qx + m) >> s) << s, ey = ((qy + m) >> s) << s, ez = ((qz + m) >> s) << s;
    int32_t ox = ((qx >> s) << s) + m, oy = ((qy >> s) << s) + m, oz = ((qz >> s) << s) + m;
    int32_t rex = qx - ex, rey = qy - ey, rez = qz - ez, rox = qx - ox, roy = qy - oy, roz = qz - oz;
    int32_t dE = rex * rex + rey * rey + rez * rez, dO = rox * rox + roy * roy + roz * roz;
    bool useE;
    if (dE != dO) useE = dE < dO;
    else if (rex != rox) useE = rex > rox;
    else if (rey != roy) useE = rey > roy;
    else useE = rez > roz;
    cx = useE ? ex : ox; cy = useE ? ey : oy; cz = useE ? ez : oz;
}
// coset index of BCC point q modulo m*BCC: lattice coords (a,b,c) with q = a(2,0,0) + b(0,2,0) + c(1,1,1), digits
// taken mod m
__host__ __device__ __forceinline__ uint32_t to_cidx(int32_t qx, int32_t qy, int32_t qz, int k) {
    const uint32_t mk = (1u << k) - 1u;
    uint32_t a = (uint32_t)((qx - qz) >> 1) & mk, b = (uint32_t)((qy - qz) >> 1) & mk, c = (uint32_t)qz & mk;
    return a | (b << k) | (c << (2 * k));
}
// inverse: coarse BCC coords s (center = m*s) + coset index -> fine q
__host__ __device__ __forceinline__ void to_decode(int32_t sx, int32_t sy, int32_t sz, uint32_t idx, int k,
                                                   int32_t& qx, int32_t& qy, int32_t& qz) {
    const uint32_t mk = (1u << k) - 1u;
    int32_t a = (int32_t)(idx & mk), b = (int32_t)((idx >> k) & mk), c = (int32_t)((idx >> (2 * k)) & mk);
    int32_t q0x = 2 * a + c, q0y = 2 * b + c, q0z = c, cx, cy, cz;
    to_center_int(q0x, q0y, q0z, k, cx, cy, cz);
    qx = (sx << k) + q0x - cx; qy = (sy << k) + q0y - cy; qz = (sz << k) + q0z - cz;
}

// --------------------------------------------------------------- block header
struct Hdr {
    uint32_t origin[3];        // float bits of the reconstruction origin
    int mode, n_exc, nu, gcode;
    uint32_t Rx, Ry;           // key box extents (x, y)
    uint64_t firstkey;         // up to 63 bits (uv field: 6-bit length)
    int32_t mn[3];             // key box origin
    int dbw, cbw, rbw;         // RAWQ widths (x, y, z)
    int ngd, ngc;              // delta / count groups
    uint8_t wd[32], wc[32];    // per-group widths
};

struct SmemSink {
    uint32_t* w; uint32_t pos;
    __device__ void put(uint32_t v, uint32_t b) {
        if (!b) return;
        v &= lowmask(b);
        uint32_t wi = pos >> 5, bo = pos & 31u;
        w[wi] |= v << bo;
        if (bo + b > 32) w[wi + 1] |= v >> (32 - bo);
        pos += b;
    }
};
struct GlobReader {
    const uint32_t* w; uint64_t pos;
    __host__ __device__ uint32_t get(uint32_t b) { uint32_t v = get_bits(w, pos, b); pos += b; return v; }
};

template <class S> __host__ __device__ void put_uv(S& s, uint32_t v) { uint32_t L = bitw(v); s.put(L, 6); s.put(v, L); }
template <class R> __host__ __device__ uint32_t get_uv(R& r) { uint32_t L = r.get(6); return r.get(L); }
// 64-bit variant: same bit layout (6-bit length L, then L bits, low word first); L <= 63
template <class S> __host__ __device__ void put_uv64(S& s, uint64_t v) {
    uint32_t L = bitw64(v); s.put(L, 6);
    if (L <= 32) s.put((uint32_t)v, L); else { s.put((uint32_t)v, 32); s.put((uint32_t)(v >> 32), L - 32); }
}
template <class R> __host__ __device__ uint64_t get_uv64(R& r) {
    uint32_t L = r.get(6);
    if (L <= 32) return r.get(L);
    uint64_t lo = r.get(32); return lo | (uint64_t)r.get(L - 32) << 32;
}
template <class S> __host__ __device__ void put_sv(S& s, int32_t v) {
    if (v == 0) s.put(0, 2); else if (v == -1) s.put(1, 2); else if (v == 1) s.put(2, 2);
    else { s.put(3, 2); put_uv(s, zigzag(v)); }
}
template <class R> __host__ __device__ int32_t get_sv(R& r) {
    uint32_t c = r.get(2);
    return c == 0 ? 0 : c == 1 ? -1 : c == 2 ? 1 : unzigzag(get_uv(r));
}

__host__ __device__ inline int gsize_of(int gcode) { return gcode == 3 ? 1024 : (32 << gcode); }

// header: origin (3 x 32), mode (4), exception flag (1) [+ count - 1 (10)]; RAWQ: box origin (3 sv) + widths
// (3 x 6); lattice: cells - 1 (10), box extents - 1 (2 uv), box origin (3 sv), group code (2), first key (uv64),
// then the delta (6 bits) and count (4 bits) group widths
template <class S>
__device__ void header_emit(S& s, const Hdr& h) {
    for (int a = 0; a < 3; ++a) s.put(h.origin[a], 32);
    s.put(h.mode, 4);
    s.put(h.n_exc ? 1 : 0, 1);
    if (h.n_exc) s.put(h.n_exc - 1, 10);
    if (h.mode == M_LOSSLESS) return;
    if (h.mode == M_RAWQ) {
        for (int a = 0; a < 3; ++a) put_sv(s, h.mn[a]);
        s.put(h.dbw, 6); s.put(h.cbw, 6); s.put(h.rbw, 6);
        return;
    }
    s.put(h.nu - 1, 10);
    put_uv(s, h.Rx - 1); put_uv(s, h.Ry - 1);
    for (int a = 0; a < 3; ++a) put_sv(s, h.mn[a]);
    s.put(h.gcode, 2); put_uv64(s, h.firstkey);
    for (int g = 0; g < h.ngd; ++g) s.put(h.wd[g], 6);
    for (int g = 0; g < h.ngc; ++g) s.put(h.wc[g], 4);
}
// parses everything up to the group widths (the decoder reads those in parallel)
template <class R>
__host__ __device__ void header_parse(R& r, Hdr& h) {
    for (int a = 0; a < 3; ++a) h.origin[a] = r.get(32);
    h.mode = r.get(4);
    h.n_exc = r.get(1) ? (int)r.get(10) + 1 : 0;
    h.ngd = h.ngc = 0; h.gcode = 0; h.firstkey = 0;
    if (h.mode == M_LOSSLESS) return;
    if (h.mode == M_RAWQ) {
        for (int a = 0; a < 3; ++a) h.mn[a] = get_sv(r);
        h.dbw = min((int)r.get(6), 32); h.cbw = min((int)r.get(6), 32); h.rbw = min((int)r.get(6), 32);
        return;
    }
    h.nu = r.get(10) + 1;
    h.Rx = get_uv(r) + 1; h.Ry = get_uv(r) + 1;
    for (int a = 0; a < 3; ++a) h.mn[a] = get_sv(r);
    h.gcode = r.get(2); h.firstkey = get_uv64(r);
    const int G = gsize_of(h.gcode);
    h.ngd = min((h.nu - 1 + G - 1) / G, 32); h.ngc = min((h.nu + G - 1) / G, 32);
}

// residual bits per point of a mode
__host__ __device__ inline int resid_bits(int mode) { return is_to(mode) ? 3 * mode : 0; }

// per-block record: chosen mode and size breakdown (read by the sampling pass; optionally returned by compress())
struct BlockStat {
    uint8_t mode, gcode; uint16_t nu, n_exc, nvq;
    uint32_t hdr_bits, delta_bits, count_bits, res_bits, exc_bits, total_bytes;
};
static_assert(sizeof(BlockStat) == 32, "BlockStat must not contain implicit padding");

__constant__ Config c_cfg;

// output path selection: the sampling pass picks the single-pass (direct) path on the device when the sampled blocks
// average more than DIRECT_BPP bits per point; inputs below AUTO_MINBLOCKS blocks keep the slots path
constexpr int DIRECT_BPP = 12;
constexpr int AUTO_MINBLOCKS = 32768;

// rigorous |v - r|^2 <= eps^2 test with directed rounding
__device__ __forceinline__ bool within_bound(float vx, float vy, float vz, float rx, float ry, float rz, float eps2_rd) {
    const float f = 1.0f + 2.384185791015625e-07f;  // 1 + 2^-22 covers the rounding of v - r
    float dx = __fmul_ru(fabsf(vx - rx), f), dy = __fmul_ru(fabsf(vy - ry), f), dz = __fmul_ru(fabsf(vz - rz), f);
    float s = __fmaf_ru(dx, dx, __fmaf_ru(dy, dy, __fmul_ru(dz, dz)));
    return s <= eps2_rd;  // false for NaN / inf
}

// --------------------------------------------------------- block-level helpers
template <int N, bool TRAIL = true>
__device__ __forceinline__ void block_reduce(int32_t (&v)[N], const bool (&is_max)[N], int32_t* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int32_t x = is_max[j] ? __reduce_max_sync(0xFFFFFFFFu, v[j]) : __reduce_min_sync(0xFFFFFFFFu, v[j]);
        if (lane == 0) red[wid * N + j] = x;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < N; ++j) {
        int32_t x = red[j];
        for (int w = 1; w < BT / 32; ++w) x = is_max[j] ? max(x, red[w * N + j]) : min(x, red[w * N + j]);
        v[j] = x;
    }
    if (TRAIL) __syncthreads();
}
__device__ __forceinline__ void block_min3(float (&v)[3], float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        float x = v[a];
        for (int o = 16; o > 0; o >>= 1) x = fminf(x, __shfl_xor_sync(0xFFFFFFFFu, x, o));
        if (lane == 0) red[wid * 3 + a] = x;
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < 3; ++a) { float x = red[a]; for (int w = 1; w < BT / 32; ++w) x = fminf(x, red[w * 3 + a]); v[a] = x; }
}
__device__ __forceinline__ void block_sum3(float (&v)[3], float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        float x = v[a];
        for (int o = 16; o > 0; o >>= 1) { float y = __shfl_xor_sync(0xFFFFFFFFu, x, o); x = x + y; }
        if (lane == 0) red[wid * 3 + a] = x;
    }
    __syncthreads();
#pragma unroll
    for (int a = 0; a < 3; ++a) { float x = red[a]; for (int w = 1; w < BT / 32; ++w) x = x + red[w * 3 + a]; v[a] = x; }
}

constexpr int RADIX_BITS = 4;
typedef cub::BlockRadixSort<uint32_t, BT, IPT, uint32_t, RADIX_BITS> SortT;
typedef cub::BlockRadixSort<uint64_t, BT, IPT, uint32_t, RADIX_BITS> SortT64;     // keys wider than 32 bits
typedef cub::BlockRadixSort<uint32_t, BT, IPT, cub::NullType, RADIX_BITS> SortKT; // key-only (no residual)
// key-only candidates wider than MSORT_MIN bits use the match-ranked sort (MSORT_BITS-bit digits)
constexpr int MSORT_MIN = 16;
constexpr int MSORT_BITS = 6;
typedef cub::BlockScan<uint32_t, BT> ScanT;
typedef cub::BlockScan<uint64_t, BT> ScanT64;

// candidate encoding state
struct Cand {
    int mode = -1, valid = 0, rb = 0, gcode = 0, nu = 0, dbw = 0, cbw = 0, rbw = 0, keybits = 0;
    uint32_t Rx = 0, Ry = 0;
    uint64_t firstkey = 0;
    int32_t mn[3] = {0, 0, 0};
    uint32_t bits = 0xFFFFFFFFu, hdr_bits = 0, delta_bits = 0, count_bits = 0, res_bits = 0;
};

struct Smem {
    union {
        typename SortT::TempStorage sort;
        typename SortT64::TempStorage sort64;
        typename SortKT::TempStorage sortk;
        typename cub::BlockRadixRankMatch<BT, MSORT_BITS, false>::TempStorage rankm;   // match-ranked sort (keys in seg_key)
        typename ScanT::TempStorage scan;
        uint32_t stage[STAGE_WORDS + 2];   // emission
    } u;
    uint64_t seg_key[NB];          // distinct sorted keys (cells)
    uint16_t seg_start[NB + 1];    // first point of each cell
    uint64_t warp_last[BT / 32];   // last key of each warp (RLE boundary)
    int32_t qbox[6];               // fine box of the quantized points (min xyz, max xyz)
    int32_t red[BT / 32 * 7];
    float redf3[BT / 32 * 3];
    uint8_t w32d[32], w32c[32];    // delta / count widths of every 32 cells
    uint32_t goff_d[33], goff_c[33];
    uint32_t gsel[4][3];           // per group code: header, delta and count bits
    unsigned tk;                   // direct path: dynamic block index (launch-order ticket)
    unsigned long long lb_off;     // direct path: byte offset of this block in the compact stream
    Hdr hdr;
    // current best candidate (raw storage: Cand has default member initializers)
    alignas(8) unsigned char bc_raw[sizeof(Cand)];
    __device__ Cand& bc() { return *reinterpret_cast<Cand*>(bc_raw); }
};

__host__ __device__ __forceinline__ uint32_t uv_bits(uint32_t v) { return 6u + bitw(v); }
__host__ __device__ __forceinline__ uint32_t uv_bits64(uint64_t v) { return 6u + bitw64(v); }
__host__ __device__ __forceinline__ uint32_t sv_bits(int32_t v) { return (v >= -1 && v <= 1) ? 2u : 2u + uv_bits(zigzag(v)); }
// header size, mirrors header_emit()
__host__ __device__ inline uint32_t hdr_bits_of(const Cand& d, int n_exc, int ngd, int ngc) {
    uint32_t b = 96 + 4 + 1 + (n_exc ? 10 : 0);
    if (d.mode == M_LOSSLESS) return b;
    if (d.mode == M_RAWQ) return b + sv_bits(d.mn[0]) + sv_bits(d.mn[1]) + sv_bits(d.mn[2]) + 18;
    b += 10 + uv_bits(d.Rx - 1) + uv_bits(d.Ry - 1) + sv_bits(d.mn[0]) + sv_bits(d.mn[1]) + sv_bits(d.mn[2]);
    return b + 2 + uv_bits64(d.firstkey) + 6u * ngd + 4u * ngc;
}

__device__ unsigned int g_err;  // header size model mismatch counter (must stay 0)
// single-pass ("direct") emission: per block (state << 62 | value), state 1 = own size, 2 = inclusive prefix
// (decoupled look-back, as in single-pass prefix scans), and the launch-order ticket counter. Both zeroed per call.
__device__ unsigned long long* g_lb;
__device__ unsigned int* g_ticket;
constexpr unsigned long long LB_AGG = 1ull << 62, LB_INC = 2ull << 62, LB_VAL = (1ull << 62) - 1;

struct KeyInfo { int valid; uint32_t Rx, Ry; uint64_t lin; int32_t mn[3]; int rb; };
constexpr uint64_t KEY64_LIMIT = 0x7FFFFFFFFFFFFFFFull;  // lin <= 2^63 - 1: every key, delta and width fits 63 bits

// Decompose fine q into (cell, residual) for a lattice mode and build linearized cell keys. Parity linearization:
// cell x and y are halved (the z parity determines their low bit).
__device__ __forceinline__ KeyInfo build_keys(int mode, const int32_t (&q)[IPT][3], uint32_t qvmask, Smem& S,
                                              uint64_t (&keys)[IPT], uint32_t (&vals)[IPT]) {
    const int k = mode;   // 0 = flat
    int32_t cc[IPT][3];
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        vals[i] = 0;
        cc[i][0] = cc[i][1] = cc[i][2] = 0;
        if (!((qvmask >> i) & 1u)) continue;
        const int32_t qx = q[i][0], qy = q[i][1], qz = q[i][2];
        if (mode == M_FLAT) {
            cc[i][0] = qx; cc[i][1] = qy; cc[i][2] = qz;
        } else {
            int32_t cx, cy, cz;
            to_center_int(qx, qy, qz, k, cx, cy, cz);
            cc[i][0] = cx >> k; cc[i][1] = cy >> k; cc[i][2] = cz >> k;
            vals[i] = to_cidx(qx, qy, qz, k);
        }
        cc[i][0] >>= 1; cc[i][1] >>= 1;
    }
    int32_t r6[6] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN, INT32_MIN};
    const bool mx6[6] = {false, false, false, true, true, true};
    if (mode == M_FLAT) {   // the flat box is the fine box (x >> 1 is monotone): no reduction
        for (int a = 0; a < 6; ++a) r6[a] = S.qbox[a];
        r6[0] >>= 1; r6[1] >>= 1; r6[3] >>= 1; r6[4] >>= 1;
    } else {
#pragma unroll
        for (int i = 0; i < IPT; ++i)
            if ((qvmask >> i) & 1u)
                for (int a = 0; a < 3; ++a) { r6[a] = min(r6[a], cc[i][a]); r6[3 + a] = max(r6[3 + a], cc[i][a]); }
        block_reduce<6>(r6, mx6, S.red);
    }
    KeyInfo ki;
    const uint64_t Rx = (uint64_t)((int64_t)r6[3] - r6[0] + 1), Ry = (uint64_t)((int64_t)r6[4] - r6[1] + 1),
                   Rz = (uint64_t)((int64_t)r6[5] - r6[2] + 1);
    // overflow-free range test: every axis range < 2^32 (fine |q| < 2^31), so Rx * Ry < 2^64 is exact; lin =
    // (Rx*Ry)*Rz is exact iff the high 64 bits of the product are 0
    const bool axes_ok = Rx <= 0xFFFFFFFFull && Ry <= 0xFFFFFFFFull && Rz <= 0xFFFFFFFFull;
    const uint64_t rxy = Rx * Ry;
    const bool exact = axes_ok && __umul64hi(rxy, Rz) == 0;
    const uint64_t lin_ = rxy * Rz;
    ki.valid = exact && lin_ <= KEY64_LIMIT;
    ki.Rx = (uint32_t)Rx; ki.Ry = (uint32_t)Ry; ki.lin = ki.valid ? lin_ : 0;
    ki.mn[0] = r6[0]; ki.mn[1] = r6[1]; ki.mn[2] = r6[2];
    ki.rb = resid_bits(mode);
    if (!ki.valid) return ki;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
        keys[i] = ((qvmask >> i) & 1u)
                      ? ((uint64_t)(cc[i][2] - r6[2]) * Ry + (uint64_t)(cc[i][1] - r6[1])) * Rx + (uint64_t)(cc[i][0] - r6[0])
                      : ~0ull;
    return ki;
}

// ---------------------------------------------------------------- match-ranked key-only sort
// LSD passes ranked with cub::BlockRadixRankMatch; keys stay in the ranker's warp-striped order between passes and
// are scattered to the blocked arrangement in the last one (staged in S.seg_key)
template <class K> struct DigitBits {
    int s, n;
    __device__ __forceinline__ uint32_t Digit(K k) const { return (uint32_t)(k >> s) & ((1u << n) - 1u); }
};
template <class K>
__device__ __forceinline__ void match_sort(K (&keys)[IPT], int end_bit, Smem& S) {
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    K* kbuf = reinterpret_cast<K*>(S.seg_key);
    for (int b0 = 0; b0 < end_bit; b0 += MSORT_BITS) {
        int ranks[IPT];
        cub::BlockRadixRankMatch<BT, MSORT_BITS, false>(S.u.rankm).RankKeys(keys, ranks, DigitBits<K>{b0, min(MSORT_BITS, end_bit - b0)});
#pragma unroll
        for (int i = 0; i < IPT; ++i) kbuf[ranks[i]] = keys[i];
        __syncthreads();
        const bool last = b0 + MSORT_BITS >= end_bit;
#pragma unroll
        for (int i = 0; i < IPT; ++i) keys[i] = kbuf[last ? t * IPT + i : w * (32 * IPT) + i * 32 + lane];
    }
}

// Everything after the sort: run-length encoding into the segment table, 32-cell widths, the best group code and the
// exact encoded size.
__device__ void eval_post(int mode, const KeyInfo& ki, int end_bit, int nvq, int n_exc, Smem& S,
                          const uint64_t (&keys)[IPT], Cand& out) {
    const int t = threadIdx.x;
    // previous thread's last key: warp shuffle; warp boundaries through shared memory
    const uint64_t up = __shfl_up_sync(0xFFFFFFFFu, keys[IPT - 1], 1);
    if ((t & 31) == 31) S.warp_last[t >> 5] = keys[IPT - 1];
    __syncthreads();
    const uint64_t prev_thread = (t & 31) ? up : (t ? S.warp_last[(t >> 5) - 1] : 0);
    uint32_t trmask = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        const int p = t * IPT + i;
        if (p >= nvq) continue;
        const uint64_t prev = i ? keys[i - 1] : prev_thread;
        if (p == 0 || keys[i] != prev) trmask |= 1u << i;
    }
    uint32_t base, nu;
    ScanT(S.u.scan).ExclusiveSum((uint32_t)__popc(trmask), base, nu);
    {
        uint32_t s = base;
#pragma unroll
        for (int i = 0; i < IPT; ++i)
            if ((trmask >> i) & 1u) { S.seg_key[s] = keys[i]; S.seg_start[s] = (uint16_t)(t * IPT + i); ++s; }
    }
    if (t == 0) S.seg_start[nu] = (uint16_t)nvq;
    __syncthreads();
    // grouped coding: deltas key[j + 1] - key[j] - 1 (nu - 1 of them), counts cnt - 1 (nu)
    const int nd = (int)nu - 1, ncnt = (int)nu;
    const int lane = t & 31, wid = t >> 5;
    for (int g = wid; g < 32; g += BT / 32) {
        uint64_t vd = 0;
        uint32_t vc = 0;
        const int j = g * 32 + lane;
        if (j < nd) vd = S.seg_key[j + 1] - S.seg_key[j] - 1u;
        if (j < ncnt) vc = (uint32_t)S.seg_start[j + 1] - S.seg_start[j] - 1u;
        const uint32_t wd = __reduce_max_sync(0xFFFFFFFFu, bitw64(vd)), wc = __reduce_max_sync(0xFFFFFFFFu, bitw(vc));
        if (lane == 0) { S.w32d[g] = (uint8_t)wd; S.w32c[g] = (uint8_t)wc; }
    }
    __syncthreads();
    out.nu = (int)nu; out.Rx = ki.Rx; out.Ry = ki.Ry; out.keybits = end_bit;
    out.mn[0] = ki.mn[0]; out.mn[1] = ki.mn[1]; out.mn[2] = ki.mn[2];
    out.firstkey = S.seg_key[0];
    out.rb = ki.rb;
    out.rbw = 0;
    {
        // warp w evaluates group code w (32/64/128/1024 symbols); lane g owns group g
        const int gc = wid;
        const int G = gsize_of(gc), r = G / 32;
        const int ngd = (nd + G - 1) / G, ngc = (ncnt + G - 1) / G;
        uint32_t bd = 0, bc = 0;
        if (lane < ngd) {
            uint32_t w = 0;
            for (int j = lane * r; j < min((lane + 1) * r, (nd + 31) / 32); ++j) w = max(w, (uint32_t)S.w32d[j]);
            bd = (uint32_t)min(G, nd - lane * G) * w;
        }
        if (lane < ngc) {
            uint32_t w = 0;
            for (int j = lane * r; j < min((lane + 1) * r, (ncnt + 31) / 32); ++j) w = max(w, (uint32_t)S.w32c[j]);
            bc = (uint32_t)min(G, ncnt - lane * G) * w;
        }
        bd = __reduce_add_sync(0xFFFFFFFFu, bd); bc = __reduce_add_sync(0xFFFFFFFFu, bc);
        if (lane == 0) {
            Cand tmp = out; tmp.gcode = gc;
            S.gsel[wid][0] = hdr_bits_of(tmp, n_exc, ngd, ngc); S.gsel[wid][1] = bd; S.gsel[wid][2] = bc;
        }
    }
    __syncthreads();
    int bw_ = 0;
    uint32_t bb = 0xFFFFFFFFu;
    for (int w = 0; w < 4; ++w) { const uint32_t tot = S.gsel[w][0] + S.gsel[w][1] + S.gsel[w][2]; if (tot < bb) { bb = tot; bw_ = w; } }
    out.gcode = bw_;
    out.hdr_bits = S.gsel[bw_][0]; out.delta_bits = S.gsel[bw_][1]; out.count_bits = S.gsel[bw_][2];
    out.res_bits = (uint32_t)nvq * out.rb;
    out.bits = out.hdr_bits + out.delta_bits + out.count_bits + out.res_bits + (uint32_t)n_exc * 96u;
}

// Sort, run-length encode and compute the exact encoded size of a candidate. Leaves sorted keys/vals in registers
// and the segment table + 32-cell widths in smem.
__device__ void eval_sorted(int mode, const int32_t (&q)[IPT][3], uint32_t qvmask, int nvq, int n_exc, Smem& S,
                            uint64_t (&keys)[IPT], uint32_t (&vals)[IPT], Cand& out) {
    const KeyInfo ki = build_keys(mode, q, qvmask, S, keys, vals);
    out = Cand();
    out.mode = mode;
    out.valid = ki.valid;
    if (!ki.valid) return;
    // padding keys masked to [0,end_bit) are all-ones (2^end_bit - 1 >= lin > every valid key)
    const int end_bit = max(1, (int)bitw64(ki.lin));
    const bool konly = ki.rb == 0;   // flat: keys only
    if (end_bit <= 32) {
        uint32_t k32[IPT];
#pragma unroll
        for (int i = 0; i < IPT; ++i) k32[i] = (uint32_t)keys[i];
        if (konly && end_bit > MSORT_MIN) match_sort(k32, end_bit, S);
        else if (konly) SortKT(S.u.sortk).Sort(k32, 0, end_bit);
        else SortT(S.u.sort).Sort(k32, vals, 0, end_bit);
#pragma unroll
        for (int i = 0; i < IPT; ++i) keys[i] = k32[i] & lowmask(end_bit);
    } else if (konly) {
        match_sort(keys, end_bit, S);
    } else {
        SortT64(S.u.sort64).Sort(keys, vals, 0, end_bit);
    }
    eval_post(mode, ki, end_bit, nvq, n_exc, S, keys, out);
}

__device__ __forceinline__ uint32_t f2u(float f) { return __float_as_uint(f); }

// ------------------------------------------------------------------------------
// compression (one CUDA block per 1024-point block)
//   SEL = SEL_COST, SIZE_ONLY = 1   sampling launch (blocks b * c.sample): exhaustive search, sizes + BlockStat only
//   SEL = SEL_COST, SIZE_ONLY = 0   exhaustive search
//   SEL = SEL_LIST, SIZE_ONLY = 0   the sampled shortlist (c.list)
// Output: the block's bytes in its fixed-stride slot, or (dout, c.direct) straight into the compact stream.
// ------------------------------------------------------------------------------
template <int SEL, int SIZE_ONLY>
__global__ __launch_bounds__(BT, XNYGZIP_MINBLOCKS) void k_compress(const float* __restrict__ X, const float* __restrict__ Y,
                                                    const float* __restrict__ Z, int64_t n, float qs, float ds, float eps,
                                                    uint8_t* __restrict__ slots, uint32_t* __restrict__ sizes,
                                                    BlockStat* __restrict__ stats, uint8_t* __restrict__ dout) {
    __shared__ Smem S;
    const Config c = c_cfg;
    const int t = threadIdx.x;
    // direct path: blocks take their index from a launch-order ticket, so every predecessor has started
    const bool direct = !SIZE_ONLY && dout != nullptr && c.direct;
    unsigned bid;
    if (direct) {
        if (t == 0) S.tk = atomicAdd(g_ticket, 1u);
        __syncthreads();
        bid = S.tk;
    } else bid = c.sample ? blockIdx.x * (unsigned)c.sample : blockIdx.x;   // compressed block index
    const int64_t b0 = (int64_t)bid * NB;
    if (b0 >= n) return;
    const int nv = (int)min((int64_t)NB, n - b0);

    float v[IPT][3];
    int32_t q[IPT][3];
    uint32_t vmask = 0, ovmask = 0, exmask = 0, qvmask = 0;
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, sm[3] = {0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        const int l = i * BT + t;
        if (l < nv) {
            v[i][0] = X[b0 + l]; v[i][1] = Y[b0 + l]; v[i][2] = Z[b0 + l];
            vmask |= 1u << i;
            for (int a = 0; a < 3; ++a) { mn[a] = fminf(mn[a], v[i][a]); sm[a] += v[i][a]; }
        } else {
            v[i][0] = v[i][1] = v[i][2] = 0.f;
        }
    }
    block_min3(mn, S.redf3);
    // fine BCC quantization
    const float neg[3] = {-mn[0] * qs, -mn[1] * qs, -mn[2] * qs};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        q[i][0] = q[i][1] = q[i][2] = 0;
        if (!((vmask >> i) & 1u)) continue;
        const float px = __fmaf_rn(v[i][0], qs, neg[0]), py = __fmaf_rn(v[i][1], qs, neg[1]), pz = __fmaf_rn(v[i][2], qs, neg[2]);
        if (!(fabsf(px) < 1073741824.f && fabsf(py) < 1073741824.f && fabsf(pz) < 1073741824.f)) { ovmask |= 1u << i; continue; }
        bcc_nearest_f(px, py, pz, q[i][0], q[i][1], q[i][2]);
        // for |p| >= 2^24 the float arithmetic of the quantizer can return a point that is not on the BCC lattice
        // (ex +- 1 is not representable). Every lattice encoding assumes BCC parity, so such points are stored
        // exactly as exceptions (as the |p| >= 2^30 overflow already is).
        if (((q[i][0] ^ q[i][1]) | (q[i][1] ^ q[i][2])) & 1) { ovmask |= 1u << i; q[i][0] = q[i][1] = q[i][2] = 0; }
    }
    const float eps2_rd = __fmul_rd(eps, eps);
    float og[3] = {mn[0], mn[1], mn[2]};
    // mean shift only when every valid point shares one fine q; the origin is re-anchored so that q*ds + origin ~=
    // mean, and accepted only if every point passes the rigorous bound check against its actual decoded coordinate.
    int32_t vbox[6];       // box of all valid points from the mean-shift reduction (reused below when exact)
    bool vbox_ok;
    {
        int32_t r7[7] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN, INT32_MIN, (int32_t)(ovmask != 0)};
        const bool mx7[7] = {false, false, false, true, true, true, true};
        for (int i = 0; i < IPT; ++i)
            if ((vmask >> i) & 1u)
                for (int a = 0; a < 3; ++a) { r7[a] = min(r7[a], q[i][a]); r7[3 + a] = max(r7[3 + a], q[i][a]); }
        block_reduce<7, false>(r7, mx7, S.red);
#pragma unroll
        for (int a = 0; a < 6; ++a) vbox[a] = r7[a];
        vbox_ok = !r7[6];
        if (!r7[6] && r7[0] == r7[3] && r7[1] == r7[4] && r7[2] == r7[5]) {
            block_sum3(sm, S.redf3);
            const float inv_nv = 1.0f / (float)nv;
            float o2[3];
            for (int a = 0; a < 3; ++a) o2[a] = __fmaf_rn(-(float)r7[a], ds, sm[a] * inv_nv);
            uint32_t fail = 0;
            for (int i = 0; i < IPT; ++i)
                if ((vmask >> i) & 1u) {
                    const float rx = __fmaf_rn((float)q[i][0], ds, o2[0]), ry = __fmaf_rn((float)q[i][1], ds, o2[1]), rz = __fmaf_rn((float)q[i][2], ds, o2[2]);
                    if (!within_bound(v[i][0], v[i][1], v[i][2], rx, ry, rz, eps2_rd)) fail = 1;
                }
            if (!__syncthreads_or((int)fail))
                for (int a = 0; a < 3; ++a) og[a] = o2[a];
        }
    }
    const uint32_t origin[3] = {f2u(og[0]), f2u(og[1]), f2u(og[2])};
    // exceptions: points whose decoded coordinate might violate the bound (float precision)
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (!((vmask >> i) & 1u)) continue;
        bool ok = !((ovmask >> i) & 1u);
        if (ok) {
            const float rx = __fmaf_rn((float)q[i][0], ds, og[0]), ry = __fmaf_rn((float)q[i][1], ds, og[1]), rz = __fmaf_rn((float)q[i][2], ds, og[2]);
            ok = within_bound(v[i][0], v[i][1], v[i][2], rx, ry, rz, eps2_rd);
        }
        if (ok) qvmask |= 1u << i; else exmask |= 1u << i;
    }
    uint32_t rk_base, rk_tot;
    ScanT(S.u.scan).ExclusiveSum((uint32_t)__popc(qvmask) | ((uint32_t)__popc(exmask) << 16), rk_base, rk_tot);
    __syncthreads();
    const int nvq = (int)(rk_tot & 0xFFFF), n_exc = (int)(rk_tot >> 16);
    int32_t qb[6] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN, INT32_MIN};
    const bool mx6[6] = {false, false, false, true, true, true};
    if (vbox_ok && n_exc == 0) {
        // no exceptions and no overflow: the quantized points are all valid points, whose box is known
#pragma unroll
        for (int a = 0; a < 6; ++a) qb[a] = vbox[a];
    } else {
        for (int i = 0; i < IPT; ++i)
            if ((qvmask >> i) & 1u)
                for (int a = 0; a < 3; ++a) { qb[a] = min(qb[a], q[i][a]); qb[3 + a] = max(qb[3 + a], q[i][a]); }
        block_reduce<6>(qb, mx6, S.red);
    }
    if (t == 0) {
#pragma unroll
        for (int a = 0; a < 6; ++a) S.qbox[a] = qb[a];
    }
    __syncthreads();

    // ---------------- analytic candidates: lossless and raw quantized
    // the best candidate lives in shared memory (S.bc); only its size and mode stay in registers
    auto make_loss = [&]() {
        Cand l;
        l.mode = M_LOSSLESS; l.valid = 1;
        l.hdr_bits = hdr_bits_of(l, 0, 0, 0);
        l.bits = l.hdr_bits + (uint32_t)nv * 96u;
        return l;
    };
    uint32_t bbits;
    int bmode;
    auto take = [&](const Cand& x) { bbits = x.bits; bmode = x.mode; if (t == 0) S.bc() = x; };
    const uint32_t loss_bits = make_loss().bits;
    take(make_loss());
    if (nvq > 0) {
        Cand raw;
        raw.mode = M_RAWQ; raw.valid = 1;
        for (int a = 0; a < 3; ++a) raw.mn[a] = qb[a];
        raw.dbw = (int)bitw((uint32_t)(qb[3] - qb[0])); raw.cbw = (int)bitw((uint32_t)(qb[4] - qb[1])); raw.rbw = (int)bitw((uint32_t)(qb[5] - qb[2]));
        raw.hdr_bits = hdr_bits_of(raw, n_exc, 0, 0);
        raw.res_bits = (uint32_t)nvq * (raw.dbw + raw.cbw + raw.rbw);
        raw.bits = raw.hdr_bits + raw.res_bits + (uint32_t)n_exc * 96u;
        take(raw);
    }
    uint64_t keys[IPT];
    uint32_t vals[IPT];
    int last = -1;  // mode whose sorted state is in registers / smem
    // winner cache: the sorted keys (and residuals) of the current best candidate are parked in this block's output
    // slot, so emission does not sort the winner again
    int cmode = -1;   // mode whose sorted state is parked in the slot
    uint8_t* const wslot = slots + (size_t)bid * SLOT_BYTES;
    auto park = [&](const Cand& x) {
        if (x.keybits <= 32) {
            uint32_t k4[IPT];
#pragma unroll
            for (int i = 0; i < IPT; ++i) k4[i] = (uint32_t)keys[i];
            uint4* d = reinterpret_cast<uint4*>(wslot + (size_t)t * IPT * 4);
#pragma unroll
            for (int i = 0; i < IPT / 4; ++i) d[i] = make_uint4(k4[4 * i], k4[4 * i + 1], k4[4 * i + 2], k4[4 * i + 3]);
        } else {
            uint4* d = reinterpret_cast<uint4*>(wslot + (size_t)t * IPT * 8);
#pragma unroll
            for (int i = 0; i < IPT / 2; ++i)
                d[i] = make_uint4((uint32_t)keys[2 * i], (uint32_t)(keys[2 * i] >> 32), (uint32_t)keys[2 * i + 1], (uint32_t)(keys[2 * i + 1] >> 32));
        }
        if (x.rb) {
            uint4* d = reinterpret_cast<uint4*>(wslot + 8192 + (size_t)t * IPT * 4);
#pragma unroll
            for (int i = 0; i < IPT / 4; ++i) d[i] = make_uint4(vals[4 * i], vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]);
        }
        cmode = x.mode;
    };
    // lower bound of every lattice candidate (header + exceptions): if RAWQ is already this small, skip the search
    bool hskip = false;
    if (nvq > 0 && bmode == M_RAWQ) {
        const uint32_t lb = 96u + 4u + 1u + (n_exc ? 10u : 0u) + 28u + 12u + (uint32_t)n_exc * 96u;
        hskip = bbits <= lb;
    }
    if (!hskip && nvq > 0) {
        if (SEL == SEL_COST) {
            // exhaustive: sort + size every mode; prune only with a residual lower bound
            Cand cf;
            eval_sorted(M_FLAT, q, qvmask, nvq, n_exc, S, keys, vals, cf);
            if (cf.valid) { last = M_FLAT; if (cf.bits < bbits) { take(cf); park(cf); } } else last = -1;
            for (int k = 1; k <= KMAX; ++k) {
                const uint32_t lb = 101u + (uint32_t)nvq * (uint32_t)resid_bits(k) + (uint32_t)n_exc * 96u;
                if (lb >= bbits) break;  // residual bits grow with k
                Cand ch;
                eval_sorted(k, q, qvmask, nvq, n_exc, S, keys, vals, ch);
                if (!ch.valid) { last = -1; continue; }  // build_keys clobbered vals
                last = k;
                if (ch.bits < bbits) { take(ch); if (k < KMAX) park(ch); }
            }
        } else {
            const int nl = c.list & 3;
            for (int j = 0; j < nl; ++j) {
                const int m = (c.list >> (2 + 3 * j)) & 7;   // 0 = flat, k = TO level k
                if (m > KMAX) continue;
                Cand ch;
                eval_sorted(m, q, qvmask, nvq, n_exc, S, keys, vals, ch);
                if (!ch.valid) { last = -1; continue; }
                last = m;
                if (ch.bits < bbits) { take(ch); if (j < nl - 1) park(ch); }
            }
            if (nl == 0 || (bmode == M_RAWQ && last < 0)) {
                // every listed candidate overflowed its keys: fall back to flat (or RAWQ if it fails too)
                if (!(nl == 1 && (c.list >> 2 & 7) == 0)) {
                    Cand ch;
                    eval_sorted(M_FLAT, q, qvmask, nvq, n_exc, S, keys, vals, ch);
                    if (ch.valid) { last = M_FLAT; if (ch.bits < bbits) take(ch); }
                }
            }
        }
    }
    if (bbits > (uint32_t)LIMIT_BITS || loss_bits < bbits) take(make_loss());
    // restore the winner's sorted state for emission
    if (is_lattice(bmode) && bmode != last && !SIZE_ONLY) {
        Cand again;
        if (cmode == bmode) {
            const Cand& w = S.bc();
            KeyInfo ki;
            ki.valid = 1; ki.Rx = w.Rx; ki.Ry = w.Ry; ki.lin = 0; ki.rb = w.rb;
            ki.mn[0] = w.mn[0]; ki.mn[1] = w.mn[1]; ki.mn[2] = w.mn[2];
            const int eb = w.keybits;
            if (eb <= 32) {
                const uint4* d = reinterpret_cast<const uint4*>(wslot + (size_t)t * IPT * 4);
#pragma unroll
                for (int i = 0; i < IPT / 4; ++i) { const uint4 v = d[i]; keys[4 * i] = v.x; keys[4 * i + 1] = v.y; keys[4 * i + 2] = v.z; keys[4 * i + 3] = v.w; }
            } else {
                const uint4* d = reinterpret_cast<const uint4*>(wslot + (size_t)t * IPT * 8);
#pragma unroll
                for (int i = 0; i < IPT / 2; ++i) { const uint4 v = d[i]; keys[2 * i] = v.x | (uint64_t)v.y << 32; keys[2 * i + 1] = v.z | (uint64_t)v.w << 32; }
            }
            if (ki.rb) {
                const uint4* d = reinterpret_cast<const uint4*>(wslot + 8192 + (size_t)t * IPT * 4);
#pragma unroll
                for (int i = 0; i < IPT / 4; ++i) { const uint4 v = d[i]; vals[4 * i] = v.x; vals[4 * i + 1] = v.y; vals[4 * i + 2] = v.z; vals[4 * i + 3] = v.w; }
            }
            again.mode = bmode; again.valid = 1;
            eval_post(bmode, ki, eb, nvq, n_exc, S, keys, again);
        } else {
            eval_sorted(bmode, q, qvmask, nvq, n_exc, S, keys, vals, again);
        }
        take(again);
    }

    // ---------------- emission into the shared stage
    __syncthreads();
    const Cand& best = S.bc();
    const uint32_t nbits = bbits;
    const uint32_t nbytes = (nbits + 7u) >> 3;
    const int mode = bmode;
    if (stats && t == 0) {
        BlockStat st;
        st.mode = (uint8_t)mode; st.gcode = (uint8_t)best.gcode; st.nu = (uint16_t)best.nu;
        st.n_exc = (uint16_t)(mode == M_LOSSLESS ? 0 : n_exc); st.nvq = (uint16_t)nvq;
        st.hdr_bits = best.hdr_bits; st.delta_bits = best.delta_bits; st.count_bits = best.count_bits;
        st.res_bits = best.res_bits; st.exc_bits = mode == M_LOSSLESS ? (uint32_t)nv * 96u : (uint32_t)n_exc * 96u;
        st.total_bytes = nbytes;
        stats[bid] = st;
    }
    if (t == 0) sizes[bid] = nbytes;
    if (SIZE_ONLY) return;
    // direct path: publish the own size; the look-back for the own offset runs after the stage is complete
    if (direct && t == 0) atomicExch(&g_lb[bid], (bid == 0 ? LB_INC : LB_AGG) | nbytes);
    const uint32_t nwords = (nbits + 31u) >> 5;
    uint32_t* const st = S.u.stage;
    for (uint32_t w = t; w < nwords + 2; w += BT) st[w] = 0;
    if (t == 0) {
        Hdr& h = S.hdr;
        for (int a = 0; a < 3; ++a) { h.origin[a] = origin[a]; h.mn[a] = best.mn[a]; }
        h.mode = mode; h.n_exc = mode == M_LOSSLESS ? 0 : n_exc; h.nu = best.nu; h.gcode = best.gcode;
        h.Rx = best.Rx; h.Ry = best.Ry; h.firstkey = best.firstkey;
        h.dbw = best.dbw; h.cbw = best.cbw; h.rbw = best.rbw; h.ngd = h.ngc = 0;
    }
    const bool groups = is_lattice(mode);
    if (groups && t < 64) {
        // warp 0: delta groups, warp 1: count groups; lane = group index
        const bool isd = t < 32;
        const int lane = t & 31, G = gsize_of(best.gcode), r = G / 32;
        const int nitems = isd ? best.nu - 1 : best.nu, ng = (nitems + G - 1) / G;
        const uint8_t* w32 = isd ? S.w32d : S.w32c;
        uint32_t w = 0, len = 0;
        if (lane < ng) {
            for (int j = lane * r; j < min((lane + 1) * r, (nitems + 31) / 32); ++j) w = max(w, (uint32_t)w32[j]);
            len = (uint32_t)min(G, nitems - lane * G) * w;
        }
        uint32_t inc = len;
        for (int o = 1; o < 32; o <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, inc, o); if (lane >= o) inc += y; }
        if (lane < ng) {
            if (isd) { S.goff_d[lane] = inc - len; S.hdr.wd[lane] = (uint8_t)w; } else { S.goff_c[lane] = inc - len; S.hdr.wc[lane] = (uint8_t)w; }
        }
    }
    __syncthreads();
    uint32_t hfixed = 0;
    if (t == 0) {
        SmemSink snk{st, 0};
        header_emit(snk, S.hdr);   // fixed part (group widths below)
        hfixed = snk.pos;
        S.gsel[0][0] = hfixed;
    }
    __syncthreads();
    hfixed = S.gsel[0][0];
    if (groups) {
        const int G = gsize_of(best.gcode), ngd = (best.nu - 1 + G - 1) / G, ngc = (best.nu + G - 1) / G;
        if (t < ngd) put_bits(st, hfixed + 6u * t, S.hdr.wd[t], 6);
        if (t < ngc) put_bits(st, hfixed + 6u * ngd + 4u * t, S.hdr.wc[t], 4);
        if (t == 0 && hfixed + 6u * ngd + 4u * ngc != best.hdr_bits) atomicAdd(&g_err, 1u);
        if (t == 0) { S.hdr.ngd = ngd; S.hdr.ngc = ngc; }
    } else if (t == 0 && hfixed != best.hdr_bits) atomicAdd(&g_err, 1u);
    __syncthreads();
    const Hdr& h = S.hdr;
    if (mode == M_LOSSLESS) {
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const int l = i * BT + t;
            if (l < nv) {
                put_bits(st, best.hdr_bits + (uint32_t)l * 96u, f2u(X[b0 + l]), 32);
                put_bits(st, best.hdr_bits + (uint32_t)l * 96u + 32u, f2u(Y[b0 + l]), 32);
                put_bits(st, best.hdr_bits + (uint32_t)l * 96u + 64u, f2u(Z[b0 + l]), 32);
            }
        }
    } else {
        uint32_t epos;
        if (mode == M_RAWQ) {
            const uint32_t W = (uint32_t)(h.dbw + h.cbw + h.rbw);
            uint32_t r = rk_base & 0xFFFF;
#pragma unroll
            for (int i = 0; i < IPT; ++i)
                if ((qvmask >> i) & 1u) {
                    const uint32_t pos = best.hdr_bits + r * W;
                    put_bits(st, pos, (uint32_t)(q[i][0] - h.mn[0]), h.dbw);
                    put_bits(st, pos + h.dbw, (uint32_t)(q[i][1] - h.mn[1]), h.cbw);
                    put_bits(st, pos + h.dbw + h.cbw, (uint32_t)(q[i][2] - h.mn[2]), h.rbw);
                    ++r;
                }
            epos = best.hdr_bits + best.res_bits;
        } else {
            const int nd = h.nu - 1;
            const uint32_t dpos = best.hdr_bits, cpos = dpos + best.delta_bits;
            const int G = gsize_of(h.gcode);
            const int gsh = __ffs(G) - 1;
            for (int j = t; j < nd; j += BT) {
                const uint64_t d = S.seg_key[j + 1] - S.seg_key[j] - 1u;
                const int g = j >> gsh;
                put_bits64(st, dpos + S.goff_d[g] + (uint32_t)(j & (G - 1)) * h.wd[g], d, h.wd[g]);
            }
            for (int j = t; j < h.nu; j += BT) {
                const uint32_t cnt = (uint32_t)S.seg_start[j + 1] - S.seg_start[j];
                const int g = j >> gsh;
                put_bits(st, cpos + S.goff_c[g] + (uint32_t)(j & (G - 1)) * h.wc[g], cnt - 1u, h.wc[g]);
            }
            const uint32_t rpos = cpos + best.count_bits;
            if (best.rb) {
#pragma unroll
                for (int i = 0; i < IPT; ++i) {
                    const int p = t * IPT + i;
                    if (p < nvq) put_bits(st, rpos + (uint32_t)p * best.rb, vals[i], best.rb);
                }
            }
            epos = rpos + best.res_bits;
        }
        uint32_t e = rk_base >> 16;
#pragma unroll
        for (int i = 0; i < IPT; ++i)
            if ((exmask >> i) & 1u) {
                const int l = i * BT + t;
                put_bits(st, epos + e * 96u, f2u(X[b0 + l]), 32);
                put_bits(st, epos + e * 96u + 32u, f2u(Y[b0 + l]), 32);
                put_bits(st, epos + e * 96u + 64u, f2u(Z[b0 + l]), 32);
                ++e;
            }
    }
    if (direct && t < 32) {
        // warp 0: sum the predecessors' published sizes 32 at a time back to the nearest inclusive prefix
        unsigned long long excl = 0;
        if (bid > 0) {
            for (long long j = (long long)bid - 1 - t;; j -= 32) {
                unsigned long long s = LB_INC;   // before block 0: inclusive prefix 0
                if (j >= 0) do { s = *(volatile unsigned long long*)&g_lb[j]; } while (!(s >> 62));
                const unsigned inc = __ballot_sync(0xFFFFFFFFu, (s >> 62) == 2u);
                const int first = inc ? __ffs(inc) - 1 : 32;   // nearest predecessor with an inclusive prefix
                unsigned long long v = t <= first ? (s & LB_VAL) : 0ull;
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
                excl += v;
                if (inc) break;
            }
            if (t == 0) atomicExch(&g_lb[bid], LB_INC | (excl + nbytes));
        }
        if (t == 0) S.lb_off = excl;
    }
    __syncthreads();
    if (direct) {
        // copy the stage to dout + ob: 32-bit stores, byte stores for the (<= 2) words shared with the neighbours
        const unsigned long long ob = S.lb_off;
        uint8_t* d = dout + ob;
        const uint32_t head = (uint32_t)((4u - (ob & 3u)) & 3u), hb = min(head, nbytes);
        const uint32_t nw = (nbytes - hb) >> 2, tail0 = hb + 4u * nw, sh = (head & 3u) * 8u;
        auto sbyte = [&](uint32_t k) { return (uint8_t)(st[k >> 2] >> (8u * (k & 3u))); };
        if (t < (int)hb) d[t] = sbyte(t);
        uint32_t* dw = reinterpret_cast<uint32_t*>(d + hb);
        for (uint32_t w = t; w < nw; w += BT) {
            const uint32_t k = hb + 4u * w;
            dw[w] = sh ? __funnelshift_r(st[k >> 2], st[(k >> 2) + 1], sh) : st[k >> 2];
        }
        if (t < (int)(nbytes - tail0)) d[tail0 + t] = sbyte(tail0 + t);
        return;
    }
    uint32_t* dst = reinterpret_cast<uint32_t*>(slots + (size_t)bid * SLOT_BYTES);
    for (uint32_t w = t; w < nwords; w += BT) dst[w] = st[w];
}

// ------------------------------------------------------------------------------
// compaction: fixed-stride slots -> contiguous byte stream (skipped if the main launch wrote it directly)
// ------------------------------------------------------------------------------
__global__ void k_gather(const uint8_t* __restrict__ slots, const uint64_t* __restrict__ offs,
                         const uint32_t* __restrict__ sizes, uint8_t* __restrict__ out, const Config* dc) {
    if (dc && dc->direct) return;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(slots + (size_t)blockIdx.x * SLOT_BYTES);
    uint8_t* d = out + offs[blockIdx.x];
    const uint32_t nb = sizes[blockIdx.x], nw = (nb + 3) >> 2;
    if ((offs[blockIdx.x] & 3) == 0) {
        uint32_t* dw = reinterpret_cast<uint32_t*>(d);
        for (uint32_t w = threadIdx.x; w < nb / 4; w += blockDim.x) dw[w] = src[w];
        for (uint32_t b = (nb / 4) * 4 + threadIdx.x; b < nb; b += blockDim.x) d[b] = (uint8_t)(src[b >> 2] >> (8 * (b & 3)));
    } else {
        for (uint32_t w = threadIdx.x; w < nw; w += blockDim.x) {
            const uint32_t x = src[w];
            for (int b = 0; b < 4; ++b)
                if (4 * w + b < nb) d[4 * w + b] = (uint8_t)(x >> (8 * b));
        }
    }
}

// ------------------------------------------------------------------------------
// decompression (one CUDA block per compressed block). Points come out block by block: lattice points in cell order,
// then the exceptions. kind (optional): 0 lattice point, 1 exception (exact), 2 lossless block; qout (optional): the
// fine lattice point of every lattice point.
// ------------------------------------------------------------------------------
__global__ __launch_bounds__(BT) void k_decompress(const uint32_t* __restrict__ comp, const uint64_t* __restrict__ offs,
                                                   int64_t n, float ds, float* __restrict__ X, float* __restrict__ Y,
                                                   float* __restrict__ Z, uint8_t* __restrict__ kind, int32_t* __restrict__ qout) {
    __shared__ Hdr H;
    __shared__ uint64_t s_key[NB];
    __shared__ union { uint32_t start[NB + 1]; uint16_t cell[NB]; } s_su;
    uint32_t* const s_start = s_su.start;
    uint16_t* const s_cell = s_su.cell;
    __shared__ uint32_t goff_d[33], goff_c[33];
    __shared__ uint32_t s_pos[4];   // header bits, delta bits, count bits, start of the group widths
    __shared__ union { typename ScanT::TempStorage s32; typename ScanT64::TempStorage s64; } scan;
    __shared__ uint32_t s_head[NB / 32];
    const int t = threadIdx.x;
    const int64_t b0 = (int64_t)blockIdx.x * NB;
    const int nv = (int)min((int64_t)NB, n - b0);
    const uint64_t base = offs[blockIdx.x] * 8ull;
    // thread 0 parses the fixed header; warps 0 / 1 read the delta / count group widths in parallel
    if (t == 0) {
        GlobReader r{comp, base};
        header_parse(r, H);
        s_pos[3] = (uint32_t)(r.pos - base);
        s_pos[0] = s_pos[3];
    }
    __syncthreads();
    if (is_lattice(H.mode) && t < 64) {
        const bool isd = t < 32; const int lane = t & 31, G = gsize_of(H.gcode);
        const int ng = isd ? H.ngd : H.ngc, items = isd ? H.nu - 1 : H.nu;
        uint32_t w = 0, len = 0;
        if (lane < ng) {
            w = isd ? get_bits(comp, base + s_pos[3] + 6u * lane, 6) : get_bits(comp, base + s_pos[3] + 6u * H.ngd + 4u * lane, 4);
            len = (uint32_t)min(G, items - lane * G) * w;
        }
        uint32_t inc = len;
        for (int o = 1; o < 32; o <<= 1) { const uint32_t y = __shfl_up_sync(0xFFFFFFFFu, inc, o); if (lane >= o) inc += y; }
        if (lane < ng) { if (isd) { H.wd[lane] = (uint8_t)w; goff_d[lane] = inc - len; } else { H.wc[lane] = (uint8_t)w; goff_c[lane] = inc - len; } }
        const uint32_t tot = __shfl_sync(0xFFFFFFFFu, inc, 31);
        if (lane == 0) { if (isd) s_pos[1] = tot; else { s_pos[2] = tot; s_pos[0] = s_pos[3] + 6u * H.ngd + 4u * H.ngc; } }
    }
    __syncthreads();
    const int mode = H.mode;
    const uint32_t hb = s_pos[0];
    const float og[3] = {__uint_as_float(H.origin[0]), __uint_as_float(H.origin[1]), __uint_as_float(H.origin[2])};
    if (mode == M_LOSSLESS) {
        for (int l = t; l < nv; l += BT) {
            const uint64_t p = base + hb + (uint64_t)l * 96u;
            X[b0 + l] = __uint_as_float(get_bits(comp, p, 32));
            Y[b0 + l] = __uint_as_float(get_bits(comp, p + 32, 32));
            Z[b0 + l] = __uint_as_float(get_bits(comp, p + 64, 32));
            if (kind) kind[b0 + l] = 2;
        }
        return;
    }
    const int n_exc = H.n_exc, nvq = nv - n_exc;
    uint32_t epos;
    if (mode == M_RAWQ) {
        const uint32_t W = (uint32_t)(H.dbw + H.cbw + H.rbw);
        for (int p = t; p < nvq; p += BT) {
            const uint64_t pos = base + hb + (uint64_t)p * W;
            int32_t qq[3] = {H.mn[0] + (int32_t)get_bits(comp, pos, H.dbw), H.mn[1] + (int32_t)get_bits(comp, pos + H.dbw, H.cbw),
                             H.mn[2] + (int32_t)get_bits(comp, pos + H.dbw + H.cbw, H.rbw)};
            X[b0 + p] = __fmaf_rn((float)qq[0], ds, og[0]); Y[b0 + p] = __fmaf_rn((float)qq[1], ds, og[1]); Z[b0 + p] = __fmaf_rn((float)qq[2], ds, og[2]);
            if (kind) kind[b0 + p] = 0;
            if (qout) { qout[3 * (b0 + p)] = qq[0]; qout[3 * (b0 + p) + 1] = qq[1]; qout[3 * (b0 + p) + 2] = qq[2]; }
        }
        epos = hb + (uint32_t)nvq * W;
    } else {
        const int k = mode;
        const int nu = H.nu, G = gsize_of(H.gcode), gsh = __ffs(G) - 1;
        const uint32_t dpos = hb, cpos = hb + s_pos[1];
        uint64_t a[IPT];
        uint32_t cn[IPT];
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const int j = t * IPT + i;
            a[i] = cn[i] = 0;
            if (j < nu) {
                if (j == 0) a[i] = H.firstkey;
                else { const int jj = j - 1, g = jj >> gsh; a[i] = get_bits64(comp, base + dpos + goff_d[g] + (uint64_t)(jj & (G - 1)) * H.wd[g], H.wd[g]) + 1u; }
                const int g = j >> gsh;
                cn[i] = get_bits(comp, base + cpos + goff_c[g] + (uint64_t)(j & (G - 1)) * H.wc[g], H.wc[g]) + 1u;
            }
        }
        ScanT64(scan.s64).InclusiveSum(a, a);
        __syncthreads();
        uint32_t tot;
        ScanT(scan.s32).ExclusiveSum(cn, cn, tot);
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
            const int j = t * IPT + i;
            if (j < nu) { s_key[j] = a[i]; s_start[j] = cn[i]; }
        }
        if (t == 0) s_start[nu] = tot;
        // point -> cell: a binary search over the cell starts, or (more than 64 cells) the scanned head flags
        const bool useidx = nu > 64;
        if (useidx) {   // head flag at the first point of every cell; cell of point p = (# heads <= p) - 1
            for (int w = t; w < NB / 32; w += BT) s_head[w] = 0;
            __syncthreads();
#pragma unroll
            for (int i = 0; i < IPT; ++i) { const int j = t * IPT + i; if (j < nu && cn[i] < (uint32_t)NB) atomicOr(&s_head[cn[i] >> 5], 1u << (cn[i] & 31)); }
            __syncthreads();
            const uint32_t hw = s_head[(t * IPT) >> 5] >> ((t * IPT) & 31);
            uint32_t hc = __popc(hw & 0xFFu), hbase, htot;
            ScanT(scan.s32).ExclusiveSum(hc, hbase, htot);
            uint32_t run = hbase;
#pragma unroll
            for (int i = 0; i < IPT; ++i) { run += (hw >> i) & 1u; s_cell[t * IPT + i] = (uint16_t)(run ? run - 1u : 0u); }
        }
        __syncthreads();
        const uint32_t rpos = cpos + s_pos[2];
        const uint32_t rb = (uint32_t)resid_bits(mode);
        const uint64_t Rx = H.Rx, Rxy = (uint64_t)H.Rx * H.Ry;
        // 32-bit key -> cell division when every key of the block fits 32 bits
        const bool k32 = s_key[nu - 1] <= 0xFFFFFFFFull && Rxy <= 0xFFFFFFFFull;
        for (int p = t; p < nvq; p += BT) {
            int lo = 0, hi = nu - 1;
            if (useidx) lo = s_cell[p];
            else while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_start[mid] <= (uint32_t)p) lo = mid; else hi = mid - 1; }
            const uint64_t key = s_key[lo];
            int32_t cc[3];
            if (k32) {
                const uint32_t kk = (uint32_t)key, rx = (uint32_t)Rx, kq = kk / rx;
                cc[0] = H.mn[0] + (int32_t)(kk - kq * rx); cc[1] = H.mn[1] + (int32_t)(kq % (uint32_t)H.Ry); cc[2] = H.mn[2] + (int32_t)(kk / (uint32_t)Rxy);
            } else {
                cc[0] = H.mn[0] + (int32_t)(key % Rx); cc[1] = H.mn[1] + (int32_t)((key / Rx) % H.Ry); cc[2] = H.mn[2] + (int32_t)(key / Rxy);
            }
            {   // parity linearization
                const int32_t pz = cc[2] & 1; cc[0] = 2 * cc[0] + pz; cc[1] = 2 * cc[1] + pz;
            }
            int32_t qq[3];
            if (mode == M_FLAT) { qq[0] = cc[0]; qq[1] = cc[1]; qq[2] = cc[2]; }
            else to_decode(cc[0], cc[1], cc[2], get_bits(comp, base + rpos + (uint64_t)p * rb, rb), k, qq[0], qq[1], qq[2]);
            X[b0 + p] = __fmaf_rn((float)qq[0], ds, og[0]); Y[b0 + p] = __fmaf_rn((float)qq[1], ds, og[1]); Z[b0 + p] = __fmaf_rn((float)qq[2], ds, og[2]);
            if (kind) kind[b0 + p] = 0;
            if (qout) { qout[3 * (b0 + p)] = qq[0]; qout[3 * (b0 + p) + 1] = qq[1]; qout[3 * (b0 + p) + 2] = qq[2]; }
        }
        epos = rpos + (uint32_t)nvq * rb;
    }
    for (int e = t; e < n_exc; e += BT) {
        const uint64_t p = base + epos + (uint64_t)e * 96u;
        const int64_t o = b0 + nvq + e;
        X[o] = __uint_as_float(get_bits(comp, p, 32)); Y[o] = __uint_as_float(get_bits(comp, p + 32, 32)); Z[o] = __uint_as_float(get_bits(comp, p + 64, 32));
        if (kind) kind[o] = 1;
    }
}

// ==============================================================================
// Host API
// ==============================================================================

struct FileHeader {               // 56 bytes ("XNG2")
    char magic[4];                // "XNG2"
    uint32_t version;             // 2
    uint64_t n;                   // points
    uint32_t grid;                // blocks
    float eps, qs, ds;            // user L2 bound, quant / dequant scale
    uint32_t cfg;                 // Config::pack(): format identifier + search settings
    uint32_t size_entry_bytes;    // 2: one uint16 byte count per block
    uint64_t payload_bytes;
    uint32_t cfg_hi;              // shortlist threshold fth (informational)
    uint32_t reserved;            // 0
};
static_assert(sizeof(FileHeader) == 56, "header layout");

inline float quant_scale_of(float eps) { return std::sqrt(5.0f) / (2.0f * eps); }
inline float dequant_scale_of(float eps) { return (2.0f * eps) / std::sqrt(5.0f); }

struct Compressed {
    FileHeader fh{};
    std::vector<uint32_t> sizes;   // per block bytes
    std::vector<uint8_t> payload;  // concatenated blocks
    std::vector<BlockStat> stats;  // optional
    size_t dev_peak_bytes = 0;     // compress(): peak requested device bytes
    uint64_t file_bytes() const { return sizeof(FileHeader) + (uint64_t)fh.grid * 2 + payload.size(); }
};

// mode histogram of the sampled blocks (exhaustive search) -> shortlist: modes (flat or level) that won >= fth % of
// the sampled blocks, at most `adapt` of them, ordered least frequent first (the most frequent is evaluated last, so
// its sorted state is reused for emission); at least one lattice candidate (the most frequent, flat if none won).
// Also picks the output path: direct (single pass) if the sampled blocks average more than DIRECT_BPP bits per point.
__global__ void k_pickmask(const BlockStat* st, int nb, int stride, Config base, Config* out) {
    __shared__ int scnt[KMAX + 2];
    __shared__ unsigned long long sbytes; __shared__ unsigned int sblk;
    if (threadIdx.x < KMAX + 2) scnt[threadIdx.x] = 0;
    if (threadIdx.x == 0) { sbytes = 0; sblk = 0; }
    __syncthreads();
    {
        int lc[KMAX + 1] = {0}, lns = 0;
        unsigned long long lb_ = 0; unsigned int lk = 0;
        for (int b = threadIdx.x * stride; b < nb; b += blockDim.x * stride) {
            lb_ += st[b].total_bytes; ++lk;
            const int m = st[b].mode;
            if (is_lattice(m)) lc[m]++;
            if (st[b].nvq) ++lns;
        }
        for (int k = 0; k <= KMAX; ++k) if (lc[k]) atomicAdd(&scnt[k], lc[k]);
        if (lns) atomicAdd(&scnt[KMAX + 1], lns);
        if (lk) { atomicAdd(&sbytes, lb_); atomicAdd(&sblk, lk); }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    base.direct = sblk && 8.0 * (double)sbytes > (double)DIRECT_BPP * NB * (double)sblk;
    const int ns = scnt[KMAX + 1];
    int sel[3], nsel = 0;
    uint32_t used = 0;
    for (int r = 0; r < base.adapt && r < 3; ++r) {
        int kb = -1;
        for (int k = 0; k <= KMAX; ++k) if (!((used >> k) & 1u) && (kb < 0 || scnt[k] > scnt[kb])) kb = k;
        if (kb < 0 || scnt[kb] == 0) break;
        if (r > 0 && scnt[kb] * 100 < base.fth * ns) break;
        used |= 1u << kb; sel[nsel++] = kb;
    }
    if (nsel == 0) sel[nsel++] = 0;
    int lst = nsel;
    for (int j = 0; j < nsel; ++j) lst |= sel[nsel - 1 - j] << (2 + 3 * j);   // reversed: least frequent first
    base.list = lst; base.adapt = 0; base.sample = 0;
    *out = base;
}

__global__ void k_widen(const uint32_t* s, uint64_t* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = s[i];
    if (i == n) o[i] = 0;
}

// device buffers of one chunk (allocated once, reused by every call)
struct DevBufs {
    int64_t cap_blocks = 0;
    float *x = nullptr, *y = nullptr, *z = nullptr;
    uint8_t *slots = nullptr, *comp = nullptr;
    size_t comp_cap = 0;
    uint32_t* sizes = nullptr;
    uint64_t *sz64 = nullptr, *offs = nullptr;
    BlockStat* stats = nullptr;
    void* tmp = nullptr; size_t tmp_bytes = 0;
    uint8_t* kind = nullptr; int32_t* q = nullptr;
    Config* dcfg = nullptr;
    unsigned long long* lb = nullptr; unsigned* ticket = nullptr;   // direct path
    size_t bytes = 0, peak_bytes = 0;   // requested device bytes (current / peak), allocator rounding excluded
    template <class T> void get(T*& p, size_t n) { XNYG_CHECK(cudaMalloc(&p, n)); bytes += n; peak_bytes = std::max(peak_bytes, bytes); }
    // compression: input, worst-case slots (also the compact stream's capacity), look-back state, size table, stats
    void alloc_compress(int64_t blocks, bool want_input = true) {
        cap_blocks = blocks;
        const size_t np = (size_t)blocks * NB;
        if (want_input) { get(x, np * 4); get(y, np * 4); get(z, np * 4); }
        get(slots, (size_t)blocks * SLOT_BYTES); comp_cap = (size_t)blocks * SLOT_BYTES;
        get(comp, comp_cap + 16);
        get(lb, (size_t)blocks * 8); get(ticket, 4);
        get(sizes, blocks * 4); get(sz64, (blocks + 1) * 8); get(offs, (blocks + 1) * 8);
        get(stats, blocks * sizeof(BlockStat));
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, sz64, offs, (int)(blocks + 1));
        get(tmp, tmp_bytes);
        get(dcfg, sizeof(Config));
    }
    // decompression: output coordinates, the compact stream (comp_bytes), block offsets [, kind / q]
    void alloc_decompress(int64_t blocks, size_t comp_bytes, bool want_kq) {
        cap_blocks = blocks;
        const size_t np = (size_t)blocks * NB;
        get(x, np * 4); get(y, np * 4); get(z, np * 4);
        comp_cap = comp_bytes;
        get(comp, comp_cap + 16);
        get(offs, (blocks + 1) * 8);
        if (want_kq) { get(kind, np); get(q, np * 12); }
    }
    void free_all() {
        cudaFree(x); cudaFree(y); cudaFree(z); cudaFree(slots); cudaFree(comp); cudaFree(sizes); cudaFree(sz64); cudaFree(offs);
        cudaFree(stats); cudaFree(tmp); cudaFree(dcfg); cudaFree(kind); cudaFree(q); cudaFree(lb); cudaFree(ticket);
        *this = DevBufs();
    }
};

constexpr int SAMPLE_CAP = 2048;   // at most this many sampled blocks per chunk (the stride grows beyond 64)
// resident CUDA blocks of the compression kernel on the whole device (one "wave")
inline int compress_wave_blocks() {
    static int w = 0;
    if (!w) {
        int dev, sms, per;
        XNYG_CHECK(cudaGetDevice(&dev));
        XNYG_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        XNYG_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_compress<SEL_LIST, 0>, BT, 0));
        w = sms * per;
    }
    return w;
}

inline unsigned int read_and_clear_errors() {
    unsigned int e = 0, z = 0;
    XNYG_CHECK(cudaMemcpyFromSymbol(&e, g_err, sizeof(e)));
    XNYG_CHECK(cudaMemcpyToSymbol(g_err, &z, sizeof(z)));
    return e;
}
// launch the compression kernels for nb blocks: exhaustive search if requested, or if the whole input fits in one
// resident wave of k_compress (it then costs one block's latency, as a sampling pass alone would). Otherwise a
// sampling pass (stride min(64, nb / 64), i.e. >= 64 sampled blocks, and at most SAMPLE_CAP of them) picks the
// shortlist on the device (no host synchronization), then the main launch.
inline void launch_compress(DevBufs& B, int64_t npts, int nb, float eps, const Config& cfg, cudaStream_t st,
                            bool want_stats, uint8_t* direct_out) {
    const float qs = quant_scale_of(eps), ds = dequant_scale_of(eps);
    int stride = std::max(1, std::min(64, nb / 64));
    stride = std::max(stride, (nb + SAMPLE_CAP - 1) / SAMPLE_CAP);
    BlockStat* const stats = want_stats ? B.stats : nullptr;
    if (cfg.select == SEL_COST || nb <= compress_wave_blocks()) {
        Config ex = cfg;
        ex.select = SEL_COST; ex.adapt = 0;
        XNYG_CHECK(cudaMemcpyToSymbolAsync(c_cfg, &ex, sizeof(Config), 0, cudaMemcpyHostToDevice, st));
        XNYG_CHECK(cudaMemcpyAsync(B.dcfg, &ex, sizeof(Config), cudaMemcpyHostToDevice, st));
        k_compress<SEL_COST, 0><<<nb, BT, 0, st>>>(B.x, B.y, B.z, npts, qs, ds, eps, B.slots, B.sizes, stats, direct_out);
        return;
    }
    Config sc = cfg;
    sc.select = SEL_COST; sc.adapt = 0; sc.sample = stride;
    XNYG_CHECK(cudaMemcpyToSymbolAsync(c_cfg, &sc, sizeof(Config), 0, cudaMemcpyHostToDevice, st));
    k_compress<SEL_COST, 1><<<(nb + stride - 1) / stride, BT, 0, st>>>(B.x, B.y, B.z, npts, qs, ds, eps, B.slots, B.sizes, B.stats, nullptr);
    k_pickmask<<<1, 256, 0, st>>>(B.stats, nb, stride, cfg, B.dcfg);
    XNYG_CHECK(cudaMemcpyToSymbolAsync(c_cfg, B.dcfg, sizeof(Config), 0, cudaMemcpyDeviceToDevice, st));
    k_compress<SEL_LIST, 0><<<nb, BT, 0, st>>>(B.x, B.y, B.z, npts, qs, ds, eps, B.slots, B.sizes, stats, direct_out);
}
// device-resident compression of nb blocks already in B.x/y/z; returns the compact size, the stream in B.comp and
// the block offsets in B.offs
inline uint64_t compress_device(DevBufs& B, int64_t npts, int nb, float eps, const Config& cfg, cudaStream_t st = 0, bool want_stats = false) {
    // large inputs: the main launch writes the compact stream itself if the sampling pass chose the direct path
    // (Config::direct); otherwise slots + scan + gather. The scan runs in both cases (it also yields the total size).
    const bool autop = nb >= AUTO_MINBLOCKS;
    if (autop) {
        XNYG_CHECK(cudaMemsetAsync(B.lb, 0, (size_t)nb * 8, st));
        XNYG_CHECK(cudaMemsetAsync(B.ticket, 0, 4, st));
        XNYG_CHECK(cudaMemcpyToSymbolAsync(g_lb, &B.lb, sizeof(B.lb), 0, cudaMemcpyHostToDevice, st));
        XNYG_CHECK(cudaMemcpyToSymbolAsync(g_ticket, &B.ticket, sizeof(B.ticket), 0, cudaMemcpyHostToDevice, st));
    }
    launch_compress(B, npts, nb, eps, cfg, st, want_stats, autop ? B.comp : nullptr);
    k_widen<<<(nb + 1 + 255) / 256, 256, 0, st>>>(B.sizes, B.sz64, nb);
    size_t tb = B.tmp_bytes;
    cub::DeviceScan::ExclusiveSum(B.tmp, tb, B.sz64, B.offs, nb + 1, st);
    k_gather<<<nb, 128, 0, st>>>(B.slots, B.offs, B.sizes, B.comp, autop ? B.dcfg : nullptr);
    uint64_t total = 0;
    XNYG_CHECK(cudaMemcpyAsync(&total, B.offs + nb, 8, cudaMemcpyDeviceToHost, st));
    XNYG_CHECK(cudaStreamSynchronize(st));
    return total;
}
// device-resident decompression of nb blocks (stream in B.comp, block offsets in B.offs) into B.x/y/z [, kind, q]
inline void decompress_device(DevBufs& B, int64_t npts, int nb, float eps, cudaStream_t st = 0) {
    k_decompress<<<nb, BT, 0, st>>>(reinterpret_cast<const uint32_t*>(B.comp), B.offs, npts, dequant_scale_of(eps), B.x, B.y, B.z, B.kind, B.q);
}

// host-to-host compression in chunks of chunk_blocks blocks (each chunk is sampled on its own)
inline Compressed compress(const float* hx, const float* hy, const float* hz, int64_t n, float eps, const Config& cfg,
                           bool want_stats = false, int64_t chunk_blocks = 32768) {
    Compressed C;
    const int64_t grid = (n + NB - 1) / NB;
    memcpy(C.fh.magic, "XNG2", 4);
    C.fh.version = 2; C.fh.cfg_hi = (uint32_t)cfg.fth; C.fh.reserved = 0; C.fh.n = (uint64_t)n; C.fh.grid = (uint32_t)grid; C.fh.eps = eps;
    C.fh.qs = quant_scale_of(eps); C.fh.ds = dequant_scale_of(eps); C.fh.cfg = cfg.pack();
    C.fh.size_entry_bytes = 2;
    C.sizes.resize(grid);
    if (want_stats) C.stats.resize(grid);
    DevBufs B;
    B.alloc_compress(std::min(chunk_blocks, std::max<int64_t>(grid, 1)));
    for (int64_t blk0 = 0; blk0 < grid; blk0 += B.cap_blocks) {
        const int nb = (int)std::min(B.cap_blocks, grid - blk0);
        const int64_t p0 = blk0 * NB, np = std::min<int64_t>((int64_t)nb * NB, n - p0);
        XNYG_CHECK(cudaMemcpy(B.x, hx + p0, np * 4, cudaMemcpyHostToDevice));
        XNYG_CHECK(cudaMemcpy(B.y, hy + p0, np * 4, cudaMemcpyHostToDevice));
        XNYG_CHECK(cudaMemcpy(B.z, hz + p0, np * 4, cudaMemcpyHostToDevice));
        const uint64_t total = compress_device(B, np, nb, eps, cfg, 0, want_stats);
        XNYG_CHECK(cudaGetLastError());
        const size_t old = C.payload.size();
        C.payload.resize(old + total);
        XNYG_CHECK(cudaMemcpy(C.payload.data() + old, B.comp, total, cudaMemcpyDeviceToHost));
        XNYG_CHECK(cudaMemcpy(C.sizes.data() + blk0, B.sizes, nb * 4, cudaMemcpyDeviceToHost));
        if (want_stats) XNYG_CHECK(cudaMemcpy(C.stats.data() + blk0, B.stats, nb * sizeof(BlockStat), cudaMemcpyDeviceToHost));
    }
    C.dev_peak_bytes = B.peak_bytes;
    B.free_all();
    if (const unsigned int e = read_and_clear_errors()) { fprintf(stderr, "xnygzip: %u header size-model mismatches\n", e); std::exit(3); }
    C.fh.payload_bytes = C.payload.size();
    return C;
}

// host-to-host decompression; kind / q optional (verification)
inline void decompress(const Compressed& C, float* ox, float* oy, float* oz, uint8_t* okind = nullptr, int32_t* oq = nullptr,
                       int64_t chunk_blocks = 32768) {
    const int64_t n = (int64_t)C.fh.n, grid = C.fh.grid;
    std::vector<uint64_t> off(grid + 1, 0);
    for (int64_t b = 0; b < grid; ++b) off[b + 1] = off[b] + C.sizes[b];
    const int64_t cap = std::min(chunk_blocks, std::max<int64_t>(grid, 1));
    uint64_t maxchunk = 0;
    for (int64_t blk0 = 0; blk0 < grid; blk0 += cap) maxchunk = std::max(maxchunk, off[std::min(grid, blk0 + cap)] - off[blk0]);
    DevBufs B;
    B.alloc_decompress(cap, std::max<uint64_t>(maxchunk, 4), okind || oq);
    std::vector<uint64_t> loc(cap + 1);
    for (int64_t blk0 = 0; blk0 < grid; blk0 += cap) {
        const int nb = (int)std::min(cap, grid - blk0);
        const int64_t p0 = blk0 * NB, np = std::min<int64_t>((int64_t)nb * NB, n - p0);
        const uint64_t c0 = off[blk0], c1 = off[blk0 + nb];
        for (int b = 0; b <= nb; ++b) loc[b] = off[blk0 + b] - c0;
        XNYG_CHECK(cudaMemcpy(B.comp, C.payload.data() + c0, c1 - c0, cudaMemcpyHostToDevice));
        XNYG_CHECK(cudaMemset(B.comp + (c1 - c0), 0, 16));
        XNYG_CHECK(cudaMemcpy(B.offs, loc.data(), (nb + 1) * 8, cudaMemcpyHostToDevice));
        if (B.q) XNYG_CHECK(cudaMemset(B.q, 0, np * 12));
        decompress_device(B, np, nb, C.fh.eps);
        XNYG_CHECK(cudaGetLastError());
        XNYG_CHECK(cudaMemcpy(ox + p0, B.x, np * 4, cudaMemcpyDeviceToHost));
        XNYG_CHECK(cudaMemcpy(oy + p0, B.y, np * 4, cudaMemcpyDeviceToHost));
        XNYG_CHECK(cudaMemcpy(oz + p0, B.z, np * 4, cudaMemcpyDeviceToHost));
        if (okind) XNYG_CHECK(cudaMemcpy(okind + p0, B.kind, np, cudaMemcpyDeviceToHost));
        if (oq) XNYG_CHECK(cudaMemcpy(oq + 3 * p0, B.q, np * 12, cudaMemcpyDeviceToHost));
    }
    B.free_all();
}

inline bool save(const std::string& path, const Compressed& C) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char*)&C.fh, sizeof(FileHeader));
    for (uint32_t s : C.sizes) { const uint16_t v = (uint16_t)s; f.write((const char*)&v, 2); }
    f.write((const char*)C.payload.data(), C.payload.size());
    return (bool)f;
}
inline bool load(const std::string& path, Compressed& C) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    C.fh = FileHeader{};
    f.read((char*)&C.fh, sizeof(FileHeader));
    if (!f || memcmp(C.fh.magic, "XNG2", 4) != 0) return false;
    if ((C.fh.cfg & ~SETTINGS_MASK) != FORMAT_WORD || C.fh.size_entry_bytes != 2 || C.fh.ds != dequant_scale_of(C.fh.eps)) {
        fprintf(stderr, "xnygzip: %s: unsupported format\n", path.c_str());
        return false;
    }
    C.sizes.resize(C.fh.grid);
    for (auto& s : C.sizes) { uint16_t v; f.read((char*)&v, 2); s = v; }
    C.payload.resize(C.fh.payload_bytes);
    f.read((char*)C.payload.data(), C.payload.size());
    return (bool)f;
}

// search budgets: "fast" (shortlist of 1), "default" (up to 3), "exhaustive" (every mode for every block)
inline bool search_budget(const std::string& name, Config& c) {
    c = Config();
    if (name == "default") c.adapt = 3;
    else if (name == "fast") c.adapt = 1;
    else if (name == "exhaustive") { c.select = SEL_COST; c.adapt = 0; }
    else return false;
    return true;
}

inline bool read_aos(const std::string& path, std::vector<float>& x, std::vector<float>& y, std::vector<float>& z) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const uint64_t sz = f.tellg();
    f.seekg(0);
    if (sz % 12) return false;
    const size_t n = sz / 12;
    std::vector<float> buf(n * 3);
    f.read((char*)buf.data(), sz);
    x.resize(n); y.resize(n); z.resize(n);
    for (size_t i = 0; i < n; ++i) { x[i] = buf[3 * i]; y[i] = buf[3 * i + 1]; z[i] = buf[3 * i + 2]; }
    return true;
}

}  // namespace xnyg

#ifndef XNYGZIP_NO_MAIN
int main(int argc, char** argv) {
    using namespace xnyg;
    if (argc < 4) {
        fprintf(stderr, "usage:\n  %s compress <in.bin> <eps|rel=R> <out.xng> [--search=fast|default|exhaustive]\n"
                        "  %s decompress <in.xng> <out.bin>\n"
                        "in.bin / out.bin: float32 x y z triples; rel=R: eps = R x bounding-box diagonal\n", argv[0], argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "compress") {
        if (argc < 5) return 1;
        std::vector<float> x, y, z;
        if (!read_aos(argv[2], x, y, z) || x.empty()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        float eps;
        const std::string e = argv[3];
        if (e.rfind("rel=", 0) == 0) {
            float mn[3] = {x[0], y[0], z[0]}, mx[3] = {x[0], y[0], z[0]};
            for (size_t i = 0; i < x.size(); ++i) {
                mn[0] = std::min(mn[0], x[i]); mx[0] = std::max(mx[0], x[i]); mn[1] = std::min(mn[1], y[i]);
                mx[1] = std::max(mx[1], y[i]); mn[2] = std::min(mn[2], z[i]); mx[2] = std::max(mx[2], z[i]);
            }
            float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
            eps = (float)(atof(e.c_str() + 4) * std::sqrt(dx * dx + dy * dy + dz * dz));
        } else eps = (float)atof(e.c_str());
        Config cfg;
        for (int i = 5; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("--search=", 0) != 0 || !search_budget(a.substr(9), cfg)) { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
        }
        Compressed C = compress(x.data(), y.data(), z.data(), (int64_t)x.size(), eps, cfg);
        if (!save(argv[4], C)) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
        printf("n=%zu eps=%.9g file_bytes=%llu ratio=%.4f bpp=%.4f\n", x.size(), eps, (unsigned long long)C.file_bytes(),
               (double)x.size() * 12.0 / C.file_bytes(), C.file_bytes() * 8.0 / x.size());
    } else if (mode == "decompress") {
        Compressed C;
        if (!load(argv[2], C)) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        std::vector<float> x(C.fh.n), y(C.fh.n), z(C.fh.n);
        decompress(C, x.data(), y.data(), z.data());
        std::ofstream f(argv[3], std::ios::binary);
        for (size_t i = 0; i < x.size(); ++i) { f.write((char*)&x[i], 4); f.write((char*)&y[i], 4); f.write((char*)&z[i], 4); }
        printf("decoded %llu points\n", (unsigned long long)C.fh.n);
    } else return 1;
    return 0;
}
#endif
