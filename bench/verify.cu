// verify.cu - strict error-bound verification, compression ratio and PSNR of XnYGZip
//
// usage: xnygzip_verify <in.bin> <dataset> <out.csv> [--R=1e-5,...] [--search=fast,default,exhaustive]
//
// The input is compressed as one chunk (the same stream as bench.cu), decoded on the GPU (coordinates, per-point
// kind and lattice point), and checked on the CPU without trusting any compression-side number. The decoder
// reorders points inside each 1,024-point chunk, so the check is per chunk:
//   - lossless chunks: the decoded points equal the originals (multiset, bitwise);
//   - exceptions: every decoded exception equals a distinct original exactly;
//   - every other original is re-quantized on the CPU (the codec's quantizer, same float arithmetic); the multiset
//     of these lattice points equals the multiset of decoded lattice points, and every decoded coordinate equals
//     q * ds + origin of its chunk, bit for bit. Each original is therefore reconstructed at q * ds + origin,
//     and its Euclidean error is checked against eps in double precision (strict, no tolerance).
// PSNR = 20 log10(diag / RMSE), RMSE over the pointwise Euclidean errors.
#define XNYGZIP_NO_MAIN
#include "../xnygzip.cu"
#include "common.h"
#include <array>
#include <omp.h>

using namespace xnyg;

struct ChunkResult {
    int64_t violations = 0, mismatches = 0;
    double sse = 0, max_err = 0;
};

static ChunkResult check_chunk(const float* X, const float* Y, const float* Z, int64_t b0, int nv, const float* ox,
                               const float* oy, const float* oz, const uint8_t* kind, const int32_t* dq,
                               const uint32_t* words, uint64_t bit_off, float qs, float ds, double eps) {
    ChunkResult r;
    Hdr h;
    GlobReader rd{words, bit_off};
    header_parse(rd, h);
    typedef std::array<uint32_t, 3> F3;
    auto bits = [](float a, float b, float c) { return F3{__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b), __builtin_bit_cast(uint32_t, c)}; };
    std::vector<F3> orig(nv);
    for (int l = 0; l < nv; ++l) orig[l] = bits(X[b0 + l], Y[b0 + l], Z[b0 + l]);
    if (h.mode == M_LOSSLESS) {
        std::vector<F3> dec(nv);
        for (int l = 0; l < nv; ++l) { dec[l] = bits(ox[b0 + l], oy[b0 + l], oz[b0 + l]); r.mismatches += kind[b0 + l] != 2; }
        std::vector<F3> a = orig;
        std::sort(a.begin(), a.end()); std::sort(dec.begin(), dec.end());
        r.mismatches += a != dec;
        return r;
    }
    // exceptions: matched to distinct, bitwise-equal originals
    std::vector<int> ord(nv);
    for (int l = 0; l < nv; ++l) ord[l] = l;
    std::sort(ord.begin(), ord.end(), [&](int i, int j) { return orig[i] < orig[j] || (orig[i] == orig[j] && i < j); });
    std::vector<char> exact(nv, 0);
    for (int l = 0; l < nv; ++l)
        if (kind[b0 + l] == 1) {
            const F3 key = bits(ox[b0 + l], oy[b0 + l], oz[b0 + l]);
            auto it = std::lower_bound(ord.begin(), ord.end(), key, [&](int i, const F3& k) { return orig[i] < k; });
            while (it != ord.end() && orig[*it] == key && exact[*it]) ++it;
            if (it == ord.end() || orig[*it] != key) ++r.mismatches; else exact[*it] = 1;
        }
    // CPU re-quantization of the chunk (block minimum, fine BCC lattice)
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    for (int l = 0; l < nv; ++l) { mn[0] = std::fmin(mn[0], X[b0 + l]); mn[1] = std::fmin(mn[1], Y[b0 + l]); mn[2] = std::fmin(mn[2], Z[b0 + l]); }
    const float neg[3] = {-mn[0] * qs, -mn[1] * qs, -mn[2] * qs};
    const float og[3] = {__builtin_bit_cast(float, h.origin[0]), __builtin_bit_cast(float, h.origin[1]), __builtin_bit_cast(float, h.origin[2])};
    std::vector<std::array<int32_t, 3>> expq, decq;
    for (int l = 0; l < nv; ++l) {
        const float p[3] = {X[b0 + l], Y[b0 + l], Z[b0 + l]};
        if (exact[l]) continue;   // stored exactly: zero error
        const float px = std::fma(p[0], qs, neg[0]), py = std::fma(p[1], qs, neg[1]), pz = std::fma(p[2], qs, neg[2]);
        std::array<int32_t, 3> q{0, 0, 0};
        const bool fits = std::fabs(px) < 1073741824.f && std::fabs(py) < 1073741824.f && std::fabs(pz) < 1073741824.f;
        if (fits) bcc_nearest_f(px, py, pz, q[0], q[1], q[2]);
        if (!fits || (((q[0] ^ q[1]) | (q[1] ^ q[2])) & 1)) { ++r.violations; continue; }   // had to be an exception
        expq.push_back(q);
        double s = 0;
        for (int a = 0; a < 3; ++a) { const double d = (double)p[a] - (double)std::fma((float)q[a], ds, og[a]); s += d * d; }
        r.sse += s;
        r.max_err = std::max(r.max_err, std::sqrt(s));
        r.violations += !(std::sqrt(s) <= eps);
    }
    for (int l = 0; l < nv; ++l) {
        if (kind[b0 + l] == 1) continue;
        if (kind[b0 + l] != 0) { ++r.mismatches; continue; }
        const int32_t* q = dq + 3 * (b0 + l);
        decq.push_back({q[0], q[1], q[2]});
        const float got[3] = {ox[b0 + l], oy[b0 + l], oz[b0 + l]};
        for (int a = 0; a < 3; ++a)
            if (__builtin_bit_cast(uint32_t, std::fma((float)q[a], ds, og[a])) != __builtin_bit_cast(uint32_t, got[a])) { ++r.mismatches; break; }
    }
    std::sort(expq.begin(), expq.end()); std::sort(decq.begin(), decq.end());
    r.mismatches += expq != decq;
    return r;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.bin> <dataset> <out.csv> [--R=r,...] [--search=fast,default,exhaustive]\n", argv[0]);
        return 1;
    }
    const std::string in = argv[1], name = argv[2], csv = argv[3];
    std::vector<std::string> Rs = split_list(paper_bounds()), budgets = {"fast", "default"};
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--R=", 0) == 0) Rs = split_list(a.substr(4));
        else if (a.rfind("--search=", 0) == 0) budgets = split_list(a.substr(9));
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    std::vector<float> X, Y, Z;
    if (!load_points(in, X, Y, Z) || X.empty()) { fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
    const int64_t n = (int64_t)X.size(), grid = (n + NB - 1) / NB;
    const double diag = bbox_diag(X, Y, Z);
    std::vector<float> ox(n), oy(n), oz(n);
    std::vector<uint8_t> kind(n);
    std::vector<int32_t> dq((size_t)n * 3);

    const bool exists = std::ifstream(csv).good();
    FILE* o = fopen(csv.c_str(), "a");
    if (!o) { fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
    if (!exists) fprintf(o, "dataset,n,R,eps,search,file_bytes,ratio,bpp,psnr,max_err_over_eps,violations,mismatches,pass\n");
    int failed = 0;
    for (const auto& Rstr : Rs)
        for (const auto& budget : budgets) {
            const float eps = (float)(atof(Rstr.c_str()) * diag);
            Config cfg;
            if (!search_budget(budget, cfg)) { fprintf(stderr, "unknown search budget %s\n", budget.c_str()); return 1; }
            const Compressed C = compress(X.data(), Y.data(), Z.data(), n, eps, cfg, false, grid);
            decompress(C, ox.data(), oy.data(), oz.data(), kind.data(), dq.data(), grid);
            std::vector<uint64_t> off(grid + 1, 0);
            for (int64_t b = 0; b < grid; ++b) off[b + 1] = off[b] + C.sizes[b];
            std::vector<uint32_t> words((C.payload.size() + 3) / 4 + 4, 0);
            memcpy(words.data(), C.payload.data(), C.payload.size());
            int64_t viol = 0, mism = 0;
            double sse = 0, maxe = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : viol, mism, sse) reduction(max : maxe)
            for (int64_t b = 0; b < grid; ++b) {
                const int64_t b0 = b * NB;
                const ChunkResult r = check_chunk(X.data(), Y.data(), Z.data(), b0, (int)std::min<int64_t>(NB, n - b0), ox.data(), oy.data(),
                                                  oz.data(), kind.data(), dq.data(), words.data(), off[b] * 8, C.fh.qs, C.fh.ds, eps);
                viol += r.violations; mism += r.mismatches; sse += r.sse; maxe = std::max(maxe, r.max_err);
            }
            const double ratio = n * 12.0 / C.file_bytes(), mse = sse / n;
            const double psnr = mse > 0 ? 10.0 * std::log10(diag * diag / mse) : INFINITY;
            const bool pass = viol == 0 && mism == 0;
            failed += !pass;
            fprintf(o, "%s,%lld,%s,%.9g,%s,%llu,%.6f,%.6f,%.4f,%.6f,%lld,%lld,%s\n", name.c_str(), (long long)n, Rstr.c_str(), eps,
                    budget.c_str(), (unsigned long long)C.file_bytes(), ratio, C.file_bytes() * 8.0 / n, psnr, maxe / eps,
                    (long long)viol, (long long)mism, pass ? "True" : "False");
            fflush(o);
            printf("%-10s R=%-6s %-10s CR %10.3f  PSNR %7.2f dB  max_err/eps %.6f  %s\n", name.c_str(), Rstr.c_str(), budget.c_str(), ratio,
                   psnr, maxe / eps, pass ? "PASS" : "FAIL");
            fflush(stdout);
        }
    fclose(o);
    return failed ? 2 : 0;
}
