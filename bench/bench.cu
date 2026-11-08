// bench.cu - compression ratio and GPU-resident throughput of XnYGZip
//
// usage: xnygzip_bench <in.bin> <dataset> <out.csv> [--R=1e-5,...] [--search=fast,default,exhaustive]
//                      [--warmup=10] [--reps=30]
//
// The input is copied to the device once; every timed region starts and ends with device-resident data
// (file I/O, host-device copies of the input and of the compressed stream are excluded).
//   compression    compress_device(): sampling pass + encoding kernel + device-wide scan + gather (or the
//                  single-pass emission), including the copy of the 8-byte total size to the host
//   decompression  prefix scan of the per-chunk size table + decoding kernel
// Times are CUDA-event times, median of `reps` runs after `warmup` runs. Throughput = uncompressed bytes
// (12 B per point) / time, 1 GB = 1e9 B. All search budgets and bounds are timed in the same process.
#define XNYGZIP_NO_MAIN
#include "../xnygzip.cu"
#include "common.h"

using namespace xnyg;

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.bin> <dataset> <out.csv> [--R=r,...] [--search=fast,default,exhaustive] [--warmup=10] [--reps=30]\n", argv[0]);
        return 1;
    }
    const std::string in = argv[1], name = argv[2], csv = argv[3];
    std::vector<std::string> Rs = split_list(paper_bounds()), budgets = {"fast", "default"};
    int warmup = 10, reps = 30;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--R=", 0) == 0) Rs = split_list(a.substr(4));
        else if (a.rfind("--search=", 0) == 0) budgets = split_list(a.substr(9));
        else if (a.rfind("--warmup=", 0) == 0) warmup = atoi(a.c_str() + 9);
        else if (a.rfind("--reps=", 0) == 0) reps = atoi(a.c_str() + 7);
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    for (auto& b : budgets) { Config c; if (!search_budget(b, c)) { fprintf(stderr, "unknown search budget %s\n", b.c_str()); return 1; } }

    std::vector<float> hx, hy, hz;
    if (!load_points(in, hx, hy, hz) || hx.empty()) { fprintf(stderr, "cannot read %s\n", in.c_str()); return 1; }
    const int64_t n = (int64_t)hx.size();
    const int grid = (int)((n + NB - 1) / NB);
    const double raw = (double)n * 12.0, diag = bbox_diag(hx, hy, hz);

    DevBufs B;
    B.alloc_compress(grid);
    XNYG_CHECK(cudaMemcpy(B.x, hx.data(), n * 4, cudaMemcpyHostToDevice));
    XNYG_CHECK(cudaMemcpy(B.y, hy.data(), n * 4, cudaMemcpyHostToDevice));
    XNYG_CHECK(cudaMemcpy(B.z, hz.data(), n * 4, cudaMemcpyHostToDevice));
    // decoded coordinates go to the slot buffer (grid x 12,544 B >= 12 B x n), which the decoder does not read;
    // each axis starts 256-byte aligned
    const size_t ax = ((size_t)n + 63) / 64 * 64;
    float* const dx = reinterpret_cast<float*>(B.slots);
    float* const dy = dx + ax;
    float* const dz = dy + ax;

    cudaDeviceProp prop;
    int dev;
    XNYG_CHECK(cudaGetDevice(&dev));
    XNYG_CHECK(cudaGetDeviceProperties(&prop, dev));
    cudaEvent_t e0, e1;
    XNYG_CHECK(cudaEventCreate(&e0)); XNYG_CHECK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& fn) {
        for (int i = 0; i < warmup; ++i) fn();
        XNYG_CHECK(cudaDeviceSynchronize());
        std::vector<double> ms;
        for (int i = 0; i < reps; ++i) {
            XNYG_CHECK(cudaEventRecord(e0));
            fn();
            XNYG_CHECK(cudaEventRecord(e1));
            XNYG_CHECK(cudaEventSynchronize(e1));
            float t; XNYG_CHECK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        XNYG_CHECK(cudaGetLastError());
        return median(ms);
    };

    const bool exists = std::ifstream(csv).good();
    FILE* o = fopen(csv.c_str(), "a");
    if (!o) { fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
    if (!exists) fprintf(o, "dataset,n,R,eps,search,file_bytes,ratio,bpp,comp_ms,comp_GBps,decomp_ms,decomp_GBps,warmup,reps,gpu,stream_hash\n");
    for (const auto& Rstr : Rs)
        for (const auto& budget : budgets) {
            const double R = atof(Rstr.c_str());
            const float eps = (float)(R * diag);
            Config cfg;
            search_budget(budget, cfg);
            uint64_t total = 0;
            const double cms = timeit([&]() { total = compress_device(B, n, grid, eps, cfg); });
            total = compress_device(B, n, grid, eps, cfg);
            if (const unsigned e = read_and_clear_errors()) { fprintf(stderr, "header size-model mismatches: %u\n", e); return 3; }
            const uint64_t file_bytes = sizeof(FileHeader) + (uint64_t)grid * 2 + total;
            // FNV-1a over the size table and the stream
            uint64_t h = 1469598103934665603ull;
            {
                std::vector<uint32_t> sz(grid);
                std::vector<uint8_t> st(total);
                XNYG_CHECK(cudaMemcpy(sz.data(), B.sizes, (size_t)grid * 4, cudaMemcpyDeviceToHost));
                XNYG_CHECK(cudaMemcpy(st.data(), B.comp, total, cudaMemcpyDeviceToHost));
                auto mix = [&](const uint8_t* p, size_t k) { for (size_t i = 0; i < k; ++i) { h ^= p[i]; h *= 1099511628211ull; } };
                mix(reinterpret_cast<const uint8_t*>(sz.data()), sz.size() * 4); mix(st.data(), st.size());
            }
            XNYG_CHECK(cudaMemset(B.comp + total, 0, 16));   // guard words read by the decoder
            const float ds = dequant_scale_of(eps);
            const double dms = timeit([&]() {
                k_widen<<<(grid + 1 + 255) / 256, 256>>>(B.sizes, B.sz64, grid);
                size_t tb = B.tmp_bytes;
                cub::DeviceScan::ExclusiveSum(B.tmp, tb, B.sz64, B.offs, grid + 1);
                k_decompress<<<grid, BT>>>(reinterpret_cast<const uint32_t*>(B.comp), B.offs, n, ds, dx, dy, dz, nullptr, nullptr);
            });
            const double ratio = raw / file_bytes, cg = raw / 1e9 / (cms / 1e3), dg = raw / 1e9 / (dms / 1e3);
            fprintf(o, "%s,%lld,%s,%.9g,%s,%llu,%.6f,%.6f,%.4f,%.2f,%.4f,%.2f,%d,%d,%s,%016llx\n", name.c_str(), (long long)n, Rstr.c_str(),
                    eps, budget.c_str(), (unsigned long long)file_bytes, ratio, file_bytes * 8.0 / n, cms, cg, dms, dg, warmup, reps,
                    prop.name, (unsigned long long)h);
            fflush(o);
            printf("%-10s R=%-6s %-10s CR %10.3f  comp %8.2f GB/s  decomp %8.2f GB/s\n", name.c_str(), Rstr.c_str(), budget.c_str(), ratio, cg, dg);
            fflush(stdout);
        }
    fclose(o);
    B.free_all();
    return 0;
}
