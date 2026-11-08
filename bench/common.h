// common.h - input loading, error bound and argument helpers shared by bench.cu and verify.cu
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// float32 x y z triples (AoS, no header) -> SoA
inline bool load_points(const std::string& path, std::vector<float>& x, std::vector<float>& y, std::vector<float>& z) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const uint64_t bytes = f.tellg();
    if (bytes % 12) return false;
    const size_t n = bytes / 12;
    f.seekg(0);
    std::vector<float> aos(n * 3);
    f.read((char*)aos.data(), bytes);
    if (!f) return false;
    x.resize(n); y.resize(n); z.resize(n);
    for (size_t i = 0; i < n; ++i) { x[i] = aos[3 * i]; y[i] = aos[3 * i + 1]; z[i] = aos[3 * i + 2]; }
    return true;
}

// bounding-box diagonal: the absolute bound of a relative bound R is eps = R x diag
inline double bbox_diag(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& z) {
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i = 0; i < x.size(); ++i) {
        mn[0] = std::fmin(mn[0], x[i]); mx[0] = std::fmax(mx[0], x[i]);
        mn[1] = std::fmin(mn[1], y[i]); mx[1] = std::fmax(mx[1], y[i]);
        mn[2] = std::fmin(mn[2], z[i]); mx[2] = std::fmax(mx[2], z[i]);
    }
    const float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
    return std::sqrt((double)dx * dx + (double)dy * dy + (double)dz * dz);
}

inline std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}

// the seven relative bounds of the paper
inline const char* paper_bounds() { return "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"; }
