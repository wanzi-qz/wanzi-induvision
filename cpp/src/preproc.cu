// preproc.cu —— letterbox + BGR→RGB + /255 融合 kernel（W4② 里程碑 2）
//
// 语义逐条对齐 tools/bench_gpu_preproc.py::GpuPreproc.__call__ 的 GPU 段：
//   x = uint8 BGR / 255（float32）→ 通道重排 RGB → bilinear(align_corners=False,
//   antialias=False) 缩放到 (nu_h, nu_w) → 常数 pad 114/255 → (1,3,640,640) float32。
//
// 与 torch 实现的两处精确对齐（不许改，改了张量对拍就漂）：
//   1. 源坐标公式 src = (dst + 0.5) * (in/out) - 0.5，负值夹 0（area_pixel_compute_
//      source_index 对 bilinear 的行为）；
//   2. 插值公式 = h0l * (w0l * p(h1,w1) + w1l * p(h1,w1p))
//                + h1l * (w0l * p(h1p,w1) + w1l * p(h1p,w1p))，
//      先 /255 再插值（torch 是在归一化后的 float 域上插值的）。
// 输出直接写 engine 的输入 device 缓冲 —— C++ 版没有 Python 版那笔 4.688 MiB
// 的 D2H 拷回（gpu_preproc.md 实测的"保守下界"差距就在这）。
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float src_index(int dst, int in_size, int out_size) {
  // area_pixel_compute_scale(align_corners=false) = in/out（float 口径）
  const float s = static_cast<float>(in_size) / static_cast<float>(out_size);
  float v = (static_cast<float>(dst) + 0.5f) * s - 0.5f;
  if (v < 0.0f) v = 0.0f;
  return v;
}

__global__ void letterbox_norm_kernel(const uint8_t* __restrict__ img, int W, int H,
                                      float* __restrict__ out, int imgsz,
                                      int nu_w, int nu_h, int top, int left,
                                      float pad_v) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= imgsz || oy >= imgsz) return;

  const int sx = ox - left, sy = oy - top;
  float rgb[3];
  if (sx < 0 || sx >= nu_w || sy < 0 || sy >= nu_h) {
    rgb[0] = rgb[1] = rgb[2] = pad_v;
  } else {
    const float fx = src_index(sx, W, nu_w);
    const float fy = src_index(sy, H, nu_h);
    int x1 = static_cast<int>(fx);
    int y1 = static_cast<int>(fy);
    const int x1p = (x1 < W - 1) ? x1 + 1 : x1;
    const int y1p = (y1 < H - 1) ? y1 + 1 : y1;
    const float wx1 = fx - x1, wy1 = fy - y1;
    const float wx0 = 1.0f - wx1, wy0 = 1.0f - wy1;

    // 输入是 BGR 交错（cv::imdecode 语义）：out[c] 读 in[2-c]，对齐 torch 的
    // x[[2,1,0]]（BGR→RGB）。先归一化再插值，插值公式与 torch 逐项一致。
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const uint8_t* base = img + (static_cast<size_t>(y1) * W + x1) * 3 + (2 - c);
      const float p00 = static_cast<float>(base[0]) / 255.0f;
      const float p01 = static_cast<float>(base[(x1p - x1) * 3]) / 255.0f;
      const float p10 = static_cast<float>(base[(y1p - y1) * W * 3]) / 255.0f;
      const float p11 =
          static_cast<float>(base[(y1p - y1) * W * 3 + (x1p - x1) * 3]) / 255.0f;
      rgb[c] = wy0 * (wx0 * p00 + wx1 * p01) + wy1 * (wx0 * p10 + wx1 * p11);
    }
  }
  float* dst = out + static_cast<size_t>(oy) * imgsz + ox;
  dst[0] = rgb[0];
  dst[static_cast<size_t>(imgsz) * imgsz] = rgb[1];
  dst[2 * static_cast<size_t>(imgsz) * imgsz] = rgb[2];
}

}  // namespace

extern "C" void launch_letterbox_norm(const uint8_t* d_img, int W, int H,
                                      float* d_out, int imgsz, int nu_w,
                                      int nu_h, int top, int left,
                                      float pad_v, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((imgsz + block.x - 1) / block.x,
                  (imgsz + block.y - 1) / block.y);
  letterbox_norm_kernel<<<grid, block, 0, stream>>>(d_img, W, H, d_out, imgsz,
                                                    nu_w, nu_h, top, left,
                                                    pad_v);
}
