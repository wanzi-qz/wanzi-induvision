// postproc.cpp —— YOLOv8-seg 后处理 C++ 移植。对照 induvision/pipeline.py 写，
// 每个小节都标了对应 Python 函数名；口径差异（如有）写在行内注释里。
#include "postproc.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace induvision {
namespace {

// ---- pipeline.sigmoid：先夹 ±60 再算，float32 全程
inline float sigmoid_f(float x) {
  const float z = std::clamp(x, -60.0f, 60.0f);
  return 1.0f / (1.0f + std::exp(-z));
}

// ---- pipeline.iou_one_vs_many（double 口径，并集 0 → 0）
double iou_one(const double b[4], const double bs[4]) {
  const double x1 = std::max(b[0], bs[0]);
  const double y1 = std::max(b[1], bs[1]);
  const double x2 = std::min(b[2], bs[2]);
  const double y2 = std::min(b[3], bs[3]);
  const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
  const double area_b =
      std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
  const double area_bs =
      std::max(0.0, bs[2] - bs[0]) * std::max(0.0, bs[3] - bs[1]);
  const double un = area_b + area_bs - inter;
  return un > 0 ? inter / std::max(un, 1e-12) : 0.0;
}

// 候选：conf 过滤后的一行（原 anchor 下标 + 类别分 + 框 + 系数指针）
struct Cand {
  int anchor;
  float score;        // float32（Python: score = cls_scores.max(axis=1) 原样保留）
  double box[4];      // xyxy 640 域（cxcywh_to_xyxy 在 float64 做）
  int label;
  const float* coeffs;  // 该 anchor 的首个系数（out0[(4+nc)*anchors + k]）
  int coeff_stride;     // 属性维步长（= anchors）
};

// ---- pipeline.nms：贪心，IoU > thr 才抑制（等于阈值两个都留），稳定排序
void nms_one_class(std::vector<int>& order, const std::vector<Cand>& cs,
                   double iou_thr, std::vector<int>* keep) {
  // np.argsort(-s, kind="stable")：分数降序、同分保持原顺序
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return cs[a].score > cs[b].score;
  });
  size_t head = 0;
  while (head < order.size()) {
    const int i = order[head++];
    keep->push_back(i);
    // rest[ious <= iou_thr]：原地过滤（保持相对顺序）
    std::vector<int> nxt;
    for (size_t k = head; k < order.size(); ++k) {
      const int j = order[k];
      if (iou_one(cs[i].box, cs[j].box) <= iou_thr) nxt.push_back(j);
    }
    // 把保留的写回 order[head..]，等价于 Python 的原地切片
    for (size_t k = 0; k < nxt.size(); ++k) order[head + k] = nxt[k];
    order.resize(head + nxt.size());
  }
}

// ---- pipeline.masks_from_proto（裁剪小块再放大，不是整图放大）
// 返回该检测的掩码像素数；退化情况（框在网格外）返回 0，不抛异常。
// coeffs 指向该 anchor 的第一个系数（out0[(4+nc)*anchors + k]），沿属性维的
// 步长是 coeff_stride（= anchors）—— out0 是 (attrs, anchors) 行主序，
// 这里步长写错过一版，帧 0 的掩码直接差 22 倍，别再"顺手"改回连续读。
int mask_pixel_count(const float* proto, int c, int mh, int mw,
                     const double box640[4], const float* coeffs,
                     int coeff_stride, const LetterboxMeta& lb, double mask_thr) {
  const double scale = static_cast<double>(lb.imgsz) / static_cast<double>(mh);
  const double inv = 1.0 / lb.ratio;
  const double dw = lb.dw, dh = lb.dh;
  const int oh = lb.orig_h, ow = lb.orig_w;

  const double x1 = box640[0] / scale, y1 = box640[1] / scale;
  const double x2 = box640[2] / scale, y2 = box640[3] / scale;
  // 对齐 torch 的 round().int()：floor(v+0.5)，再夹到 [0, lim]
  const auto g = [](double v, int lim) {
    return std::min(std::max(static_cast<int>(std::floor(v + 0.5)), 0), lim);
  };
  const int cx1 = g(x1, mw), cy1 = g(y1, mh), cx2 = g(x2, mw), cy2 = g(y2, mh);
  if (cx2 <= cx1 || cy2 <= cy1) return 0;

  const int cw = cx2 - cx1, chh = cy2 - cy1;  // 裁剪区网格尺寸
  // m = sigmoid(cf @ crop)：float32 口径。np 的 float32 matmul 有 SIMD 分块，
  // 逐元素累加顺序与标量实现可差 ~1ulp；阈值 0.5 边界上的像素可能翻转，
  // 这是跨运行时的口径内差异，判据（match_rate / 掩码 IoU）按 §4 走。
  std::vector<float> m(static_cast<size_t>(cw) * chh);
  for (int yy = 0; yy < chh; ++yy) {
    for (int xx = 0; xx < cw; ++xx) {
      float acc = 0.0f;
      const float* col = proto + (static_cast<size_t>(cy1 + yy) * mw + cx1 + xx);
      for (int ch = 0; ch < c; ++ch)
        acc += coeffs[static_cast<size_t>(ch) * coeff_stride]
               * col[static_cast<size_t>(ch) * mh * mw];
      m[static_cast<size_t>(yy) * cw + xx] = sigmoid_f(acc);
    }
  }
  // tw/th：这块网格在原图上的真实宽高（Python: int(round(...))，银行家舍入）
  const int tw = std::max(1, static_cast<int>(py_round(cw * scale * inv)));
  const int th = std::max(1, static_cast<int>(py_round(chh * scale * inv)));
  const int px1 = static_cast<int>(py_round((cx1 * scale - dw) * inv));
  const int py1 = static_cast<int>(py_round((cy1 * scale - dh) * inv));
  const int dx1 = std::max(px1, 0), dy1 = std::max(py1, 0);
  const int dx2 = std::min(px1 + tw, ow), dy2 = std::min(py1 + th, oh);
  if (dx2 <= dx1 || dy2 <= dy1) return 0;

  // cv2.resize INTER_LINEAR：与 Python 同一算法（float32 域）
  const cv::Mat src(chh, cw, CV_32F, m.data());
  cv::Mat up;
  cv::resize(src, up, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);

  // 调试探针（INDUVISION_MASK_DBG=<dir>）：把 m / up 落盘给 Python 对拍
  if (const char* dbg = std::getenv("INDUVISION_MASK_DBG")) {
    static std::atomic<int> seq{0};
    const int n = seq.fetch_add(1);
    char nm[512];
    std::snprintf(nm, sizeof(nm), "%s/m_%d_%dx%d.bin", dbg, n, chh, cw);
    if (FILE* f = std::fopen(nm, "wb")) {
      std::fwrite(m.data(), 4, m.size(), f);
      std::fclose(f);
    }
    std::snprintf(nm, sizeof(nm), "%s/up_%d_%dx%d.bin", dbg, n, th, tw);
    if (FILE* f = std::fopen(nm, "wb")) {
      std::fwrite(up.ptr<float>(0), 4, static_cast<size_t>(th) * tw, f);
      std::fclose(f);
    }
  }

  // 贴回全图并数像素（Python: full[dy1:dy2, dx1:dx2] = up[...] >= mask_thr）
  const int sx1 = dx1 - px1, sy1 = dy1 - py1;
  const int ww = dx2 - dx1, hh = dy2 - dy1;
  int cnt = 0;
  for (int yy = 0; yy < hh; ++yy) {
    const float* row = up.ptr<float>(sy1 + yy) + sx1;
    for (int xx = 0; xx < ww; ++xx)
      if (row[xx] >= mask_thr) ++cnt;
  }
  return cnt;
}

}  // namespace

double py_round(double v) {
  // 银行家舍入：.5 取偶。std::nearbyint 按 FE_TONEAREST 就是这个语义，
  // 但显式写出来防止有人"顺手"换成 std::round。
  const double f = std::floor(v);
  const double diff = v - f;
  if (diff > 0.5) return f + 1.0;
  if (diff < 0.5) return f;
  return (std::fmod(f, 2.0) == 0.0) ? f : f + 1.0;
}

std::vector<Det> postprocess(const float* out0, int attrs, int anchors,
                             const float* out1, int proto_c, int proto_h,
                             int proto_w, const LetterboxMeta& lb,
                             double conf_thr, double iou_thr, int max_det,
                             double mask_thr, bool need_masks) {
  const int nc = attrs - 4 - proto_c;
  if (nc <= 0) return {};

  // ---- _as_predictions：pred[k][a] = out0[a * anchors + k]
  // cls_scores = pred[:, 4:4+nc]；score = max，label = argmax（首个最大）
  std::vector<Cand> cs;
  cs.reserve(256);
  for (int k = 0; k < anchors; ++k) {
    const float* row = out0 + k;
    float best = -std::numeric_limits<float>::infinity();
    int bi = 0;
    for (int c = 0; c < nc; ++c) {
      const float v = row[static_cast<size_t>(4 + c) * anchors];
      if (v > best) { best = v; bi = c; }
    }
    if (!(best > conf_thr)) continue;  // Python: score > conf_thr 严格大于
    Cand cd;
    cd.anchor = k;
    cd.score = best;
    cd.label = bi;
    cd.box[0] = row[0 * static_cast<size_t>(anchors)];
    cd.box[1] = row[1 * static_cast<size_t>(anchors)];
    cd.box[2] = row[2 * static_cast<size_t>(anchors)];
    cd.box[3] = row[3 * static_cast<size_t>(anchors)];
    // cxcywh → xyxy（float64）
    const double hw = cd.box[2] / 2.0, hh = cd.box[3] / 2.0;
    const double cx = cd.box[0], cy = cd.box[1];
    cd.box[0] = cx - hw; cd.box[1] = cy - hh;
    cd.box[2] = cx + hw; cd.box[3] = cy + hh;
    cd.coeffs = out0 + static_cast<size_t>(4 + nc) * anchors + k;
    cd.coeff_stride = anchors;
    cs.push_back(cd);
  }
  if (cs.empty()) return {};

  // ---- batched_nms：逐类 NMS，跨类互不抑制；结果按分数全局降序
  std::vector<int> keep;
  {
    // 类别按 np.unique 的升序分组
    std::vector<int> labels;
    for (const auto& c : cs) labels.push_back(c.label);
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    for (const int c : labels) {
      std::vector<int> sel;
      for (size_t i = 0; i < cs.size(); ++i)
        if (cs[i].label == c) sel.push_back(static_cast<int>(i));
      nms_one_class(sel, cs, iou_thr, &keep);
    }
    std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) {
      return cs[a].score > cs[b].score;
    });
    if (max_det > 0 && static_cast<int>(keep.size()) > max_det)
      keep.resize(max_det);
  }

  // ---- scale_boxes_to_orig + masks_from_proto + 汇总
  // c.box 是 640 域 xyxy；掩码解码直接吃它（Python 传的就是 scale_boxes 之前的
  // boxes），d.box 才是 (box - dw) / ratio 的原图坐标。上一版把 c.box 先转了
  // 原图再喂掩码解码，网格直接错位，帧 0 掩码 102559 → 72140，别改回去。
  std::vector<Det> out;
  out.reserve(keep.size());
  for (const int i : keep) {
    const Cand& c = cs[i];
    Det d;
    d.score = static_cast<double>(c.score);
    d.cls = c.label;
    if (need_masks) {
      d.mask_pixels = mask_pixel_count(out1, proto_c, proto_h, proto_w,
                                       c.box, c.coeffs, c.coeff_stride, lb,
                                       mask_thr);
    }
    // 减浮点半 pad（不是 top/left！），除 ratio，按原图 clip —— 逐行对齐
    d.box[0] = std::clamp((c.box[0] - lb.dw) / lb.ratio, 0.0, static_cast<double>(lb.orig_w));
    d.box[1] = std::clamp((c.box[1] - lb.dh) / lb.ratio, 0.0, static_cast<double>(lb.orig_h));
    d.box[2] = std::clamp((c.box[2] - lb.dw) / lb.ratio, 0.0, static_cast<double>(lb.orig_w));
    d.box[3] = std::clamp((c.box[3] - lb.dh) / lb.ratio, 0.0, static_cast<double>(lb.orig_h));
    out.push_back(d);
  }
  return out;
}

}  // namespace induvision
