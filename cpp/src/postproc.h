// postproc.h —— YOLOv8-seg 后处理的 C++ 移植（W4② 里程碑 2）
//
// 语义逐条对齐 induvision/pipeline.py::postprocess() 一族：
//   _as_predictions / sigmoid / cxcywh_to_xyxy / batched_nms /
//   scale_boxes_to_orig / masks_from_proto。
// 每一处与 Python 版的数值口径差异都在实现处用注释钉住；不允许"顺手优化"。
#pragma once

#include <vector>

namespace induvision {

// letterbox 几何参数（对齐 deploy.letterbox / bench_gpu_preproc.letterbox_meta）。
// dw/dh 是**浮点半 pad**（scale_boxes 用它），top/left 是取整后的实际 pad
// （前处理 kernel 用它）——两者在余量为奇数时差 1 像素，这是 ultralytics 口径。
struct LetterboxMeta {
  int orig_h = 0, orig_w = 0;
  int imgsz = 640;
  int nu_w = 0, nu_h = 0;   // new_unpad
  int top = 0, left = 0;
  double dw = 0.0, dh = 0.0;  // (imgsz - nu_w) / 2.0 浮点
  double ratio = 1.0;
};

// 一个检测结果（对齐 pipeline.Detection；掩码只留像素数——压测不传 810KB 的 bool）。
struct Det {
  double score = 0.0;
  int cls = 0;
  double box[4] = {0, 0, 0, 0};  // xyxy，原图坐标
  int mask_pixels = 0;
};

// out0: (attrs, anchors) float32，行主序（out0[a*anchors + k] = 第 a 属性第 k 锚点）
// out1: (proto_c, proto_h, proto_w) float32，prototype
// 返回按分数降序、截断 max_det 的检测（对齐 postprocess 的输出顺序）。
std::vector<Det> postprocess(const float* out0, int attrs, int anchors,
                             const float* out1, int proto_c, int proto_h,
                             int proto_w, const LetterboxMeta& lb,
                             double conf_thr, double iou_thr, int max_det,
                             double mask_thr, bool need_masks);

// Python 的 round() 是银行家舍入（round-half-to-even），C++ std::round 是
// 半远离零。masks_from_proto 里有三处 round()，.5 恰好出现时会差 1 像素，
// 所以这里复刻银行家舍入，不允许换实现。
double py_round(double v);

}  // namespace induvision
