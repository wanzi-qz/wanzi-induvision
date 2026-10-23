// bench.cpp —— W4② C++ 推理版压测：decode → GPU 前处理 → TRT 推理 → CPU 后处理
//
// 两种模式（对齐 tools/bench_pipeline.py 的口径）：
//   serial    单线程逐帧，阶段内逐段计时；
//   pipeline  三线程 + 环形队列(capacity) + slot 池(pinned in/out + event)，
//             语义对齐 Python 版："每帧只 synchronize 一次"（post 段等 event）、
//             "出错要把令牌还回池子"、"等待计数按调用记不按循环轮数记"。
//
// 与 Python 版的分工（docs/cpp_infer.md §4）：
//   - 资源采样不在这写（口径一漂两张表不可比），C++ 压测跑在 Python 采样器监视下；
//   - 跨运行时对拍在 Python 侧做（tools/check_cpp_bench.py）；
//     本程序内部（serial vs pipeline）自己比 500 帧全等。
//
// dump 能力（对拍探针，只在 serial 模式生效）：
//   --dump-preproc N 前 N 帧：原始 BGR bin + (1,3,640,640) float bin + lb meta
//   --dump-outs     N 前 N 帧：out0/out1 bin + lb meta（后处理移植等价性检验）
//   --dump-dets     N 前 N 帧：完整检测记录（match_rate / IoU 对拍用）
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define NOMINMAX  // windows.h 的 min/max 宏会打碎 std::min/std::max
#include <windows.h>
#include <shellapi.h>

#include "engine_header.h"
#include "postproc.h"

namespace {

// MSVC 的 main() argv 是 ANSI 码表（GBK）编码：中文参数（--dump-dir 之类）
// 直接拿去 u8path 会按非法 UTF-8 解析，进程 terminate（0xC0000409）。
// 统一从 GetCommandLineW 重取，转成 UTF-8 —— 与 frames.txt 里的 UTF-8 路径
// 和 Python 侧的 str 口径一致。
std::vector<std::string> utf8_args() {
  int n = 0;
  LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
  if (!w) return {};
  std::vector<std::string> out;
  out.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0,
                                        nullptr, nullptr);
    std::string s(static_cast<size_t>(len > 0 ? len - 1 : 0), '\0');
    if (len > 1)
      WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s.data(), len, nullptr, nullptr);
    out.push_back(std::move(s));
  }
  LocalFree(w);
  return out;
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt  ] %s\n", msg);
  }
};

[[noreturn]] void die(const std::string& what) {
  std::fprintf(stderr, "[err  ] %s\n", what.c_str());
  std::exit(2);
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess)                                                  \
      die(std::string("CUDA 错误 ") + cudaGetErrorName(err__) + " @ " #call +  \
          ": " + cudaGetErrorString(err__));                                   \
  } while (0)

std::vector<uint8_t> read_file(const std::string& path) {
  // 数据集路径带中文（…\简历\induvision\…）：MSVC 的 ifstream 直接吃 UTF-8
  // char* 会按 ANSI 码表解释而打不开 —— 和 Python 侧 cv2.imread 的坑同源
  // （deploy.read_image_bgr 的 docstring）。经 filesystem::u8path 走宽字符 API。
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) die("打不开文件：" + path);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                              std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const void* data, size_t n) {
  std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) die("写不了文件：" + path);
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
}

// ---- 环形队列（语义对齐 pipeline.RingBuffer：容量、关闭广播、非阻塞 put）
template <typename T>
class Ring {
 public:
  explicit Ring(size_t cap) : cap_(cap) {}
  bool put(T v) {
    std::unique_lock<std::mutex> lk(m_);
    cv_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    cv_empty_.notify_one();
    return true;
  }
  bool get(T& v) {
    std::unique_lock<std::mutex> lk(m_);
    cv_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    v = std::move(q_.front());
    q_.pop_front();
    cv_full_.notify_one();
    return true;
  }
  void close() {
    std::unique_lock<std::mutex> lk(m_);
    closed_ = true;
    cv_empty_.notify_all();
    cv_full_.notify_all();
  }
 private:
  size_t cap_;
  std::deque<T> q_;
  std::mutex m_;
  std::condition_variable cv_full_, cv_empty_;
  bool closed_ = false;
};

// ---- slot 池（pinned 暂存 + pinned 输出 + event；令牌贯穿 decode→post）
struct Slot {
  uint8_t* pin_in = nullptr;      // BGR 原图暂存（pinned）
  float* pin_out0 = nullptr;      // (1,51,8400)
  float* pin_out1 = nullptr;      // (1,32,160,160)
  cudaEvent_t ev = nullptr;
  size_t img_bytes = 0;
  int img_w = 0, img_h = 0;
  bool in_use = false;
};

class SlotPool {
 public:
  explicit SlotPool(std::vector<Slot> slots) : slots_(std::move(slots)) {}
  int acquire() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] {
      for (size_t i = 0; i < slots_.size(); ++i)
        if (!slots_[i].in_use) { free_ = static_cast<int>(i); return true; }
      return false;
    });
    const int i = free_;
    slots_[i].in_use = true;
    free_ = -1;
    return i;
  }
  void release(int i) {
    std::unique_lock<std::mutex> lk(m_);
    slots_[i].in_use = false;
    cv_.notify_one();
  }
  Slot& at(int i) { return slots_[i]; }
 private:
  std::vector<Slot> slots_;
  std::mutex m_;
  std::condition_variable cv_;
  int free_ = -1;
};

// ---- letterbox 几何（逐行对齐 bench_gpu_preproc.letterbox_meta / deploy.letterbox）
induvision::LetterboxMeta letterbox_meta(int w, int h, int imgsz) {
  induvision::LetterboxMeta m;
  m.orig_w = w; m.orig_h = h; m.imgsz = imgsz;
  m.ratio = std::min(static_cast<double>(imgsz) / h,
                     static_cast<double>(imgsz) / w);
  m.nu_w = static_cast<int>(induvision::py_round(w * m.ratio));
  m.nu_h = static_cast<int>(induvision::py_round(h * m.ratio));
  const double dh = imgsz - m.nu_h, dw = imgsz - m.nu_w;
  m.dh = dh / 2.0; m.dw = dw / 2.0;
  m.top = static_cast<int>(induvision::py_round(dh / 2 - 0.1));
  m.left = static_cast<int>(induvision::py_round(dw / 2 - 0.1));
  return m;
}

void launch_preproc(const uint8_t* d_img, int W, int H, float* d_out,
                    const induvision::LetterboxMeta& m, cudaStream_t s);

extern "C" {
// preproc.cu 提供
void launch_letterbox_norm(const uint8_t* d_img, int W, int H, float* d_out,
                           int imgsz, int nu_w, int nu_h, int top, int left,
                           float pad_v, cudaStream_t stream);
}

void launch_preproc(const uint8_t* d_img, int W, int H, float* d_out,
                    const induvision::LetterboxMeta& m, cudaStream_t s) {
  launch_letterbox_norm(d_img, W, H, d_out, m.imgsz, m.nu_w, m.nu_h, m.top,
                        m.left, 114.0f / 255.0f, s);
}

// ---- 极简 JSON 输出（路径含中文与反斜杠，逐字符转义）
struct Json {
  std::string s;
  void kv_str(const std::string& k, const std::string& v, bool comma = true) {
    if (comma) s += ",";
    s += "\"" + k + "\":\"";
    for (const char ch : v) {
      if (ch == '"' || ch == '\\') { s += '\\'; s += ch; }
      else if (ch == '\n') s += "\\n";
      else s += ch;
    }
    s += "\"";
  }
  template <typename T>
  void kv(const std::string& k, const T& v, bool comma = true) {
    if (comma) s += ",";
    s += "\"" + k + "\":" + std::to_string(v);
  }
  void raw(const std::string& k, const std::string& v, bool comma = true) {
    if (comma) s += ",";
    s += "\"" + k + "\":" + v;
  }
};

// 检测记录落 JSON 数组元素
std::string det_json(const induvision::Det& d) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"score\":%.6f,\"cls\":%d,\"box\":[%.3f,%.3f,%.3f,%.3f],"
                "\"mask_pixels\":%d}",
                d.score, d.cls, d.box[0], d.box[1], d.box[2], d.box[3],
                d.mask_pixels);
  return buf;
}

// ---- FrameResult（对齐 bench_pipeline.FrameResult 的三元组口径）
struct FrameResult {
  int idx = 0;
  std::string path;
  int n_det = 0;
  long long mask_pixels = 0;
  double score_sum = 0.0;
  std::vector<induvision::Det> dets;  // 仅 --dump-dets 时保留
  double e2e_ms = 0.0;                // decode 开始 → post 结束
};

struct PostTask {
  int slot, idx;
  std::string path;
  double t_start_ms;  // steady_clock 相对毫秒
  induvision::LetterboxMeta lb;
  double infer_ms;  // infer 段 busy（异步发射耗时，供分段报表）
};

double since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc; (void)argv;  // 编码不可靠，用 utf8_args()（见上方注释）
  // stdout 重定向到管道时是全缓冲：崩溃（如加载期 DLL 问题）时输出会整块丢，
  // 排障看到的只有空串。无缓冲换取"死前说了什么"。
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  std::string engine_path, list_path, json_out, dump_dir, mode = "pipeline";
  int repeats = 2, capacity = 4, dump_preproc = 0, dump_outs = 0, dump_dets = 0;
  int imgsz = 640;
  double conf_thr = 0.25, iou_thr = 0.7, mask_thr = 0.5;
  int max_det = 300;
  bool need_masks = true;
  const std::vector<std::string> uargs = utf8_args();
  const int uargc = static_cast<int>(uargs.size());
  for (int i = 1; i < uargc; ++i) {
    const std::string& a = uargs[i];
    auto next = [&](const char* n) -> std::string {
      if (i + 1 >= uargc) die(std::string(n) + " 缺参数值");
      return uargs[++i];
    };
    if (a == "--engine") engine_path = next("--engine");
    else if (a == "--list") list_path = next("--list");
    else if (a == "--repeats") repeats = std::stoi(next("--repeats"));
    else if (a == "--capacity") capacity = std::stoi(next("--capacity"));
    else if (a == "--imgsz") imgsz = std::stoi(next("--imgsz"));
    else if (a == "--conf-thr") conf_thr = std::stod(next("--conf-thr"));
    else if (a == "--iou-thr") iou_thr = std::stod(next("--iou-thr"));
    else if (a == "--mask-thr") mask_thr = std::stod(next("--mask-thr"));
    else if (a == "--max-det") max_det = std::stoi(next("--max-det"));
    else if (a == "--no-masks") need_masks = false;
    else if (a == "--dump-preproc") { dump_preproc = std::stoi(next("--dump-preproc")); }
    else if (a == "--dump-outs") { dump_outs = std::stoi(next("--dump-outs")); }
    else if (a == "--dump-dets") { dump_dets = std::stoi(next("--dump-dets")); }
    else if (a == "--dump-dir") dump_dir = next("--dump-dir");
    else if (a == "--json-out") json_out = next("--json-out");
    else if (a == "--mode") mode = next("--mode");
    else die("未知参数：" + a);
  }
  if (engine_path.empty() || list_path.empty()) die("缺 --engine / --list");
  if (capacity < 2) die("capacity 至少 2（slot 池太小流水线没意义）");

  // ---- 读帧清单（Python 侧生成，保证与 W4① 同一 canonical 顺序）
  std::vector<std::string> files;
  {
    std::ifstream f(std::filesystem::u8path(list_path), std::ios::binary);
    if (!f) die("打不开清单：" + list_path);
    std::string line;
    while (std::getline(f, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.pop_back();
      if (!line.empty()) files.push_back(line);
    }
  }
  if (files.empty()) die("清单为空");
  struct Frame { int idx; const std::string* path; };
  std::vector<Frame> frames;
  frames.reserve(files.size() * repeats);
  for (int r = 0; r < repeats; ++r)
    for (size_t i = 0; i < files.size(); ++i)
      frames.push_back({static_cast<int>(r * files.size() + i), &files[i]});
  const int n_frames = static_cast<int>(frames.size());

  // ---- engine
  const auto bytes = read_file(engine_path);
  const size_t off = induvision::engine_stream_offset(bytes.data(), bytes.size());
  static Logger logger;
  auto* runtime = nvinfer1::createInferRuntime(logger);
  auto* engine = runtime->deserializeCudaEngine(bytes.data() + off, bytes.size() - off);
  if (!engine) die("deserializeCudaEngine 失败");
  auto* ctx = engine->createExecutionContext();
  if (!ctx) die("createExecutionContext 失败");

  // ---- I/O 契约自检（§7.2，不写死 tensor 名）
  std::string in_name, o0_name, o1_name;
  size_t in_bytes = 0, o0_bytes = 0, o1_bytes = 0;
  int proto_c = 0, proto_h = 0, proto_w = 0;
  {
    int n_in = 0, n_out = 0;
    const int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char* name = engine->getIOTensorName(i);
      const auto shape = engine->getTensorShape(name);
      size_t elems = 1;
      for (int d = 0; d < shape.nbDims; ++d) {
        if (shape.d[d] < 0) die(std::string(name) + " 有动态维，契约变了");
        elems *= static_cast<size_t>(shape.d[d]);
      }
      if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        die(std::string(name) + " 不是 FLOAT");
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        ++n_in; in_name = name; in_bytes = elems * sizeof(float);
      } else if (n_out == 0) {
        ++n_out; o0_name = name; o0_bytes = elems * sizeof(float);
      } else {
        ++n_out; o1_name = name; o1_bytes = elems * sizeof(float);
        // output1 是 (1,C,H,W)
        proto_c = shape.d[1]; proto_h = shape.d[2]; proto_w = shape.d[3];
      }
    }
    if (n_in != 1 || n_out != 2) die("I/O 数量与契约（1 入 2 出）不符");
  }
  const int anchors = static_cast<int>(o0_bytes / sizeof(float) /
                                       (4 + 15 + proto_c));
  const int attrs = 4 + 15 + proto_c;
  std::printf("[io   ] in %zu B / out0 %zu B / out1 %zu B（attrs=%d anchors=%d proto=%dx%dx%d）\n",
              in_bytes, o0_bytes, o1_bytes, attrs, anchors, proto_c, proto_h, proto_w);

  // ---- device / pinned 内存
  uint8_t* d_img = nullptr;
  float* d_in = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_img), 4096 * 4096 * 3));  // 上限 4K 三通道，够 MVTec 900²
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), in_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o0), o0_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o1), o1_bytes));
  if (!ctx->setTensorAddress(in_name.c_str(), d_in) ||
      !ctx->setTensorAddress(o0_name.c_str(), d_o0) ||
      !ctx->setTensorAddress(o1_name.c_str(), d_o1))
    die("setTensorAddress 失败");
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto mk_pin = [](void** p, size_t n) {
    CUDA_CHECK(cudaHostAlloc(p, n, cudaHostAllocDefault));
  };

  // ---- slot 池
  std::vector<Slot> slots(static_cast<size_t>(capacity));
  for (auto& s : slots) {
    mk_pin(reinterpret_cast<void**>(&s.pin_in), 4096 * 4096 * 3);
    mk_pin(reinterpret_cast<void**>(&s.pin_out0), o0_bytes);
    mk_pin(reinterpret_cast<void**>(&s.pin_out1), o1_bytes);
    CUDA_CHECK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
  }
  SlotPool pool(std::move(slots));

  const auto t_prog = std::chrono::steady_clock::now();
  std::atomic<long long> busy_ns{0}, wait_ns{0};

  // ================================================================ serial
  std::vector<FrameResult> results(n_frames);
  // dump 探针是"读一次文件、跑一次前处理/推理并落盘"的一次性行为，放压测循环
  // 里会污染计时，所以 main 里走 probe_dump()，serial 模式不做 dump。
  auto run_serial = [&](int round) {
    for (int fi = 0; fi < n_frames; ++fi) {
      const auto t0 = std::chrono::steady_clock::now();
      const std::string& path = *frames[fi].path;
      // -- decode：读字节 + imdecode（np.fromfile + imdecode 语义，中文路径免疫）
      const auto raw = read_file(path);
      cv::Mat img = cv::imdecode(cv::Mat(1, static_cast<int>(raw.size()), CV_8U,
                                         const_cast<uint8_t*>(raw.data())),
                                 cv::IMREAD_COLOR);
      if (img.empty()) die("imdecode 失败：" + path);
      const auto lb = letterbox_meta(img.cols, img.rows, imgsz);
      // -- 前处理：pinned 暂存 → H2D → kernel
      Slot& s = pool.at(0);
      std::memcpy(s.pin_in, img.data,
                  static_cast<size_t>(img.rows) * img.cols * 3);
      CUDA_CHECK(cudaMemcpyAsync(d_img, s.pin_in,
                                 static_cast<size_t>(img.rows) * img.cols * 3,
                                 cudaMemcpyHostToDevice, stream));
      launch_preproc(d_img, img.cols, img.rows, d_in, lb, stream);
      // -- 推理 + D2H + 每帧一次同步
      if (!ctx->enqueueV3(stream)) die("enqueueV3 失败");
      CUDA_CHECK(cudaMemcpyAsync(s.pin_out0, d_o0, o0_bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(s.pin_out1, d_o1, o1_bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // -- 后处理
      auto dets = induvision::postprocess(s.pin_out0, attrs, anchors,
                                          s.pin_out1, proto_c, proto_h,
                                          proto_w, lb, conf_thr, iou_thr,
                                          max_det, mask_thr, need_masks);
      FrameResult& r = results[fi];
      r.idx = frames[fi].idx; r.path = path; r.n_det = static_cast<int>(dets.size());
      r.dets = std::move(dets);
      for (const auto& d : r.dets) r.mask_pixels += d.mask_pixels;
      for (const auto& d : r.dets) r.score_sum += d.score;
      r.e2e_ms = since(t0);
    }
    (void)round;
  };
  // dump_preproc / dump_outs 是"读一次文件、跑一次前处理/推理并落盘"的一次性
  // 探针，放压测循环里会污染计时。单独做：
  auto probe_dump = [&](int n_pre, int n_out) {
    if (n_pre <= 0 && n_out <= 0) return;
    std::filesystem::create_directories(std::filesystem::u8path(dump_dir));  // 同 read_file：窄字符会按 GBK 误读中文路径
    for (int fi = 0; fi < std::max(n_pre, n_out); ++fi) {
      const auto raw = read_file(*frames[fi].path);
      cv::Mat img = cv::imdecode(cv::Mat(1, static_cast<int>(raw.size()), CV_8U,
                                         const_cast<uint8_t*>(raw.data())),
                                 cv::IMREAD_COLOR);
      if (img.empty()) die("imdecode 失败：" + *frames[fi].path);
      const auto lb = letterbox_meta(img.cols, img.rows, imgsz);
      char nm[80];
      if (fi < n_pre) {
        std::snprintf(nm, sizeof(nm), "%05d_raw.bin", fi);
        write_file(dump_dir + "/" + nm, img.data,
                   static_cast<size_t>(img.rows) * img.cols * 3);
        CUDA_CHECK(cudaMemcpyAsync(d_img, img.data,
                                   static_cast<size_t>(img.rows) * img.cols * 3,
                                   cudaMemcpyHostToDevice, stream));
        launch_preproc(d_img, img.cols, img.rows, d_in, lb, stream);
        std::vector<float> chw(in_bytes / 4);
        CUDA_CHECK(cudaMemcpyAsync(chw.data(), d_in, in_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::snprintf(nm, sizeof(nm), "%05d_chw.bin", fi);
        write_file(dump_dir + "/" + nm, chw.data(), in_bytes);
        Json j;
        j.s = "{";
        j.kv("w", img.cols, false); j.kv("h", img.rows); j.kv("imgsz", imgsz);
        j.kv("nu_w", lb.nu_w); j.kv("nu_h", lb.nu_h);
        j.kv("top", lb.top); j.kv("left", lb.left);
        std::snprintf(nm, sizeof(nm), ",\"dw\":%.17g,\"dh\":%.17g,\"ratio\":%.17g}",
                      lb.dw, lb.dh, lb.ratio);
        j.s += nm;
        std::snprintf(nm, sizeof(nm), "%05d_meta.json", fi);
        write_file(dump_dir + "/" + nm, j.s.data(), j.s.size());
      }
      if (fi < n_out) {
        CUDA_CHECK(cudaMemcpyAsync(d_img, img.data,
                                   static_cast<size_t>(img.rows) * img.cols * 3,
                                   cudaMemcpyHostToDevice, stream));
        launch_preproc(d_img, img.cols, img.rows, d_in, lb, stream);
        if (!ctx->enqueueV3(stream)) die("enqueueV3 失败");
        std::vector<float> o0(o0_bytes / 4), o1(o1_bytes / 4);
        CUDA_CHECK(cudaMemcpyAsync(o0.data(), d_o0, o0_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(o1.data(), d_o1, o1_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::snprintf(nm, sizeof(nm), "%05d_out0.bin", fi);
        write_file(dump_dir + "/" + nm, o0.data(), o0_bytes);
        std::snprintf(nm, sizeof(nm), "%05d_out1.bin", fi);
        write_file(dump_dir + "/" + nm, o1.data(), o1_bytes);
        Json j;
        j.s = "{";
        j.kv("w", img.cols, false); j.kv("h", img.rows); j.kv("imgsz", imgsz);
        j.kv("nu_w", lb.nu_w); j.kv("nu_h", lb.nu_h);
        j.kv("top", lb.top); j.kv("left", lb.left);
        std::snprintf(nm, sizeof(nm), ",\"dw\":%.17g,\"dh\":%.17g,\"ratio\":%.17g}",
                      lb.dw, lb.dh, lb.ratio);
        j.s += nm;
        std::snprintf(nm, sizeof(nm), "%05d_io_meta.json", fi);
        write_file(dump_dir + "/" + nm, j.s.data(), j.s.size());
      }
      std::printf("[dump ] 帧 %d 探针已落盘\n", fi);
    }
  };

  // ================================================================ pipeline
  Ring<PostTask> q1(static_cast<size_t>(capacity)), q2(static_cast<size_t>(capacity));
  Ring<FrameResult> q3(static_cast<size_t>(capacity) * 2);
  std::atomic<int> decoded{0}, inferred{0};
  auto run_pipeline = [&] {
    decoded = inferred = 0;
    // -- decode 线程：读图 + imdecode + meta + 拷 pinned 暂存
    std::thread t_dec([&] {
      for (int fi = 0; fi < n_frames; ++fi) {
        const auto tw0 = std::chrono::steady_clock::now();
        const std::string& path = *frames[fi].path;
        const auto raw = read_file(path);
        cv::Mat img = cv::imdecode(cv::Mat(1, static_cast<int>(raw.size()), CV_8U,
                                           const_cast<uint8_t*>(raw.data())),
                                   cv::IMREAD_COLOR);
        if (img.empty()) die("imdecode 失败：" + path);
        const auto lb = letterbox_meta(img.cols, img.rows, imgsz);
        const int slot = pool.acquire();
        Slot& s = pool.at(slot);
        s.img_bytes = static_cast<size_t>(img.rows) * img.cols * 3;
        s.img_w = img.cols; s.img_h = img.rows;
        std::memcpy(s.pin_in, img.data, s.img_bytes);
        const auto tw1 = std::chrono::steady_clock::now();
        busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tw1 - tw0).count();
        PostTask t{slot, frames[fi].idx, path, since(t_prog), lb,
                   0.0};
        decoded++;
        if (!q1.put(std::move(t))) { pool.release(slot); return; }
      }
      q1.close();
    });
    // -- infer 线程：H2D → kernel → enqueueV3 → D2H → 记 event（全程异步发射）
    std::thread t_inf([&] {
      PostTask t;
      while (q1.get(t)) {
        const auto tw0 = std::chrono::steady_clock::now();
        Slot& s = pool.at(t.slot);
        CUDA_CHECK(cudaMemcpyAsync(d_img, s.pin_in, s.img_bytes,
                                   cudaMemcpyHostToDevice, stream));
        launch_preproc(d_img, s.img_w, s.img_h, d_in, t.lb, stream);
        if (!ctx->enqueueV3(stream)) die("enqueueV3 失败");
        CUDA_CHECK(cudaMemcpyAsync(s.pin_out0, d_o0, o0_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(s.pin_out1, d_o1, o1_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(s.ev, stream));
        const auto tw1 = std::chrono::steady_clock::now();
        busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tw1 - tw0).count();
        t.infer_ms = std::chrono::duration<double, std::milli>(tw1 - tw0).count();
        inferred++;
        if (!q2.put(t)) { pool.release(t.slot); return; }
      }
      q2.close();
    });
    // -- post 线程：等 event（每帧唯一一次同步）→ 后处理 → 还令牌
    std::thread t_post([&] {
      PostTask t;
      while (q2.get(t)) {
        const auto tw0 = std::chrono::steady_clock::now();
        Slot& s = pool.at(t.slot);
        CUDA_CHECK(cudaEventSynchronize(s.ev));
        const auto tw1 = std::chrono::steady_clock::now();
        auto dets = induvision::postprocess(s.pin_out0, attrs, anchors,
                                            s.pin_out1, proto_c, proto_h,
                                            proto_w, t.lb, conf_thr, iou_thr,
                                            max_det, mask_thr, need_masks);
        FrameResult r;
        r.idx = t.idx; r.path = t.path;
        r.n_det = static_cast<int>(dets.size());
        for (const auto& d : dets) { r.mask_pixels += d.mask_pixels; r.score_sum += d.score; }
        r.dets = std::move(dets);
        r.e2e_ms = since(t_prog) - t.t_start_ms;
        pool.release(t.slot);
        const auto tw2 = std::chrono::steady_clock::now();
        busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tw2 - tw1).count();
        if (!q3.put(std::move(r))) return;
      }
      q3.close();
    });
    // -- 主线程按 idx 收集（order 语义）
    int got = 0;
    FrameResult r;
    while (q3.get(r)) {
      results[r.idx] = std::move(r);
      ++got;
      if (got == n_frames) break;
    }
    t_dec.join(); t_inf.join(); t_post.join();
  };

  // ---- dump 探针（压测前做一次，不污染计时）
  probe_dump(dump_preproc, dump_outs);

  // ---- 跑
  const auto wall0 = std::chrono::steady_clock::now();
  if (dump_preproc > 0 || dump_outs > 0) {
    std::puts("[mode ] 探针模式：不跑压测（探针在 serial 语义下落盘）");
  } else if (mode == "serial") {
    run_serial(0);
  } else {
    run_pipeline();
  }
  const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

  // ---- 汇总
  long long tot_dets = 0, tot_mp = 0;
  double tot_ss = 0.0;
  std::vector<double> e2es;
  for (const auto& r : results) {
    tot_dets += r.n_det;
    tot_mp += r.mask_pixels;
    tot_ss += r.score_sum;
    e2es.push_back(r.e2e_ms);
  }
  std::sort(e2es.begin(), e2es.end());
  auto pct = [&](double p) -> double {
    if (e2es.empty()) return 0.0;
    const double pos = p * (e2es.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, e2es.size() - 1);
    const double f = pos - lo;
    return e2es[lo] * (1 - f) + e2es[hi] * f;
  };
  char ss[64];
  std::snprintf(ss, sizeof(ss), "%.4f", tot_ss);

  Json j;
  j.s = "{";
  j.kv_str("schema", "induvision.cpp_bench.v1", false);
  j.kv_str("mode", (dump_preproc > 0 || dump_outs > 0) ? "probe" : mode);
  j.kv_str("engine", engine_path);
  j.kv("repeats", repeats);
  j.kv("capacity", capacity);
  j.kv("frames", n_frames);
  j.s += ",\"wall_s\":" + std::to_string(wall_s);
  j.s += ",\"fps\":" + std::to_string(n_frames / wall_s);
  j.s += ",\"checksum\":{\"frames\":" + std::to_string(n_frames) +
         ",\"dets\":" + std::to_string(tot_dets) +
         ",\"mask_pixels\":" + std::to_string(tot_mp) +
         ",\"score_sum\":" + ss + "}";
  j.s += ",\"e2e_ms\":{\"p50\":" + std::to_string(pct(0.50)) +
         ",\"p95\":" + std::to_string(pct(0.95)) +
         ",\"p99\":" + std::to_string(pct(0.99)) +
         ",\"max\":" + std::to_string(e2es.back()) + "}";
  {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    j.s += ",\"gpu\":\"" + std::string(prop.name) + "\"";
    j.s += ",\"activation_mib\":" +
           std::to_string(engine->getDeviceMemorySizeV2() / 1048576.0);
  }
  j.s += ",\"stage_busy_s\":" + std::to_string(busy_ns.load() / 1e9);
  // 逐帧 + 检测记录
  j.s += ",\"per_frame\":[";
  for (int i = 0; i < n_frames; ++i) {
    const auto& r = results[i];
    if (i) j.s += ",";
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"idx\":%d,\"n_det\":%d,\"mask_pixels\":%lld,"
                  "\"score_sum\":%.6f,\"e2e_ms\":%.3f}",
                  r.idx, r.n_det, r.mask_pixels, r.score_sum, r.e2e_ms);
    j.s += buf;
  }
  j.s += "]";
  j.s += ",\"dets\":{";
  bool first = true;
  for (int i = 0; i < n_frames && i < dump_dets; ++i) {
    if (!first) j.s += ",";
    first = false;
    j.s += "\"" + std::to_string(i) + "\":[";
    for (size_t k = 0; k < results[i].dets.size(); ++k) {
      if (k) j.s += ",";
      j.s += det_json(results[i].dets[k]);
    }
    j.s += "]";
  }
  j.s += "}}";
  if (!json_out.empty()) write_file(json_out, j.s.data(), j.s.size());
  std::printf("[done ] frames %d, wall %.3f s, fps %.2f, dets %lld, mp %lld, "
              "score_sum %s\n",
              n_frames, wall_s, n_frames / wall_s, tot_dets, tot_mp, ss);
  std::printf("[done ] e2e P50 %.2f / P95 %.2f / P99 %.2f / max %.2f ms\n",
              pct(0.50), pct(0.95), pct(0.99), e2es.back());

  // ---- 清理
  for (int i = 0; i < capacity; ++i) { /* slot 们随进程退出，显式释放显存即可 */ }
  cudaFree(d_img); cudaFree(d_in); cudaFree(d_o0); cudaFree(d_o1);
  cudaStreamDestroy(stream);
  delete ctx; delete engine; delete runtime;
  return 0;
}
