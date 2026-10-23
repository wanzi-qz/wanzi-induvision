// infer_one.cpp —— W4② 里程碑 1：engine 反序列化 + 单帧推理 + 输出落盘对拍
//
// 口径（对齐 docs/cpp_infer.md §7 的契约，全部实测过，别猜）：
//   images  INPUT  (1,3,640,640)  FLOAT  4,915,200 B
//   output0 OUTPUT (1,51,8400)    FLOAT  1,713,600 B
//   output1 OUTPUT (1,32,160,160) FLOAT  3,276,800 B
// shape 全静态 → 不需要 setInputShape / profile 查询，每帧
// setTensorAddress × 3 + enqueueV3(stream)。
//
// 不依赖 OpenCV：输入是 Python 侧 preprocess() 导出的 (1,3,640,640) float32 bin，
// 输出落盘后由 Python 侧与 TrtRunner 的结果逐元素比对（tools/check_cpp_infer.py）。
// 前处理上 GPU 是里程碑 2 的事，这一步先把"反序列化 + 推理 + 数值对拍"打通。
//
// 用法：
//   infer_one --engine weights\baseline15_trt_fp16.engine --input in.bin
//             [--out0 o0.bin --out1 o1.bin] [--iters N] [--warmup W]
//   infer_one --engine xxx.engine --dump-header   （只打印头信息，不占 GPU）
#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "engine_header.h"

namespace {

// TRT 所有 API 都要一个 logger。级别取 WARNING：INFO 会被 engine 构建刷屏，
// ERROR 之上的问题不用等它说。
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[trt  ] %s\n", msg);
    }
  }
};

[[noreturn]] void die(const std::string& what) {
  std::fprintf(stderr, "[err  ] %s\n", what.c_str());
  std::exit(2);
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("打不开文件：" + path);
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(buf.data()), n);
  if (!f) die("读不完整：" + path);
  return buf;
}

std::vector<float> read_bin_f32(const std::string& path, size_t expect_elems) {
  const auto raw = read_file(path);
  if (raw.size() != expect_elems * sizeof(float)) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "输入字节数不符：%s 是 %zu B，期望 %zu B",
                  path.c_str(), raw.size(), expect_elems * sizeof(float));
    die(msg);
  }
  std::vector<float> v(expect_elems);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

void write_bin_f32(const std::string& path, const float* data, size_t elems) {
  std::ofstream f(path, std::ios::binary);
  if (!f) die("写不了文件：" + path);
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(elems * sizeof(float)));
}

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    const cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                           \
      die(std::string("CUDA 错误 ") + cudaGetErrorName(err__) + " @ " #call \
          + ": " + cudaGetErrorString(err__));                            \
    }                                                                     \
  } while (0)

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path, input_path, out0_path, out1_path;
  int iters = 1, warmup = 3;
  bool dump_header = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) die(std::string(name) + " 缺参数值");
      return argv[++i];
    };
    if (a == "--engine") engine_path = next("--engine");
    else if (a == "--input") input_path = next("--input");
    else if (a == "--out0") out0_path = next("--out0");
    else if (a == "--out1") out1_path = next("--out1");
    else if (a == "--iters") iters = std::stoi(next("--iters"));
    else if (a == "--warmup") warmup = std::stoi(next("--warmup"));
    else if (a == "--dump-header") dump_header = true;
    else die("未知参数：" + a);
  }
  if (engine_path.empty()) die("缺 --engine");

  // ---- 1. 读文件 + 剥头（§7.1：不是裸流！）
  const auto bytes = read_file(engine_path);
  const size_t off = induvision::engine_stream_offset(bytes.data(), bytes.size());
  std::printf("[head ] 文件 %zu B，头 %zu B，裸流偏移 %zu\n",
              bytes.size(), off == 0 ? 0 : off - 4, off);
  if (dump_header) {
    size_t jlen = 0;
    if (char* js = induvision::engine_header_json(bytes.data(), bytes.size(), &jlen)) {
      std::fwrite(js, 1, jlen, stdout);
      std::fputc('\n', stdout);
    } else {
      std::puts("[head ] 未检测到 ultralytics 头，按裸流处理");
    }
    return 0;
  }
  if (off == 0) {
    std::puts("[warn ] 未检测到 ultralytics 头，按裸流处理");
  }

  // ---- 2. 反序列化
  static Logger logger;
  auto* runtime = nvinfer1::createInferRuntime(logger);
  if (!runtime) die("createInferRuntime 失败");
  auto* engine = runtime->deserializeCudaEngine(bytes.data() + off,
                                                bytes.size() - off);
  if (!engine) die("deserializeCudaEngine 失败（八成是没剥头，见 cpp_infer.md §7.1）");
  std::printf("[engine] 反序列化成功\n");

  // ---- 3. I/O 自检：不写死 tensor 名，遍历出来对契约
  struct IoBuf { std::string name; bool is_input; size_t bytes; void* dev; };
  std::vector<IoBuf> ios;
  const int32_t nb = engine->getNbIOTensors();
  size_t total_in = 0, total_out = 0;
  for (int32_t i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    const auto mode = engine->getTensorIOMode(name);
    const auto shape = engine->getTensorShape(name);
    size_t elems = 1;
    bool has_dynamic = false;
    for (int d = 0; d < shape.nbDims; ++d) {
      if (shape.d[d] < 0) has_dynamic = true;
      else elems *= static_cast<size_t>(shape.d[d]);
    }
    if (has_dynamic) die(std::string(name) + " 有动态维，契约（§7.2 全静态）变了，停下检查");
    if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
      die(std::string(name) + " 不是 FLOAT，契约变了");
    const size_t nbytes = elems * sizeof(float);
    const bool is_input = mode == nvinfer1::TensorIOMode::kINPUT;
    void* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, nbytes));
    ios.push_back({name, is_input, nbytes, dev});
    (is_input ? total_in : total_out) += nbytes;
    std::string dims;
    for (int d = 0; d < shape.nbDims; ++d)
      dims += (d ? "x" : "") + std::to_string(shape.d[d]);
    std::printf("[io   ] %-8s %-22s %-16s %10zu B\n",
                is_input ? "INPUT" : "OUTPUT", name, dims.c_str(), nbytes);
  }
  std::printf("[io   ] 输入合计 %zu B（期望 4,915,200），输出合计 %zu B（期望 4,990,400）\n",
              total_in, total_out);

  // ---- 4. context + 绑地址（全静态，每帧只需这三步 + enqueue）
  auto* ctx = engine->createExecutionContext();
  if (!ctx) die("createExecutionContext 失败");
  for (const auto& b : ios) {
    if (!ctx->setTensorAddress(b.name.c_str(), b.dev))
      die("setTensorAddress 失败：" + b.name);
  }
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const IoBuf* in = nullptr;
  std::vector<const IoBuf*> outs;
  for (const auto& b : ios) {
    if (b.is_input) in = &b;
    else outs.push_back(&b);
  }
  if (!in || outs.size() != 2) die("I/O 数量与契约不符：期望 1 入 2 出");

  // ---- 5. 输入 H2D（一次性；对拍口径 = 单帧推理，不含读图与前处理）
  std::vector<float> host_in(in->bytes / sizeof(float));
  if (!input_path.empty()) {
    host_in = read_bin_f32(input_path, host_in.size());
  } else {
    std::puts("[warn ] 未给 --input，用全零输入（数值对拍必须给真图导出的 bin）");
  }
  CUDA_CHECK(cudaMemcpyAsync(in->dev, host_in.data(), in->bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // ---- 6. 推理 + 输出落盘
  std::vector<std::vector<float>> host_out(outs.size());
  for (size_t k = 0; k < outs.size(); ++k)
    host_out[k].resize(outs[k]->bytes / sizeof(float));
  auto run_once = [&]() {
    CUDA_CHECK(cudaMemcpyAsync(in->dev, host_in.data(), in->bytes,
                               cudaMemcpyHostToDevice, stream));
    if (!ctx->enqueueV3(stream)) die("enqueueV3 失败");
    for (size_t k = 0; k < outs.size(); ++k)
      CUDA_CHECK(cudaMemcpyAsync(host_out[k].data(), outs[k]->dev, outs[k]->bytes,
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 每帧只 synchronize 这一次
  };
  for (int i = 0; i < warmup; ++i) run_once();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) run_once();
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  std::printf("[perf ] %d iter，H2D+enqueue+D2H+sync 均值 %.3f ms/帧\n", iters, ms);

  if (!out0_path.empty()) write_bin_f32(out0_path, host_out[0].data(), host_out[0].size());
  if (!out1_path.empty()) write_bin_f32(out1_path, host_out[1].data(), host_out[1].size());
  if (!out0_path.empty() || !out1_path.empty())
    std::printf("[out  ] 输出已落盘（out0 %zu B / out1 %zu B），交给 Python 侧逐元素比对\n",
                host_out[0].size() * 4, host_out[1].size() * 4);

  // ---- 7. 清理（顺序：context/stream/engine/runtime；失败路径不逐一管，进程退出兜底）
  CUDA_CHECK(cudaStreamDestroy(stream));
  for (const auto& b : ios) CUDA_CHECK(cudaFree(b.dev));
  delete ctx;
  delete engine;
  delete runtime;
  return 0;
}
