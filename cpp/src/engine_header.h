// engine_header.h —— .engine 文件里 ultralytics 头的解析（W4② 里程碑 1）
//
// 背景见 docs/cpp_infer.md §7.1：tools/build_trt.py 的 write_engine_file() 按
// ultralytics 的约定把 JSON metadata 前置写进了 engine 文件：
//
//     [4 字节小端长度 n] [n 字节 JSON metadata] [裸 TRT engine 流]
//
// 直接把整个文件喂 deserializeCudaEngine 会报
// "Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed"，
// 报错里的 844 是文件头长度字段，不是版本号 —— 这是上一轮白绕一圈换来的教训。
//
// 本文件复刻 Python 侧 induvision/trt_infer.py::decode_header() 的语义：
// 认不出头就回退到偏移 0（当裸流用），两边判据必须一致，否则会出现
// "Python 能读、C++ 读不了"或反过来的情况。
#pragma once

#include <cstddef>
#include <cstdint>

namespace induvision {

// 计算裸 TRT 流在文件字节里的起始偏移。
// 判据（与 trt_infer.py::decode_header() 逐条对应）：
//   1. 不足 4 字节 → 0；
//   2. 前 4 字节按小端读成长度 n；n == 0 或 4 + n > 文件长度 → 0；
//   3. [4, 4+n) 段不是一个 JSON object → 0；
//   4. 否则偏移 4 + n。
//
// 与 Python 版的已知偏差（有意为之，写在注释里防止后人"顺手修复"）：
// Python 版用 json.loads 做完整解析；这里只验证首尾非空白字节是 '{' / '}'。
// 理由：头由本仓库 tools/build_trt.py 写入，内容形态固定；完整 JSON 解析器
// 是 500 行的依赖，换来的判据差异只影响"恰好长得像 JSON object 的裸流"——
// 那需要裸流前 4 字节是小端合法长度、中间是平衡大括号，概率上等同
// Python 版 docstring 里"赌 engine 的原始字节几乎不可能正好构成合法 JSON"。
// 三档真 engine + 垃圾字节 + 裸流的回归用例在 tests（Python 侧）与
// tools/check_engine_header.py 两侧都钉着。
size_t engine_stream_offset(const uint8_t* data, size_t size);

// 便捷重载：返回第 4+1 ~ 4+n 字节的 JSON 文本（找不到头则返回空串）。
// 只用于打印 metadata 里的类别名，不参与判据。
char* engine_header_json(const uint8_t* data, size_t size, size_t* out_len);

}  // namespace induvision
