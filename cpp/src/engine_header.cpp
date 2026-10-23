// engine_header.cpp —— 判据见同名头文件的注释
#include "engine_header.h"

#include <cctype>

namespace induvision {

static bool looks_like_json_object(const uint8_t* p, size_t n) {
    // 首个非空白字节 == '{'，最后一个非空白字节 == '}'
    size_t b = 0;
    while (b < n && std::isspace(static_cast<unsigned char>(p[b]))) ++b;
    if (b >= n || p[b] != '{') return false;
    size_t e = n;
    while (e > b && std::isspace(static_cast<unsigned char>(p[e - 1]))) --e;
    return e > b && p[e - 1] == '}';
}

size_t engine_stream_offset(const uint8_t* data, size_t size) {
    if (size < 4) return 0;                       // 判据 1：不足 4 字节
    const uint32_t n = static_cast<uint32_t>(data[0]) |
                       (static_cast<uint32_t>(data[1]) << 8) |
                       (static_cast<uint32_t>(data[2]) << 16) |
                       (static_cast<uint32_t>(data[3]) << 24);
    if (n == 0 || n > size - 4) return 0;         // 判据 2：长度不合法
    if (!looks_like_json_object(data + 4, n)) return 0;  // 判据 3：不是 JSON object
    return 4 + static_cast<size_t>(n);            // 判据 4
}

char* engine_header_json(const uint8_t* data, size_t size, size_t* out_len) {
    const size_t off = engine_stream_offset(data, size);
    if (out_len) *out_len = 0;
    if (off == 0) return nullptr;
    if (out_len) *out_len = off - 4;
    // 调用方持有 data 的生命周期，这里返回可写指针只是为了省一次拷贝；
    // 只读用途，别改内容。
    return reinterpret_cast<char*>(const_cast<uint8_t*>(data + 4));
}

}  // namespace induvision
