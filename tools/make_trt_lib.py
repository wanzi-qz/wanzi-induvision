# -*- coding: utf-8 -*-
"""S5(b)：从 TensorRT 的 DLL 合成 MSVC import lib（全离线，cpp_infer.md §2 S5）。

前提：必须在 **x64 Native Tools Command Prompt for VS 2022** 里跑（需要
dumpbin.exe 和 lib.exe 在 PATH 里），或用 --vsdev 参数自动定位 vcvars64.bat。

原理：
  dumpbin /exports nvinfer_10.dll   →  导出符号表（第 4 列是符号名）
  写 .def：LIBRARY "nvinfer_10.dll" + EXPORTS 列表
  lib /def:... /machine:x64 /out:nvinfer.lib
  →  得到 import lib，运行时加载器按 LIBRARY 名找 pip 装的 DLL，不用改名。

先拿小的 nvinfer_plugin_10.dll（41 MB）试手，通了再对 nvinfer_10.dll（358 MB）。
nvonnxparser_10.dll 不需要——engine 已在 W3b 用 Python 建好，C++ 侧只反序列化。

用法：
  python tools/make_trt_lib.py --dll-dir <tensorrt_libs 目录> --out-dir cpp/trt_lib
  python tools/make_trt_lib.py --dll-dir ... --out-dir ... --vsdev   # 自动 vcvars
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_DEFAULT = REPO_ROOT / "cpp" / "trt_lib"

#: 要合成的 DLL（顺序就是执行顺序：先小的试手）。nvonnxparser 不在清单里，见 docstring。
TARGETS = ["nvinfer_plugin_10.dll", "nvinfer_10.dll"]

#: dumpbin /exports 输出里一行导出的形状：
#    ordinal hint RVA        name
#         1A   2C 00001000   fooBar
_ROW = re.compile(r"^\s*[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$")


def parse_exports(dumpbin_out: str) -> list[str]:
    names: list[str] = []
    in_section = False
    for line in dumpbin_out.splitlines():
        if "ordinal" in line.lower() and "hint" in line.lower():
            in_section = True
            continue
        if in_section:
            m = _ROW.match(line)
            if m:
                names.append(m.group(1))
            elif line.strip() == "" and names:
                break  # 空行 = 导出段结束
    if not names:
        raise RuntimeError("dumpbin 输出里没解析到任何符号，dumpbin 版本/输出格式变了？")
    return names


def make_one(dll: Path, out_dir: Path, vsdev: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dll.stem                     # nvinfer_10 / nvinfer_plugin_10
    # TRT 的头里 __declspec(dllimport) 的是 nvinfer.dll / nvinfer_plugin.dll
    # 这个链接名（不带 _10），所以 import lib 取短名，链接 -lnvinfer 才找得到；
    # 运行期真正的 DLL 名由 .def 的 LIBRARY 行决定（= 原文件名，含 _10）。
    lib_name = re.sub(r"_10$", "", stem) + ".lib"
    def_path = out_dir / (stem + ".def")

    dumps = _run_vs(["dumpbin", "/exports", str(dll)], vsdev)
    if dumps.returncode != 0:
        raise RuntimeError(f"dumpbin 失败（rc={dumps.returncode}）：{dumps.stderr[:300]}")
    names = parse_exports(dumps.stdout)
    print(f"[def  ] {dll.name}: {len(names)} 个导出符号 -> {def_path.name}")
    with open(def_path, "w", encoding="ascii", newline="\n") as f:
        f.write(f'LIBRARY "{dll.name}"\nEXPORTS\n')
        f.writelines(f"    {n}\n" for n in names)

    r = _run_vs(["lib", f"/def:{def_path}", "/machine:x64",
                 f"/out:{out_dir / lib_name}"], vsdev)
    if r.returncode != 0:
        raise RuntimeError(f"lib.exe 失败（rc={r.returncode}）：{(r.stderr or r.stdout)[-400:]}")
    lib = out_dir / lib_name
    if not lib.is_file():
        raise RuntimeError(f"lib.exe 说成功但 {lib} 不存在")
    print(f"[lib  ] {lib}  {lib.stat().st_size / 1024:.1f} KiB")
    return lib


def _run_vs(cmd: list[str], vsdev: bool):
    """在 VS 环境里跑一条命令；--vsdev 时先挂 vcvars64.bat。

    两个坑（都踩过）：
    * vcvars64.bat 的 banner 是 GBK，text=True 默认 utf-8 会在读取线程崩；
    * subprocess 列表参数经 list2cmdline 会给整条命令再套引号，cmd /c 收到
      转义过的引号后把 vcvars 路径当坏命令。所以写成临时 .bat 执行，绕开引号地狱。
    """
    if not vsdev:
        return subprocess.run(cmd, capture_output=True, text=True,
                              encoding="gbk", errors="replace")
    vcvars = _find_vcvars()
    bat = OUT_DIR_DEFAULT.parent / "_tmp_runvs.bat"
    with open(bat, "w", encoding="ascii", newline="\r\n") as f:
        f.write(f'@echo off\r\ncall "{vcvars}" x64 >nul\r\n')
        f.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\r\n")
    return subprocess.run(["cmd", "/c", str(bat)], capture_output=True, text=True,
                          encoding="gbk", errors="replace")


def _find_vcvars() -> str:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.is_file():
        raise RuntimeError("找不到 vswhere，先装 VS Build Tools（cpp_infer.md §2 S2）")
    out = subprocess.run([str(vswhere), "-products", "*", "-latest", "-property",
                          "installationPath"], capture_output=True, text=True).stdout.strip()
    vcvars = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError(f"vcvars64.bat 不存在：{vcvars}（VC.Tools.x86.x64 组件没装上？）")
    return str(vcvars)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 TRT DLL 合成 MSVC import lib（S5(b)）")
    ap.add_argument("--dll-dir", required=True,
                    help="tensorrt_libs 目录（含 nvinfer_10.dll 等 11 个 DLL）")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "cpp" / "trt_lib"))
    ap.add_argument("--vsdev", action="store_true",
                    help="不在 VS 命令行里跑时自动挂 vcvars64.bat")
    ap.add_argument("--only", choices=TARGETS, default=None,
                    help="只合成某一个 DLL（默认按小→大顺序全做）")
    args = ap.parse_args(argv)

    dll_dir = Path(args.dll_dir)
    if not dll_dir.is_dir():
        print(f"[err  ] 目录不存在：{dll_dir}")
        return 2
    for exe in ("dumpbin", "lib"):
        if args.vsdev:
            break  # 由 vcvars 提供
        if shutil.which(exe) is None:
            print(f"[err  ] {exe}.exe 不在 PATH：请改在 x64 Native Tools 命令行里跑，"
                  f"或加 --vsdev")
            return 2
    out_dir = Path(args.out_dir)

    targets = [args.only] if args.only else TARGETS
    for name in targets:
        dll = dll_dir / name
        if not dll.is_file():
            print(f"[err  ] {dll} 不存在")
            return 2
        try:
            make_one(dll, out_dir, args.vsdev)
        except RuntimeError as e:
            print(f"[err  ] {e}")
            return 1
    print(f"[done ] import lib 就绪：{out_dir}（CMake 传 -DTENSORRT_LIBRARY_DIR={out_dir}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
