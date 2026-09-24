"""C++ PUCT 内核（``_native/puct_native.cpp``）的加载器：首次使用时用 g++ 编译，按源码哈希缓存。

- 经 ctypes 调用：``CDLL`` 的函数调用期间自动释放 GIL，多线程下搜索与前向可以真正重叠。
- 缓存目录：``$UNICHESS_KIT_NATIVE_CACHE``，默认 ``~/.cache/unichess/kit_native/<哈希>/``；
  先编译到临时文件再 ``os.replace``，多进程同时首次加载也安全。
- 编译失败直接抛 ``NativeBuildError``，**不静默回退 Python 实现**（T 曾因静默回退白跑一轮，见 T 5f81219）。
- 必须 ``-ffp-contract=off``：编译器把乘加合并成 FMA 会少一次舍入，与 numpy 不再逐位一致。
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

ABI_VERSION = 1
SOURCE = Path(__file__).with_name("_native") / "puct_native.cpp"
CXX_FLAGS = ["-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC",
             "-fvisibility=hidden"]

_lock = threading.Lock()
_lib = None


class NativeBuildError(RuntimeError):
    pass


def _cache_dir(digest: str) -> Path:
    root = os.environ.get("UNICHESS_KIT_NATIVE_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "unichess" / "kit_native"
    return base / digest


def build(force: bool = False) -> Path:
    """编译（或命中缓存）并返回 .so 路径。"""
    cxx = os.environ.get("CXX", "g++")
    src = SOURCE.read_bytes()
    digest = hashlib.sha256(src + " ".join([cxx] + CXX_FLAGS).encode()).hexdigest()[:16]
    out_dir = _cache_dir(digest)
    so = out_dir / "puct_native.so"
    if so.exists() and not force:
        return so
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".build-", suffix=".so", dir=out_dir)
    os.close(fd)
    try:
        proc = subprocess.run([cxx, *CXX_FLAGS, "-o", tmp, str(SOURCE)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise NativeBuildError(f"编译 C++ PUCT 失败（{cxx}）：\n{proc.stderr[-4000:]}")
        os.replace(tmp, so)
    except FileNotFoundError as exc:
        raise NativeBuildError(f"找不到 C++ 编译器 {cxx!r}（C++ PUCT 需要 g++）") from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return so


_c_void_p = ctypes.c_void_p
_f32p = ctypes.POINTER(ctypes.c_float)
_f64p = ctypes.POINTER(ctypes.c_double)
_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)
_u16p = ctypes.POINTER(ctypes.c_uint16)
_u64p = ctypes.POINTER(ctypes.c_uint64)
_intp = ctypes.POINTER(ctypes.c_int)

_SIGNATURES = {
    "kp_abi_version": (ctypes.c_int, []),
    "kp_last_error": (ctypes.c_char_p, []),
    "kp_ctx_new": (_c_void_p, [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
    "kp_ctx_free": (None, [_c_void_p]),
    "kp_set_root": (ctypes.c_int, [_c_void_p, _u64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int,
                                   ctypes.c_int, _u16p, ctypes.c_int]),
    "kp_begin": (ctypes.c_int, [_c_void_p, _c_void_p]),
    "kp_encode_root": (ctypes.c_int, [_c_void_p, _f32p]),
    "kp_expand_root": (ctypes.c_int, [_c_void_p, _f32p, _f32p, _f32p]),
    "kp_collect": (ctypes.c_int, [_c_void_p, ctypes.c_int, _f32p, _intp, _u16p, ctypes.c_int]),
    "kp_probe_answer": (ctypes.c_int, [_c_void_p, ctypes.c_int, ctypes.c_double]),
    "kp_apply": (ctypes.c_int, [_c_void_p, _f32p, _f32p, _f32p, ctypes.c_int]),
    "kp_collisions": (ctypes.c_int64, [_c_void_p]),
    "kp_node_new": (_c_void_p, []),
    "kp_node_free": (None, [_c_void_p]),
    "kp_node_state": (ctypes.c_int, [_c_void_p, _intp, _intp, _f64p, _i64p]),
    "kp_node_arrays": (None, [_c_void_p, _u16p, _f32p, _i32p, _f32p, _f32p]),
    "kp_node_set_P": (ctypes.c_int, [_c_void_p, _f32p, ctypes.c_int]),
    "kp_node_set_terminal": (None, [_c_void_p, ctypes.c_double]),
    "kp_node_child": (_c_void_p, [_c_void_p, ctypes.c_int]),
    "kp_node_advance": (_c_void_p, [_c_void_p, ctypes.c_int]),
    "kp_node_pv": (ctypes.c_int, [_c_void_p, ctypes.c_int, _u16p]),
    "kp_root_legal": (ctypes.c_int, [_c_void_p, _u16p, ctypes.c_int]),
    "kp_root_rules": (ctypes.c_int, [_c_void_p, ctypes.c_int, _f64p, _intp, _intp]),
    "kp_root_fen": (ctypes.c_int, [_c_void_p, ctypes.c_char_p, ctypes.c_int]),
    "kp_pairwise_f32": (ctypes.c_float, [_f32p, ctypes.c_int]),
    "kp_pairwise_f64": (ctypes.c_double, [_f64p, ctypes.c_int]),
}


def lib():
    """加载（必要时编译）并返回配置好签名的 ctypes 库。"""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            handle = ctypes.CDLL(str(build()))
            for name, (res, args) in _SIGNATURES.items():
                fn = getattr(handle, name)
                fn.restype = res
                fn.argtypes = args
            if handle.kp_abi_version() != ABI_VERSION:
                raise NativeBuildError("C++ PUCT 的 ABI 版本与 native.py 不符")
            _lib = handle
    return _lib


def last_error() -> str:
    msg = lib().kp_last_error()
    return msg.decode("utf-8", "replace") if msg else ""


def check(rc: int, what: str) -> int:
    if rc < 0 and rc != -1:
        raise RuntimeError(f"C++ PUCT {what} 失败：{last_error()}")
    return rc
