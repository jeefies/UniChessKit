"""按路径加载引擎插件（composition root 的一部分）。

远端 conda 环境与其他项目共用、约定不改动，所以不用 entry point / pip 安装，
而是 ``EngineSpec(root, factory="包.模块:函数", kwargs)``：把 root 放进 sys.path 再导入。

吸取 R ``_import_isolated`` 的教训：T 和 R 以前都有顶层包 ``core`` / ``model`` / ``search``，
同进程加载时后来者拿到先到者的模块，**无声地算错**（64 token 编码当 19 平面用）。
T/R 改名为 unichess_t / unichess_r 后不再冲突；这里再加一道保险：导入后核对模块文件
确实位于 root 之下，并拒绝已被别处占用的同名顶层包。

工厂约定：``factory(**kwargs)`` 加载一次模型并返回 ``PlayerFactory``——
一个无参可调用对象，每次调用返回一个新的 Player（每局一个）。
工厂所在模块可以声明 ``KIT_SPI_VERSION``，与 kit 的 SPI_VERSION 不符时拒绝加载。
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import SPI_VERSION


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineSpec:
    factory: str                       # "unichess_r.kit_adapter:make_player_factory"
    root: Optional[str] = None         # 引擎仓库根目录（放进 sys.path）
    kwargs: dict = field(default_factory=dict)
    label: str = ""                    # 结果里显示的名字；空则用 factory
    # 只影响怎么跑、不影响结果的参数（如本次运行的临时服务目录）：同样传给工厂，
    # 但不进 identity()，因而不进结果文件的配置哈希（续跑时可以不同）
    runtime: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSpec":
        unknown = set(d) - {"factory", "root", "kwargs", "label", "runtime"}
        if unknown:
            raise RegistryError(f"EngineSpec 有未知字段 {sorted(unknown)}")
        return cls(factory=d["factory"], root=d.get("root"), kwargs=dict(d.get("kwargs", {})),
                   label=d.get("label", ""), runtime=dict(d.get("runtime", {})))

    def identity(self) -> dict:
        """决定结果的部分（进配置哈希）。"""
        return {"factory": self.factory, "root": self.root, "kwargs": self.kwargs,
                "label": self.label}

    def to_dict(self) -> dict:
        d = self.identity()
        if self.runtime:
            d["runtime"] = self.runtime
        return d

    @property
    def name(self) -> str:
        return self.label or self.factory


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_object(target: str, root: Optional[str] = None) -> Any:
    if ":" not in target:
        raise RegistryError(f"工厂 {target!r} 应写成 '包.模块:函数'")
    mod_name, attr = target.split(":", 1)
    top = mod_name.split(".", 1)[0]
    root_path = Path(root).expanduser() if root else None
    if root_path is not None:
        if not root_path.is_dir():
            raise RegistryError(f"引擎根目录不存在：{root_path}")
        existing = sys.modules.get(top)
        if existing is not None:
            f = getattr(existing, "__file__", None)
            if f is None or not _under(Path(f), root_path):
                raise RegistryError(f"顶层包 {top!r} 已从 {f} 加载，与 {root_path} 冲突")
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))
    module = importlib.import_module(mod_name)
    f = getattr(module, "__file__", None)
    if root_path is not None and (f is None or not _under(Path(f), root_path)):
        raise RegistryError(f"{mod_name} 实际加载自 {f}，不在引擎根目录 {root_path} 下")
    spi = getattr(module, "KIT_SPI_VERSION", SPI_VERSION)
    if spi != SPI_VERSION:
        raise RegistryError(f"{mod_name} 声明 KIT_SPI_VERSION={spi}，kit 为 {SPI_VERSION}")
    try:
        return getattr(module, attr)
    except AttributeError:
        raise RegistryError(f"{mod_name} 没有 {attr}") from None


def build_player_factory(spec: EngineSpec):
    factory = load_object(spec.factory, spec.root)
    overlap = set(spec.kwargs) & set(spec.runtime)
    if overlap:
        raise RegistryError(f"kwargs 与 runtime 有重复参数 {sorted(overlap)}")
    player_factory = factory(**spec.kwargs, **spec.runtime)
    if not callable(player_factory):
        raise RegistryError(f"{spec.factory} 应返回可调用的 PlayerFactory")
    return player_factory
