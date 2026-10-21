"""Focal Loss：给 YOLO-seg 的分类分支换上「难例聚焦」的损失。

为什么需要
----------
copy-paste 解决的是「稀有类实例太少」，但缺陷检测还有第二重不均衡：
**前景/背景不均衡**。一张 640x640 的图上 TAL 会给出上万个 anchor 候选，
正样本只有几个，剩下的全是易分负样本。普通 BCE 会被海量易分负样本的梯度淹没，
focal loss 用 ``(1-p_t)^gamma`` 把易分样本的权重压下去，让梯度集中在难例上。

作用范围（重要）
----------------
只替换分类分支的 BCE（``v8DetectionLoss.bce``，seg 任务的 cls 分支继承同一个）。
分割掩码分支（``BCEDiceLoss``）与 box/DFL 分支不动——那些分支的目标分布不是
「海量易分负样本」，换 focal 没有理论依据。

两个必须知道的实现细节（都踩过）
--------------------------------
1. **criterion 有两种形态**（ultralytics 8.4.183 实测）：

   * v8 系（``yolov8s-seg`` 等）：``init_criterion()`` 返回 ``v8SegmentationLoss``，
     它自己身上就有 ``.bce``；
   * E2E 系（``yolo26n`` 等端到端免 NMS 模型）：返回 ``E2ELoss``，**自己没有 ``.bce``**，
     真正的 ``.bce`` 藏在 ``one2many`` / ``one2one`` 两个 ``v8DetectionLoss`` 里。

   所以替换必须递归找。规则是「自己有 bce 就替换、并停止继续下钻」——
   这样既能覆盖 E2E，又不会误伤更深层里同名的属性。
2. **补丁要挂在 ``on_train_start`` 回调上**：``Model.train()`` 内部会执行
   ``trainer.get_model(cfg=self.model.yaml)`` **重建**一个 nn.Module 再赋回
   ``self.model``（源码里就两行，见 ``ultralytics/engine/model.py``）。
   训练前在旧对象上打的实例补丁会被整个丢掉，必须在重建之后再打。

torch 是可选依赖：CI 不装 torch，本模块在 CI 里不会被 import；
单测用 ``pytest.importorskip("torch")`` 跳过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "focal_bce_with_logits",
    "FocalBCEWithLogitsLoss",
    "install_focal_loss",
    "patch_criterion_factory",
    "resolve_base_model",
    "FocalHandle",
]

# E2ELoss -> one2many/one2one -> v8DetectionLoss.bce，两层足够，再深没有意义
_MAX_CRITERION_DEPTH = 2


def focal_bce_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
) -> torch.Tensor:
    """逐元素 focal loss，**不做 reduction**（调用方自己 mean/sum）。

    ``gamma=0 且 alpha=None`` 时退化为标准 BCE，单测就拿这条做等价性断言。
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    if gamma == 0.0 and alpha is None:
        return bce
    p = torch.sigmoid(pred)
    p_t = target * p + (1.0 - target) * (1.0 - p)
    loss = bce * (1.0 - p_t) ** gamma
    if alpha is not None:
        loss = loss * (target * alpha + (1.0 - target) * (1.0 - alpha))
    return loss


class FocalBCEWithLogitsLoss(nn.Module):
    """与 ``nn.BCEWithLogitsLoss(reduction="none")`` 调用签名一致的 focal 版本。

    ultralytics 的 ``v8DetectionLoss`` 里 cls 分支就是 ``self.bce(pred, target)``
    逐元素算完再自己聚合，所以只要把 ``criterion.bce`` 换成这个对象即可，
    不需要改 ultralytics 一行源码。
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma 必须 >= 0，收到 {gamma}")
        if alpha is not None and not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha 应在 (0,1) 或 None，收到 {alpha}")
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_bce_with_logits(pred, target, self.gamma, self.alpha)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, alpha={self.alpha}"


def resolve_base_model(model: Any) -> Optional[Any]:
    """从 ultralytics ``YOLO`` 包装器里取出真正的 ``BaseModel``(nn.Module)。

    传进来的本来就是 BaseModel 时原样返回；取不到返回 ``None``。
    注意不能用 ``hasattr(model, "init_criterion")`` 判断——``Model.__getattr__``
    会把未知属性转发到内层 nn.Module，包装器上「看起来」什么都有。
    """
    cand = model
    for _ in range(3):
        if isinstance(cand, nn.Module) and hasattr(type(cand), "init_criterion"):
            return cand
        inner = getattr(cand, "model", None)
        if inner is None or inner is cand:
            return None
        cand = inner
    return None


def _replace_bce(
    obj: Any,
    loss_cls: nn.Module,
    *,
    depth: int = 0,
    seen: Optional[set] = None,
) -> int:
    """递归把 criterion 里的 ``bce`` 换成 focal，返回替换处数。

    规则：**自己有 ``bce`` 就替换，并停止下钻**。
    v8 系在 depth 0 命中；E2E 系在 depth 1（``one2many`` / ``one2one``）命中。
    停止下钻是为了不误伤更深层里可能同名的属性。
    """
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > _MAX_CRITERION_DEPTH:
        return 0
    seen.add(id(obj))
    if hasattr(obj, "bce"):
        obj.bce = loss_cls
        return 1
    obj_vars = getattr(obj, "__dict__", None)
    if not obj_vars:
        return 0
    n = 0
    for value in list(obj_vars.values()):
        if isinstance(value, (torch.Tensor, str, bytes, bool, int, float, complex, type(None))):
            continue
        if isinstance(value, dict):
            for sub in list(value.values()):
                if hasattr(sub, "__dict__"):
                    n += _replace_bce(sub, loss_cls, depth=depth + 1, seen=seen)
            continue
        if hasattr(value, "__dict__"):
            n += _replace_bce(value, loss_cls, depth=depth + 1, seen=seen)
    return n


def patch_criterion_factory(base_model: Any, loss_cls: nn.Module, registry: Optional[list] = None) -> int:
    """覆盖 ``base_model.init_criterion``，让它建出来的 criterion 用 focal 当 cls 损失。

    只在**实例**上覆盖，不动全局类，多个模型互不干扰。
    ``registry`` 传入 list 时记录 ``(对象, 原方法)`` 供 ``FocalHandle.undo()`` 复原。
    """
    if registry is not None and any(obj is base_model for obj, _ in registry):
        # 同一个对象别重复覆盖：第二次的 orig 会是上一次的补丁，undo 时会把 focal 又装回去
        return 0
    orig = base_model.init_criterion

    def patched():
        crit = orig()
        n = _replace_bce(crit, loss_cls)
        if n == 0:
            raise RuntimeError(
                f"{type(crit).__name__} 里找不到 bce 属性，ultralytics 版本可能变了；"
                f"focal 注入中止（宁可不训也不要静默训了个 BCE）"
            )
        return crit

    base_model.init_criterion = patched
    if registry is not None:
        registry.append((base_model, orig))
    return 1


@dataclass
class FocalHandle:
    """``install_focal_loss`` 的返回值，``undo()`` 恢复原状，方便做 A/B 消融。"""

    model: Any = None
    loss: Optional[nn.Module] = None
    patched: List[Tuple[Any, Any]] = field(default_factory=list)
    _cb_saved: Optional[Tuple[str, list]] = field(default=None, repr=False)

    def undo(self) -> None:
        if self._cb_saved is not None and self.model is not None:
            event, saved = self._cb_saved
            callbacks = getattr(self.model, "callbacks", None)
            if isinstance(callbacks, dict):
                callbacks[event] = list(saved)
            self._cb_saved = None
        for obj, orig in reversed(self.patched):  # LIFO：后打的先撤
            obj.init_criterion = orig
            # criterion 是懒建的，已经建过就要丢掉，否则 undo 之后仍在用 focal
            obj.__dict__.pop("criterion", None)
        self.patched = []

    def describe(self) -> str:
        return f"{type(self.loss).__name__}({self.loss.extra_repr()})" if self.loss is not None else "none"


def install_focal_loss(
    model: Any,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    *,
    verbose: bool = True,
) -> FocalHandle:
    """把 focal loss 注入 ultralytics 模型（detect / segment、v8 / E2E 均支持）。

    入参可以是 ``YOLO(...)`` 包装器，也可以是其内层 ``model.model``：
    传包装器时会额外注册 ``on_train_start`` 回调，因为 ``Model.train()`` 会重建
    nn.Module，训练前打的补丁会被丢掉（详见模块 docstring）。
    """
    loss_cls = FocalBCEWithLogitsLoss(gamma, alpha)
    handle = FocalHandle(model=model, loss=loss_cls)

    wrapper = model if hasattr(model, "add_callback") and hasattr(model, "callbacks") else None
    base = resolve_base_model(model)

    if wrapper is not None:
        event = "on_train_start"
        saved = list(wrapper.callbacks.get(event, []))

        def _hook(trainer: Any) -> None:
            target = getattr(trainer, "model", None) or base
            if target is None:
                raise RuntimeError("on_train_start 时拿不到 trainer.model，无法注入 focal loss")
            patch_criterion_factory(target, loss_cls, handle.patched)
            if verbose:
                print(f"[focal] on_train_start: {type(target).__name__} 分类分支 BCE -> {handle.describe()}")

        wrapper.add_callback(event, _hook)
        handle._cb_saved = (event, saved)

    if base is not None:
        # 立刻给当前这个 nn.Module 也打上：不经过 train() 的直接调用（单测 / val）也能生效
        patch_criterion_factory(base, loss_cls, handle.patched)
    elif wrapper is None:
        raise TypeError(
            f"{type(model).__name__} 既不是 ultralytics YOLO 包装器，也没有 init_criterion，无法注入 focal loss"
        )

    return handle