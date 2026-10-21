"""focal loss 注入的单测。

CI 不装 torch，模块级 importorskip 会让整个文件跳过；
带真实 ultralytics 权重的两个用例只在本地跑（权重不入库）。

覆盖顺序：数学 -> 递归找 bce -> 工厂补丁 -> install/undo -> 真模型。
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from induvision.focal import (  # noqa: E402
    FocalBCEWithLogitsLoss,
    _replace_bce,
    focal_bce_with_logits,
    install_focal_loss,
    patch_criterion_factory,
    resolve_base_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V8_SEG_WEIGHTS = REPO_ROOT / "weights" / "yolov8s-seg.pt"   # v8 系：criterion 自带 bce
E2E_WEIGHTS = REPO_ROOT / "yolo26n.pt"                       # E2E 系：bce 藏在 one2many/one2one


# --------------------------------------------------------------------------
# 数学
# --------------------------------------------------------------------------


def test_gamma0_without_alpha_is_plain_bce():
    torch.manual_seed(0)
    pred = torch.randn(4, 7)
    target = (torch.rand(4, 7) > 0.5).float()
    got = focal_bce_with_logits(pred, target, gamma=0.0, alpha=None)
    want = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-6)


def test_focal_downweights_easy_examples_only():
    pred = torch.tensor([[6.0, -6.0, 0.0]])   # 两个易分 + 一个难分
    target = torch.tensor([[1.0, 0.0, 1.0]])
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    foc = focal_bce_with_logits(pred, target, gamma=2.0, alpha=None)
    assert (foc <= bce + 1e-6).all()
    easy = (foc[0, :2] / bce[0, :2]).max().item()
    hard = (foc[0, 2] / bce[0, 2]).item()
    assert easy < 1e-3 < hard <= 1.0, (easy, hard)


def test_alpha_reweights_branches():
    pred = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 0.0]])
    no_a = focal_bce_with_logits(pred, target, gamma=2.0, alpha=None)
    with_a = focal_bce_with_logits(pred, target, gamma=2.0, alpha=0.25)
    assert torch.allclose(with_a[0, 0], no_a[0, 0] * 0.25)
    assert torch.allclose(with_a[0, 1], no_a[0, 1] * 0.75)


def test_module_is_drop_in_for_bce_with_logits_none():
    """签名必须跟 nn.BCEWithLogitsLoss(reduction="none") 一致，否则换进 ultralytics 就炸。"""
    torch.manual_seed(1)
    pred, target = torch.randn(3, 5), (torch.rand(3, 5) > 0.5).float()
    focal = FocalBCEWithLogitsLoss(gamma=0.0, alpha=None)
    ref = nn.BCEWithLogitsLoss(reduction="none")
    assert torch.allclose(focal(pred, target), ref(pred, target), atol=1e-6)
    assert "gamma=0.0" in repr(focal) and "alpha=None" in repr(focal)


def test_gradient_is_finite_and_nonzero():
    pred = torch.randn(3, 5, requires_grad=True)
    target = (torch.rand(3, 5) > 0.5).float()
    FocalBCEWithLogitsLoss()(pred, target).mean().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


@pytest.mark.parametrize("gamma,alpha", [(-1.0, 0.25), (2.0, 0.0), (2.0, 1.0), (2.0, -0.5)])
def test_invalid_params_raise(gamma, alpha):
    with pytest.raises(ValueError):
        FocalBCEWithLogitsLoss(gamma, alpha)


# --------------------------------------------------------------------------
# 递归替换 bce
# --------------------------------------------------------------------------


class _HasBce:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()


class _NoBce:
    pass


class _FakeE2E:
    """模仿 E2ELoss：自己没有 bce，真正的 bce 在 one2many / one2one 两个子损失里。"""

    def __init__(self):
        self.one2many = _HasBce()
        self.one2one = _HasBce()


class _L3:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()


class _L2:
    def __init__(self):
        self.c = _L3()


class _L1:
    def __init__(self):
        self.b = _L2()


class _L0:
    def __init__(self):
        self.a = _L1()


class _BothLevels:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()
        self.child = _HasBce()


class _DictHolder:
    def __init__(self):
        self.heads = {"a": _HasBce()}


def test_replace_bce_covers_v8_and_e2e_shapes():
    focal = FocalBCEWithLogitsLoss()
    v8 = _HasBce()
    assert _replace_bce(v8, focal) == 1 and v8.bce is focal

    e2e = _FakeE2E()
    assert _replace_bce(e2e, focal) == 2
    assert e2e.one2many.bce is focal and e2e.one2one.bce is focal


def test_replace_bce_stops_descending_after_hit():
    """命中就停，不误伤更深层里同名的属性。"""
    obj = _BothLevels()
    focal = FocalBCEWithLogitsLoss()
    assert _replace_bce(obj, focal) == 1
    assert obj.bce is focal
    assert obj.child.bce is not focal


def test_replace_bce_respects_depth_limit():
    assert _replace_bce(_L0(), FocalBCEWithLogitsLoss()) == 0


def test_replace_bce_descends_into_dict_values():
    h = _DictHolder()
    focal = FocalBCEWithLogitsLoss()
    assert _replace_bce(h, focal) == 1
    assert h.heads["a"].bce is focal


def test_replace_bce_survives_reference_cycles():
    a, b = _NoBce(), _NoBce()
    a.other, b.other = b, a
    a.self_ref = a
    assert _replace_bce(a, FocalBCEWithLogitsLoss()) == 0


# --------------------------------------------------------------------------
# criterion 工厂补丁
# --------------------------------------------------------------------------


class _FakeCriterion:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()


class _FakeBase(nn.Module):
    """模仿 BaseModel：init_criterion() 造 criterion。"""

    def init_criterion(self):
        return _FakeCriterion()


class _NoBceCriterion:
    pass


class _BrokenBase(nn.Module):
    def init_criterion(self):
        return _NoBceCriterion()


def test_patch_factory_swaps_bce_and_dedupes():
    m = _FakeBase()
    reg = []
    assert patch_criterion_factory(m, FocalBCEWithLogitsLoss(), reg) == 1
    assert isinstance(m.init_criterion().bce, FocalBCEWithLogitsLoss)
    # 同一个对象重复注册要返回 0，否则 undo 时 orig 是上一次的补丁，会把 focal 又装回去
    assert patch_criterion_factory(m, FocalBCEWithLogitsLoss(), reg) == 0
    assert len(reg) == 1
    obj, orig = reg[0]
    obj.init_criterion = orig
    assert not isinstance(m.init_criterion().bce, FocalBCEWithLogitsLoss)


def test_patch_factory_fails_loudly_when_no_bce_found():
    """宁可报错也不要静默训了个 BCE。"""
    m = _BrokenBase()
    patch_criterion_factory(m, FocalBCEWithLogitsLoss())
    with pytest.raises(RuntimeError, match="找不到 bce"):
        m.init_criterion()


def test_resolve_base_model_unwraps_layers():
    class _Wrapper:
        def __init__(self, inner):
            self.model = inner

    base = _FakeBase()
    assert resolve_base_model(base) is base
    assert resolve_base_model(_Wrapper(base)) is base
    assert resolve_base_model(_Wrapper(_Wrapper(base))) is base
    assert resolve_base_model(object()) is None
    assert resolve_base_model(_Wrapper(None)) is None


def test_resolve_base_model_is_not_fooled_by_getattr_forwarding():
    """ultralytics 的 Model.__getattr__ 会把未知属性转发给内层 nn.Module，
    所以不能用 hasattr(model, "init_criterion") 判断——那会让包装器「看起来」就是 BaseModel。
    """

    class _Forwarding:
        def __init__(self, inner):
            self.model = inner

        def __getattr__(self, item):
            return getattr(self.model, item)

    base = _FakeBase()
    w = _Forwarding(base)
    assert hasattr(w, "init_criterion")      # 转发造成的假象
    assert not isinstance(w, nn.Module)
    assert resolve_base_model(w) is base     # 但解析出来的必须是内层


# --------------------------------------------------------------------------
# install / undo
# --------------------------------------------------------------------------


class _FakeWrapper:
    """模仿 ultralytics YOLO 包装器：callbacks 字典 + add_callback + 内层 .model。"""

    def __init__(self, base=None):
        self.callbacks = {"on_train_start": []}
        if base is not None:
            self.model = base

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)


class _FakeTrainer:
    def __init__(self, model):
        self.model = model


def _ensure_train_args(base):
    """从 .pt 直接加载时 model.args 是 checkpoint 里存的 dict，
    而 v8SegmentationLoss.__init__ 要读 model.args.overlap_mask。
    训练器在 train() 里会执行 model.args = trainer.args，这里补上同一步。
    """
    from ultralytics.cfg import get_cfg

    if not hasattr(getattr(base, "args", None), "overlap_mask"):
        base.args = get_cfg()
    return base.args


def test_install_patches_current_module_and_registers_hook():
    base = _FakeBase()
    wrapper = _FakeWrapper(base)
    before = list(wrapper.callbacks["on_train_start"])
    h = install_focal_loss(wrapper, gamma=2.0, alpha=0.25, verbose=False)

    # 不经过 train() 的直接调用（单测 / val）也要生效
    assert isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)
    assert h.describe().startswith("FocalBCEWithLogitsLoss(gamma=2.0")
    cbs = wrapper.callbacks["on_train_start"]
    assert len(cbs) == len(before) + 1

    # Model.train() 内部会重建 nn.Module，钩子必须能给新对象打上补丁
    rebuilt = _FakeBase()
    cbs[-1](_FakeTrainer(rebuilt))
    assert isinstance(rebuilt.init_criterion().bce, FocalBCEWithLogitsLoss)
    assert rebuilt.init_criterion().bce is base.init_criterion().bce  # 共享同一个 focal 对象

    base.criterion = base.init_criterion()  # 懒建缓存
    h.undo()
    assert wrapper.callbacks["on_train_start"] == before
    assert "criterion" not in base.__dict__, "undo 后必须丢掉懒建的 criterion，否则还在用 focal"
    assert not isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)
    assert not isinstance(rebuilt.init_criterion().bce, FocalBCEWithLogitsLoss)


def test_install_on_bare_base_model_has_no_callback():
    base = _FakeBase()
    h = install_focal_loss(base, verbose=False)
    assert h._cb_saved is None
    assert isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)
    h.undo()
    assert not isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)


def test_nested_installs_undo_in_lifo_order():
    base = _FakeBase()
    h1 = install_focal_loss(base, gamma=1.0, verbose=False)
    h2 = install_focal_loss(base, gamma=3.0, verbose=False)
    assert base.init_criterion().bce.gamma == 3.0
    h2.undo()
    assert base.init_criterion().bce.gamma == 1.0  # 退回第一层，不是直接裸奔
    h1.undo()
    assert not isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)


def test_install_rejects_unusable_object():
    with pytest.raises(TypeError, match="无法注入 focal loss"):
        install_focal_loss(object())


def test_hook_raises_when_trainer_model_is_missing():
    wrapper = _FakeWrapper()  # 没有 .model，resolve 不到 BaseModel
    h = install_focal_loss(wrapper, verbose=False)
    assert h.patched == []
    with pytest.raises(RuntimeError, match="拿不到 trainer.model"):
        wrapper.callbacks["on_train_start"][0](object())


# --------------------------------------------------------------------------
# 真模型（本地权重，CI 跳过）
# --------------------------------------------------------------------------


@pytest.mark.skipif(not V8_SEG_WEIGHTS.is_file(), reason=f"本地没有 {V8_SEG_WEIGHTS.name}（权重不入库）")
def test_real_v8_segment_model():
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    m = YOLO(str(V8_SEG_WEIGHTS))
    base = resolve_base_model(m)
    assert base is not None and isinstance(base, nn.Module)
    before = list(m.callbacks["on_train_start"])

    h = install_focal_loss(m, gamma=2.0, alpha=0.25, verbose=False)
    _ensure_train_args(base)
    crit = base.init_criterion()          # v8SegmentationLoss：bce 就在它自己身上
    assert isinstance(crit.bce, FocalBCEWithLogitsLoss)
    assert crit.bce.gamma == 2.0 and crit.bce.alpha == 0.25
    # 掩码分支不能被动：那是 BCEDiceLoss，目标分布不是「海量易分负样本」
    from ultralytics.utils.loss import BCEDiceLoss

    assert isinstance(crit.bcedice_loss, BCEDiceLoss)
    assert len(m.callbacks["on_train_start"]) == len(before) + 1

    h.undo()
    assert m.callbacks["on_train_start"] == before
    assert not isinstance(base.init_criterion().bce, FocalBCEWithLogitsLoss)


@pytest.mark.skipif(not E2E_WEIGHTS.is_file(), reason=f"本地没有 {E2E_WEIGHTS.name}（权重不入库）")
def test_real_e2e_model_bce_is_nested():
    """YOLO26 这类端到端模型返回 E2ELoss，自己没有 bce，得递归进 one2many/one2one。"""
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    m = YOLO(str(E2E_WEIGHTS))
    base = resolve_base_model(m)
    assert base is not None

    h = install_focal_loss(m, gamma=2.0, alpha=0.25, verbose=False)
    crit = base.init_criterion()
    assert not hasattr(crit, "bce"), "E2ELoss 结构变了，递归深度可能要跟着调"
    assert crit.one2many.bce is h.loss and crit.one2one.bce is h.loss

    h.undo()
    crit2 = base.init_criterion()
    assert crit2.one2many.bce is not h.loss
    assert not isinstance(crit2.one2many.bce, FocalBCEWithLogitsLoss)
    assert not isinstance(crit2.one2one.bce, FocalBCEWithLogitsLoss)