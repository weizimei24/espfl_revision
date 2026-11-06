"""
slim_models.py
==============
Index-level subnetworks of the shared template (VGG16 / ResNet18 from models.py).

Why this module exists
----------------------
ESPFL needs three operations on *arbitrary* channel subsets of the full template:

    1. extract  : Θ_G  ->  client subnetwork  Θ_G ⊙ m_i  (physically sliced, so
                  the client really trains / uploads only its active coordinates)
    2. scatter  : client update  ->  template coordinates Ω_i   (for SPPA)
    3. count    : ρ(M) = ||m||_0 / d  without building a model   (capacity control)

The previous implementation mixed prefix slicing, SVD expansion and per-arch
special cases, which placed retained channels at the wrong template coordinates
(VGG) and counted pruned coordinates as observed (ResNet).  Here every tensor of
the template is described once by the *channel groups* indexing its dimensions,
so all three operations are generic and provably consistent.

Channel groups
--------------
A group is a set of channels that must be kept / removed together:
    VGG16   : 'c0'..'c12' (conv outputs), 'h1','h2' (projector hidden units)
    ResNet18: 'stem', 'r1'..'r4' (residual stream of each stage, coupled at the
              addition nodes), 'b{s}{k}' (internal width of block k in stage s)
    pseudo  : 'input' (image channels), 'classes' (logits)  -- never sliced

A client structure is an `IndexSet` : {group: sorted LongTensor of template
indices}.  Initial L/M/S budget tiers use prefix index sets.

State-dict keys of the slim models are *identical* to models.build_model(...),
so the ESPFL template can be checked against the reference architectures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IndexSet = Dict[str, torch.Tensor]

VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']          # models.py appends the last 'M'
RESNET_BASE = [64, 96, 128, 256, 512]                           # stem, layer1..4 (models.ResNetFed)

_DATASET = {
    'cifar10':  {'num_classes': 10,  'in_channels': 3, 'input_size': 32},
    'cifar100': {'num_classes': 100, 'in_channels': 3, 'input_size': 32},
    'femnist':  {'num_classes': 62,  'in_channels': 1, 'input_size': 28},
    'mnist':    {'num_classes': 10,  'in_channels': 1, 'input_size': 28},
}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Group:
    name: str
    full: int            # width in the full template
    prunable: bool       # may EASA contract it?  (budget scaling applies regardless)
    scalable: bool = True  # False for pseudo groups 'input' / 'classes'


@dataclass
class ParamSpec:
    key: str                     # state_dict key
    out_group: Optional[str]     # group indexing dim 0
    in_group: Optional[str]      # group indexing dim 1
    tail: int = 1                # product of remaining dims (kernel size)
    in_repeat: int = 1           # flatten factor (conv channel -> linear features)


class ArchSpec:
    """Channel-group description of one backbone/dataset pair."""

    def __init__(self, arch: str, groups: List[Group], params: List[ParamSpec],
                 num_classes: int, in_channels: int, spatial_last: int,
                 layer_order: List[str]):
        self.arch = arch
        self.groups: Dict[str, Group] = {g.name: g for g in groups}
        self.params = params
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_last = spatial_last
        # groups in forward order, used for logging / layer coefficients
        self.layer_order = layer_order
        self.prunable = [g for g in layer_order if self.groups[g].prunable]
        # group -> list of ParamSpec whose dim0 is that group ("own" parameter block)
        self.out_params: Dict[str, List[ParamSpec]] = {}
        for p in params:
            if p.out_group is not None:
                self.out_params.setdefault(p.out_group, []).append(p)
        self.full_widths = {g: self.groups[g].full for g in self.groups}
        self.d = self.count(self.full_widths)

    # ── widths / index sets ────────────────────────────────────────────────
    def scale_widths(self, scale: float) -> Dict[str, int]:
        w = {}
        for name, g in self.groups.items():
            w[name] = g.full if not g.scalable else max(1, int(g.full * scale))
        return w

    def prefix_index(self, scale: float) -> IndexSet:
        return {g: torch.arange(w, dtype=torch.long) for g, w in self.scale_widths(scale).items()}

    def full_index(self) -> IndexSet:
        return self.prefix_index(1.0)

    @staticmethod
    def widths_of(idx: IndexSet) -> Dict[str, int]:
        return {g: int(t.numel()) for g, t in idx.items()}

    def count(self, widths: Dict[str, int]) -> int:
        n = 0
        for p in self.params:
            a = widths[p.out_group] if p.out_group else 1
            b = widths[p.in_group] * p.in_repeat if p.in_group else 1
            n += a * b * p.tail
        return n

    def rho(self, widths: Dict[str, int]) -> float:
        return self.count(widths) / self.d

    # ── model construction ────────────────────────────────────────────────
    def build(self, widths: Dict[str, int]) -> nn.Module:
        if self.arch == 'vgg16':
            return SlimVGG16(widths, self.num_classes, self.in_channels)
        return SlimResNet18(widths, self.num_classes, self.in_channels)


def _vgg_feature_positions() -> List[Tuple[int, int]]:
    """[(conv_k, position of the Conv2d inside `features`)], mirroring models._make_layers."""
    pos, k, out = 0, 0, []
    for x in VGG16_CFG:
        if x == 'M':
            pos += 1
            continue
        out.append((k, pos))
        pos += 3            # Conv, BN, ReLU
        k += 1
    return out


def make_arch_spec(model_name: str, dataset_name: str, prune_fc_hidden: bool = True) -> ArchSpec:
    ds = _DATASET[dataset_name.lower()]
    C, ncls, H = ds['in_channels'], ds['num_classes'], ds['input_size']
    name = model_name.lower()

    if name == 'vgg16':
        if H != 32:
            raise ValueError("VGG16 template in models.py assumes 32x32 inputs.")
        convs = [x for x in VGG16_CFG if x != 'M']
        groups = [Group('input', C, False, scalable=False)]
        groups += [Group(f'c{k}', w, prunable=(k > 0)) for k, w in enumerate(convs)]
        groups += [Group('h1', 4096, prune_fc_hidden), Group('h2', 4096, prune_fc_hidden),
                   Group('classes', ncls, False, scalable=False)]
        n_pool = sum(1 for x in VGG16_CFG if x == 'M')
        spatial = H // (2 ** n_pool)
        params: List[ParamSpec] = []
        prev = 'input'
        for k, pos in _vgg_feature_positions():
            g = f'c{k}'
            params += [ParamSpec(f'features.{pos}.weight', g, prev, tail=9),
                       ParamSpec(f'features.{pos}.bias', g, None),
                       ParamSpec(f'features.{pos + 1}.weight', g, None),
                       ParamSpec(f'features.{pos + 1}.bias', g, None)]
            prev = g
        params += [ParamSpec('projector.0.weight', 'h1', prev, in_repeat=spatial * spatial),
                   ParamSpec('projector.0.bias', 'h1', None),
                   ParamSpec('projector.3.weight', 'h2', 'h1'),
                   ParamSpec('projector.3.bias', 'h2', None),
                   ParamSpec('classifier.weight', 'classes', 'h2'),
                   ParamSpec('classifier.bias', 'classes', None)]
        order = [f'c{k}' for k in range(len(convs))] + ['h1', 'h2']
        return ArchSpec('vgg16', groups, params, ncls, C, spatial, order)

    if name in ('resnet18',):
        groups = [Group('input', C, False, scalable=False),
                  Group('stem', RESNET_BASE[0], prunable=False)]
        order = []
        for s in range(1, 5):
            groups.append(Group(f'r{s}', RESNET_BASE[s], True))
            for k in range(2):
                groups.append(Group(f'b{s}{k}', RESNET_BASE[s], True))
        groups.append(Group('classes', ncls, False, scalable=False))
        params = [ParamSpec('conv1.0.weight', 'stem', 'input', tail=9),
                  ParamSpec('conv1.1.weight', 'stem', None),
                  ParamSpec('conv1.1.bias', 'stem', None)]
        for s in range(1, 5):
            for k in range(2):
                pre = f'layer{s}.{k}'
                in_stream = ('stem' if s == 1 else f'r{s - 1}') if k == 0 else f'r{s}'
                b, r = f'b{s}{k}', f'r{s}'
                params += [ParamSpec(f'{pre}.left.0.weight', b, in_stream, tail=9),
                           ParamSpec(f'{pre}.left.1.weight', b, None),
                           ParamSpec(f'{pre}.left.1.bias', b, None),
                           ParamSpec(f'{pre}.left.3.weight', r, b, tail=9),
                           ParamSpec(f'{pre}.left.4.weight', r, None),
                           ParamSpec(f'{pre}.left.4.bias', r, None)]
                if k == 0:
                    params += [ParamSpec(f'{pre}.shortcut.0.weight', r, in_stream, tail=1),
                               ParamSpec(f'{pre}.shortcut.1.weight', r, None),
                               ParamSpec(f'{pre}.shortcut.1.bias', r, None)]
                order.append(b)
            order.append(f'r{s}')
        params += [ParamSpec('fc.weight', 'classes', 'r4'), ParamSpec('fc.bias', 'classes', None)]
        return ArchSpec('resnet18', groups, params, ncls, C, 1, order)

    raise ValueError(f"Unsupported model for ESPFL slim templates: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Slim models (same state_dict keys as models.py; optional channel gates for AKI)
# ─────────────────────────────────────────────────────────────────────────────

class _Gated(nn.Module):
    """Mixin: `self.gates[group]` is an optional (C,) multiplier on that group's activations.
    Gates are plain attributes (not buffers) so they never enter the state_dict."""

    def set_gates(self, gates: Optional[Dict[str, torch.Tensor]]) -> None:
        self.gates = gates or {}

    def _g(self, x: torch.Tensor, group: str) -> torch.Tensor:
        gate = self.gates.get(group) if getattr(self, 'gates', None) else None
        if gate is None:
            return x
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * gate.view(shape)


class SlimVGG16(_Gated):
    def __init__(self, widths: Dict[str, int], num_classes: int, in_channels: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        self._relu_group: Dict[int, str] = {}
        c_in, k = in_channels, 0
        for x in VGG16_CFG:
            if x == 'M':
                layers.append(nn.MaxPool2d(2, 2))
                continue
            w = widths[f'c{k}']
            layers += [nn.Conv2d(c_in, w, 3, padding=1),
                       nn.BatchNorm2d(w, track_running_stats=False),
                       nn.ReLU(inplace=True)]
            self._relu_group[len(layers) - 1] = f'c{k}'
            c_in, k = w, k + 1
        self.features = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.Linear(c_in, widths['h1']), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(widths['h1'], widths['h2']), nn.ReLU(inplace=True), nn.Dropout())
        self.classifier = nn.Linear(widths['h2'], num_classes)
        self.gates = {}

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._relu_group:
                x = self._g(x, self._relu_group[i])
        x = x.view(x.size(0), -1)
        p = self.projector
        x = self._g(p[1](p[0](x)), 'h1')
        x = p[2](x)
        x = self._g(p[4](p[3](x)), 'h2')
        x = p[5](x)
        return self.classifier(x)


class _SlimBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, stride: int, projection: bool):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_mid, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out, track_running_stats=False))
        # The template's shortcut type is fixed by position (projection for block 0
        # of every stage), never by the current widths.
        self.shortcut = nn.Sequential()
        if projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out, track_running_stats=False))

    def forward(self, x, gate_fn, g_mid: str, g_out: str):
        h = self.left[2](self.left[1](self.left[0](x)))
        h = gate_fn(h, g_mid)
        h = self.left[4](self.left[3](h))
        return gate_fn(F.relu(h + self.shortcut(x)), g_out)


class SlimResNet18(_Gated):
    def __init__(self, widths: Dict[str, int], num_classes: int, in_channels: int = 3):
        super().__init__()
        stem = widths['stem']
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem, track_running_stats=False),
            nn.ReLU(inplace=True))
        c_in = stem
        for s, stride in zip(range(1, 5), (1, 2, 2, 2)):
            r = widths[f'r{s}']
            blocks = [_SlimBlock(c_in, widths[f'b{s}0'], r, stride, projection=True),
                      _SlimBlock(r, widths[f'b{s}1'], r, 1, projection=False)]
            setattr(self, f'layer{s}', nn.Sequential(*blocks))
            c_in = r
        self.fc = nn.Linear(c_in, num_classes)
        self.gates = {}

    def forward(self, x):
        out = self.conv1(x)
        for s in range(1, 5):
            for k, blk in enumerate(getattr(self, f'layer{s}')):
                out = blk(out, self._g, f'b{s}{k}', f'r{s}')
        out = F.avg_pool2d(out, 4)
        return self.fc(out.view(out.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
# Index operations
# ─────────────────────────────────────────────────────────────────────────────

def _in_index(spec: ArchSpec, p: ParamSpec, idx: IndexSet) -> torch.Tensor:
    ii = idx[p.in_group]
    if p.in_repeat == 1:
        return ii
    r = p.in_repeat
    return (ii.view(-1, 1) * r + torch.arange(r).view(1, -1)).reshape(-1)


def extract_state(spec: ArchSpec, state: Dict[str, torch.Tensor], idx: IndexSet,
                  dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Slice `state` (indexed by `spec` coordinates of *that* state) down to `idx`.

    `state` may be the full template (idx = absolute template indices) or a client
    subnetwork (idx = positions within that subnetwork) -- the operation is the same.
    """
    out = {}
    for p in spec.params:
        t = state[p.key]
        if p.out_group is not None:
            t = t.index_select(0, idx[p.out_group].to(t.device))
        if p.in_group is not None:
            t = t.index_select(1, _in_index(spec, p, idx).to(t.device))
        out[p.key] = t.to(dtype) if dtype is not None else t.clone()
    return out


def relative_positions(outer: IndexSet, inner: IndexSet) -> IndexSet:
    """Positions of `inner` (absolute) inside `outer` (absolute); requires inner ⊆ outer."""
    rel = {}
    for g, o in outer.items():
        i = inner[g]
        pos = torch.searchsorted(o, i)
        if pos.numel() and (pos.max() >= o.numel() or not torch.equal(o[pos], i)):
            raise ValueError(f"group {g}: inner index set is not nested in outer")
        rel[g] = pos
    return rel


class CoverageAccumulator:
    """Server-side accumulator for coordinate-wise aggregation.

    For every template tensor we keep  num = Σ_i w_i Δ_i  over the coordinates Ω_i
    that client i actually uploaded, and the observation weight  den = Σ_{i∈O_j} w_i.
    The weight tensor is kept at (out, in) channel resolution (kernel dims broadcast).
    """

    def __init__(self, spec: ArchSpec, template_state: Dict[str, torch.Tensor]):
        self.spec = spec
        self.num = {p.key: torch.zeros_like(template_state[p.key], dtype=torch.float32)
                    for p in spec.params}
        self.den = {}
        for p in spec.params:
            shp = template_state[p.key].shape
            self.den[p.key] = torch.zeros(shp[:2] if len(shp) >= 2 else shp, dtype=torch.float32)
        self.total_weight = 0.0

    def add(self, delta: Dict[str, torch.Tensor], idx: IndexSet, weight: float) -> None:
        self.total_weight += weight
        for p in self.spec.params:
            d = delta[p.key].float().cpu() * weight
            if p.out_group is None:            # (does not occur in our specs)
                self.num[p.key] += d
                self.den[p.key] += weight
                continue
            o = idx[p.out_group]
            if p.in_group is None:
                self.num[p.key][o] += d
                self.den[p.key][o] += weight
            else:
                i = _in_index(self.spec, p, idx)
                oo, ii = o.view(-1, 1), i.view(1, -1)
                self.num[p.key][oo, ii] += d          # indices are unique -> plain += is exact
                self.den[p.key][oo, ii] += weight

    def apply(self, template_state: Dict[str, torch.Tensor], mode: str = 'sppa',
              server_lr: float = 1.0) -> Tuple[Dict[str, torch.Tensor], float]:
        """mode='sppa'    : Θ_j += Σ_{i∈O_j} w_i Δ_ij / Σ_{i∈O_j} w_i   (O_j ≠ ∅), else unchanged
           mode='zerofill': Θ_j += Σ_{i∈O_j} w_i Δ_ij / Σ_{i∈C_t} w_i   (unobserved = zero update)
        Returns the new state and the fraction of template coordinates observed this round."""
        new_state, covered, total = {}, 0, 0
        for p in self.spec.params:
            g = template_state[p.key].float()
            num, den = self.num[p.key], self.den[p.key]
            if num.dim() > den.dim():
                den = den.view(den.shape + (1,) * (num.dim() - den.dim())).expand_as(num)
            if mode == 'sppa':
                upd = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
            elif mode == 'zerofill':
                upd = num / max(self.total_weight, 1e-12)
            else:
                raise ValueError(mode)
            new_state[p.key] = (g + server_lr * upd).to(template_state[p.key].dtype)
            covered += int((den > 0).sum())
            total += num.numel()
        return new_state, covered / max(total, 1)


def check_against_reference(model_name: str, dataset_name: str) -> None:
    """Assert that the slim full-width template has exactly models.build_model's keys/shapes."""
    from models import build_model
    spec = make_arch_spec(model_name, dataset_name)
    for scale in (1.0, 0.71, 0.5):
        ref = build_model(model_name, dataset_name, scale=scale).state_dict()
        slim = spec.build(spec.scale_widths(scale)).state_dict()
        assert list(ref.keys()) == list(slim.keys()), (scale, set(ref) ^ set(slim))
        for k in ref:
            assert ref[k].shape == slim[k].shape, (scale, k, ref[k].shape, slim[k].shape)
        n_ref = sum(v.numel() for v in ref.values())
        assert n_ref == spec.count(spec.scale_widths(scale)), (scale, n_ref)
