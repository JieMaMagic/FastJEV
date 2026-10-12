"""Training-free phase-specific dense FFN width experiments.

Only the text SwiGLU intermediate dimension is sliced. The original model is
never modified: private compact projections retain the original base and active
vanilla LoRA arithmetic, with gate/up output rows and down input columns sliced.
Both the complete original weights and compact replicas remain resident.

Contexts temporarily override instance forward methods. They are intended for
single-threaded inference, not training, generation, or concurrent requests.
Prefix/candidate phase comes from the actual backbone call's past_key_values.
"""
from contextlib import AbstractContextManager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("full", "prefix", "candidate", "both")
_MISSING = object()


def _base_and_adapters(projection):
    if isinstance(projection, nn.Linear):
        return projection, []
    base = getattr(projection, "base_layer", None)
    if not isinstance(base, nn.Linear) or not hasattr(projection, "lora_A"):
        raise TypeError(f"Only dense nn.Linear and vanilla PEFT LoRA are supported: {type(projection)}")
    if getattr(projection, "merged", False):
        raise ValueError("Unmerged LoRA is required; merging changes baseline arithmetic")
    if getattr(projection, "fan_in_fan_out", False):
        raise ValueError("Transposed LoRA base weights are unsupported")
    if getattr(projection, "disable_adapters", False):
        return base, []
    adapters = []
    for name in projection.active_adapters:
        if name not in projection.lora_A:
            continue
        if name in getattr(projection, "lora_variant", {}):
            raise ValueError("DoRA and other LoRA variants are unsupported")
        a, b = projection.lora_A[name], projection.lora_B[name]
        if not isinstance(a, nn.Linear) or not isinstance(b, nn.Linear):
            raise TypeError("LoRA factors must be dense Linear modules")
        if projection.lora_dropout[name].training:
            raise ValueError("LoRA dropout must be in evaluation mode")
        adapters.append((a, b, float(projection.scaling[name])))
    return base, adapters


def _projection_macs(projection):
    base, adapters = _base_and_adapters(projection)
    dense = base.in_features * base.out_features
    lora = sum(a.in_features * a.out_features + b.in_features * b.out_features
               for a, b, _ in adapters)
    return dense, lora


class _DenseProjection(nn.Module):
    def __init__(self, source, indices, axis):
        super().__init__()
        base, adapters = _base_and_adapters(source)
        ix = indices.to(base.weight.device)
        self.weight = nn.Parameter(base.weight.detach().index_select(axis, ix).clone(), requires_grad=False)
        bias = base.bias
        self.bias = (None if bias is None else nn.Parameter(
            (bias.detach().index_select(0, ix) if axis == 0 else bias.detach()).clone(), requires_grad=False))
        self.in_features, self.out_features = self.weight.shape[1], self.weight.shape[0]
        self.adapters = nn.ModuleList()
        self.scalings = []
        self.cast_input = bool(getattr(source, "cast_input_dtype_enabled", True))
        for a, b, scaling in adapters:
            factor = nn.Module()
            ai, bi = indices.to(a.weight.device), indices.to(b.weight.device)
            factor.a = nn.Parameter((a.weight.detach().index_select(1, ai) if axis == 1
                                     else a.weight.detach()).clone(), requires_grad=False)
            factor.b = nn.Parameter((b.weight.detach().index_select(0, bi) if axis == 0
                                     else b.weight.detach()).clone(), requires_grad=False)
            factor.a_bias = (None if a.bias is None else nn.Parameter(a.bias.detach().clone(), requires_grad=False))
            factor.b_bias = (None if b.bias is None else nn.Parameter(
                (b.bias.detach().index_select(0, bi) if axis == 0 else b.bias.detach()).clone(), requires_grad=False))
            self.adapters.append(factor)
            self.scalings.append(scaling)
        self.eval()

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        original_dtype = result.dtype
        for factor, scale in zip(self.adapters, self.scalings):
            if self.cast_input:
                x = x.to(factor.a.dtype)
            result = result + F.linear(F.linear(x, factor.a, factor.a_bias), factor.b, factor.b_bias) * scale
        return result.to(original_dtype)

    def macs(self):
        return (self.weight.numel(), sum(f.a.numel() + f.b.numel() for f in self.adapters))


class _CompactFFN(nn.Module):
    def __init__(self, source, indices):
        super().__init__()
        self.gate_proj = _DenseProjection(source.gate_proj, indices, 0)
        self.up_proj = _DenseProjection(source.up_proj, indices, 0)
        self.down_proj = _DenseProjection(source.down_proj, indices, 1)
        self.act_fn = source.act_fn
        self.intermediate_size = int(indices.numel())
        self.eval()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PhaseFFNController:
    """Own independent compact FFNs and reversible phase-dispatch contexts.

    ``calibration_context`` executes the unchanged model, collecting the actual
    SwiGLU activations received by each original down projection. The joint
    score is sqrt(sum(z**2)/valid_tokens) * ||effective_down_column||_2. Prefix
    and candidate tokens form one union; neither labels nor gradients are used.
    """
    def __init__(self, backbone):
        if backbone.training:
            raise ValueError("Set the backbone to eval() before constructing the controller")
        self.backbone = backbone
        self.layers = {name: module for name, module in backbone.named_modules()
                       if all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"))}
        if not self.layers:
            raise ValueError("No dense text SwiGLU FFNs were found")
        self.layer_names = list(self.layers)
        self.original_widths = {}
        self.original_macs = {}
        for name, mlp in self.layers.items():
            gate, _ = _base_and_adapters(mlp.gate_proj)
            up, _ = _base_and_adapters(mlp.up_proj)
            down, _ = _base_and_adapters(mlp.down_proj)
            if gate.weight.shape != up.weight.shape or gate.out_features != down.in_features or gate.in_features != down.out_features:
                raise ValueError(f"Unexpected dense SwiGLU layout at {name}")
            self.original_widths[name] = gate.out_features
            pieces = [_projection_macs(getattr(mlp, attr)) for attr in ("gate_proj", "up_proj", "down_proj")]
            self.original_macs[name] = tuple(sum(p[i] for p in pieces) for i in range(2))
        self.compact = {}
        self.kept_indices = {}
        self.preparation = None
        self.banks = {}
        self.selected_key = None
        self._active = None
        self._frame = None
        self._calibration = {}

    def calibration_context(self, reset=True):
        return _PhaseContext(self, "full", inspect=True, calibrate=True, reset=reset)

    def context(self, mode, inspect=False):
        return _PhaseContext(self, mode, inspect=inspect)

    @torch.no_grad()
    def prepare(self, keep_indices_per_layer, key=None):
        if self._active is not None:
            raise RuntimeError("Cannot replace compact modules inside an active context")
        if set(keep_indices_per_layer) != set(self.layers):
            raise ValueError("Provide exactly one index collection for every discovered FFN")
        checked = {}
        for name, width in self.original_widths.items():
            raw = torch.as_tensor(keep_indices_per_layer[name])
            if raw.ndim != 1 or raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError(f"Indices must be a one-dimensional integer list at {name}")
            ix = raw.to(device="cpu", dtype=torch.long)
            if not 1 <= ix.numel() <= width or ix.unique().numel() != ix.numel() or int(ix.min()) < 0 or int(ix.max()) >= width:
                raise ValueError(f"Invalid or duplicated retained indices at {name}")
            checked[name] = ix
        if key is not None and not isinstance(key, str):
            raise TypeError("Replica bank key must be a string or None")
        # Baseline modules are not attached to replicas and replicas are not
        # attached to the model. Named banks remain resident during all modes.
        compact = {name: _CompactFFN(self.layers[name], ix) for name, ix in checked.items()}
        indices = {name: ix.tolist() for name, ix in checked.items()}
        details = {}
        for name, replica in compact.items():
            pieces = [getattr(replica, attr).macs() for attr in ("gate_proj", "up_proj", "down_proj")]
            params = list(replica.parameters())
            details[name] = {"original_width": self.original_widths[name], "kept_width": replica.intermediate_size,
                             "original_dense_macs_per_row": self.original_macs[name][0],
                             "original_lora_macs_per_row": self.original_macs[name][1],
                             "compact_dense_macs_per_row": sum(p[0] for p in pieces),
                             "compact_lora_macs_per_row": sum(p[1] for p in pieces),
                             "compact_parameter_count": sum(p.numel() for p in params),
                             "compact_resident_bytes": sum(p.numel() * p.element_size() for p in params)}
        preparation = {"key": key, "layers": details,
                            "compact_parameter_count": sum(v["compact_parameter_count"] for v in details.values()),
                            "compact_resident_bytes": sum(v["compact_resident_bytes"] for v in details.values()),
                            "original_ffn_parameter_count": sum(p.numel() for mlp in self.layers.values() for p in mlp.parameters()),
                            "original_model_parameter_count": sum(p.numel() for p in self.backbone.parameters()),
                            "global_parameters_removed": 0,
                            "replicas_attached_to_original_model": False}
        self.banks[key] = {"compact": compact, "indices": indices, "metadata": preparation}
        self.select(key)
        return preparation

    def select(self, key=None):
        """Select a prepared bank by reference; no tensor copies or allocation."""
        if self._active is not None:
            raise RuntimeError("Cannot switch replica bank inside an active context")
        if key not in self.banks:
            raise KeyError(key)
        bank = self.banks[key]
        self.compact, self.kept_indices = bank["compact"], bank["indices"]
        self.preparation, self.selected_key = bank["metadata"], key
        return self.preparation

    @property
    def resident_replica_bytes(self):
        return sum(bank["metadata"]["compact_resident_bytes"] for bank in self.banks.values())

    @property
    def resident_replica_parameters(self):
        return sum(bank["metadata"]["compact_parameter_count"] for bank in self.banks.values())

    @torch.no_grad()
    def calibration_scores(self):
        if self._active is not None:
            raise RuntimeError("Read scores after leaving calibration context")
        records = {}
        for name, mlp in self.layers.items():
            record = self._calibration.get(name)
            if record is None or any(record[phase]["tokens"] < 1 for phase in ("prefix", "candidate")):
                raise RuntimeError(f"Both phases require valid calibration tokens at {name}")
            prefix, candidate = record["prefix"], record["candidate"]
            joint = ((prefix["sum_sq"] + candidate["sum_sq"]) / (prefix["tokens"] + candidate["tokens"])).sqrt()
            base, adapters = _base_and_adapters(mlp.down_proj)
            # FP32 materialization is temporary and does not merge or mutate
            # the original PEFT module. This is solely a ranking statistic.
            effective = base.weight.detach().float().clone()
            for a, b, scale in adapters:
                effective.addmm_(b.weight.detach().float(), a.weight.detach().float(), alpha=scale)
            norms = effective.square().sum(dim=0).sqrt().to(torch.float64)
            score = joint * norms
            if not torch.isfinite(score).all():
                raise RuntimeError(f"Non-finite calibration score at {name}")
            records[name] = {"width": self.original_widths[name],
                             "prefix_tokens": prefix["tokens"], "candidate_tokens": candidate["tokens"],
                             "prefix_rms": (prefix["sum_sq"] / prefix["tokens"]).sqrt().cpu().tolist(),
                             "candidate_rms": (candidate["sum_sq"] / candidate["tokens"]).sqrt().cpu().tolist(),
                             "joint_rms": joint.cpu().tolist(), "down_column_norm": norms.cpu().tolist(),
                             "score": score.cpu().tolist()}
        return {"definition": "RMS of actual SwiGLU activation over union of valid prefix/candidate tokens times effective down-column L2 norm including active LoRA",
                "padding_excluded": True, "labels_used": False, "gradients_used": False, "layers": records}


class _PhaseContext(AbstractContextManager):
    def __init__(self, controller, mode, inspect=False, calibrate=False, reset=False):
        if mode not in MODES:
            raise ValueError(mode)
        self.controller, self.mode = controller, mode
        self.inspect, self.calibrate, self.reset = bool(inspect), bool(calibrate), bool(reset)
        self._restores, self._hooks = [], []
        self.stats = {"mode": mode, "inspect": bool(inspect), "calibration": bool(calibrate),
                      "selected_bank": controller.selected_key,
                      "backbone_calls": {"prefix": 0, "candidate": 0},
                      "phases": {phase: {"calls": 0, "valid_tokens": 0, "padded_tokens": 0,
                                          "executed_rows": 0, "ffn_calls": 0,
                                          "compact_ffn_calls": 0, "original_dense_macs": 0,
                                          "actual_dense_macs": 0, "original_lora_macs": 0,
                                          "actual_lora_macs": 0} for phase in ("prefix", "candidate")},
                      "layers": {}, "restored": False}

    def _patch(self, module, forward):
        self._restores.append((module, module.__dict__.get("forward", _MISSING)))
        module.forward = MethodType(forward, module)

    def __enter__(self):
        c = self.controller
        if c._active is not None:
            raise RuntimeError("Nested phase contexts are unsupported")
        if torch.is_grad_enabled() or c.backbone.training:
            raise RuntimeError("Phase FFN contexts require inference/no_grad and eval()")
        if self.mode != "full" and set(c.compact) != set(c.layers):
            raise RuntimeError("Call prepare() before executing compact FFNs")
        c._active = self
        if self.calibrate and self.reset:
            c._calibration = {}
        original_backbone = c.backbone.forward

        def backbone_forward(_backbone, *args, **kwargs):
            if c._frame is not None:
                raise RuntimeError("Recursive backbone calls are unsupported")
            if len(args) > 1:
                raise ValueError("Pass cache and attention mask as keyword arguments")
            phase = "candidate" if kwargs.get("past_key_values") is not None else "prefix"
            self.stats["backbone_calls"][phase] += 1
            mask = None
            frame = {"phase": phase, "valid_mask": None}
            if self.inspect:
                value = kwargs.get("input_ids", args[0] if args else None)
                if value is None:
                    value = kwargs.get("inputs_embeds")
                if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
                    raise ValueError("Expected input_ids[B,T] or inputs_embeds[B,T,H]")
                batch, tokens = value.shape[:2]
                mask = kwargs.get("attention_mask")
                if mask is None:
                    mask = torch.ones((batch, tokens), device=value.device, dtype=torch.bool)
                elif mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] < tokens:
                    raise ValueError("Phase accounting requires a two-dimensional token attention mask")
                else:
                    mask = mask[:, -tokens:].bool()
                valid = int(mask.sum().item())
                frame.update(valid_mask=mask, shape=(batch, tokens), rows=batch * tokens, valid=valid)
                stats = self.stats["phases"][phase]
                stats["calls"] += 1
                stats["valid_tokens"] += valid
                stats["executed_rows"] += batch * tokens
                stats["padded_tokens"] += batch * tokens - valid
            c._frame = frame
            try:
                return original_backbone(*args, **kwargs)
            finally:
                c._frame = None

        try:
            self._patch(c.backbone, backbone_forward)
            for name, mlp in c.layers.items():
                original_forward = mlp.forward
                replica = c.compact.get(name)
                self.stats["layers"][name] = {"original_width": c.original_widths[name],
                                               "kept_width": replica.intermediate_size if replica is not None else None,
                                               "prefix_calls": 0, "candidate_calls": 0}

                def mlp_forward(_mlp, x, _name=name, _original=original_forward, _replica=replica):
                    frame = c._frame
                    if frame is None:
                        raise RuntimeError("FFN was called outside the explicit backbone phase")
                    phase = frame["phase"]
                    compact = self.mode == "both" or self.mode == phase
                    if self.inspect:
                        if tuple(x.shape[:2]) != frame["shape"]:
                            raise ValueError("FFN rows do not match input token mask")
                        details = self.stats["phases"][phase]
                        details["ffn_calls"] += 1
                        details["compact_ffn_calls"] += int(compact)
                        self.stats["layers"][_name][phase + "_calls"] += 1
                        dense, lora = c.original_macs[_name]
                        details["original_dense_macs"] += dense * frame["rows"]
                        details["original_lora_macs"] += lora * frame["rows"]
                        if compact:
                            prepared = c.preparation["layers"][_name]
                            dense, lora = prepared["compact_dense_macs_per_row"], prepared["compact_lora_macs_per_row"]
                        details["actual_dense_macs"] += dense * frame["rows"]
                        details["actual_lora_macs"] += lora * frame["rows"]
                    return _replica(x) if compact else _original(x)

                self._patch(mlp, mlp_forward)
                if self.calibrate:
                    def collect(_projection, args, _name=name):
                        frame = c._frame
                        z = args[0].detach()
                        if tuple(z.shape[:2]) != frame["shape"]:
                            raise ValueError("SwiGLU rows do not match valid-token mask")
                        phase = frame["phase"]
                        record = c._calibration.setdefault(_name, {})
                        accum = record.setdefault(phase, {"tokens": 0, "sum_sq": torch.zeros(
                            z.shape[-1], dtype=torch.float64, device=z.device)})
                        # Select valid rows before reduction: padding values are
                        # never counted, including nonzero padded hidden states.
                        squares = z[frame["valid_mask"]].float().square().sum(dim=0, dtype=torch.float64)
                        accum["sum_sq"].add_(squares)
                        accum["tokens"] += frame["valid"]
                    self._hooks.append(mlp.down_proj.register_forward_pre_hook(collect))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for module, previous in reversed(self._restores):
            if previous is _MISSING:
                del module.forward
            else:
                module.forward = previous
        self._restores.clear()
        self.controller._frame = None
        self.controller._active = None
        self.stats["restored"] = True
        return False
