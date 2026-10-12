"""Candidate-only FFN, attention, and decoder-block bypass for terminal Jev scoring.

The explicit outer ``past_key_values`` argument distinguishes the complete
shared-prefix pass from the terminal candidate pass. Original weights, decoder
indices, and cache slots remain intact. Skipped attention leaves its branch
cache slot at the prefix state; that returned cache is not a valid continuation
cache. This controller rejects reuse of such a cache through the same controller.
It is intended for single-threaded inference, not generation or training.

FFN and attention interventions return zeros before the installed decoder's
residual addition. Whole blocks return their input without executing either
normalization, attention, or FFN. All instance methods are restored on exit.
"""
from contextlib import AbstractContextManager
from numbers import Integral
from types import MethodType
from weakref import WeakKeyDictionary

import torch

from mso.phase_ffn import PhaseFFNController, _base_and_adapters, _projection_macs


COMPONENTS = ("ffn", "attention", "block")
_MISSING = object()
_ACTIVE_BACKBONES = WeakKeyDictionary()


class PhaseComponentSkipController:
    def __init__(self, backbone):
        discovered = PhaseFFNController(backbone)
        self.backbone = backbone
        self.layers = {name: module for name, module in backbone.named_modules()
                       if type(module).__name__ == "Qwen3_5DecoderLayer"}
        if not self.layers:
            raise ValueError("No supported Qwen3_5DecoderLayer modules were found")
        self.layer_names = list(self.layers)
        self.layer_count = len(self.layer_names)
        self.layer_types = [layer.block_type for layer in self.layers.values()]
        self.attentions = {}
        self.original_macs = {}
        self.original_widths = {}
        self.attention_projection_macs = {}
        self.attention_projections = {}
        self._active = None
        self._frame = None
        self._terminal_caches = WeakKeyDictionary()
        records = []
        for index, (name, layer) in enumerate(self.layers.items()):
            mlp_name = name + ".mlp"
            if discovered.layers.get(mlp_name) is not layer.mlp:
                raise ValueError(f"Unrecognized decoder FFN layout: {name}")
            self.original_macs[name] = discovered.original_macs[mlp_name]
            self.original_widths[name] = discovered.original_widths[mlp_name]
            if layer.block_type == "linear_attention":
                attention = layer.linear_attn
                projection_names = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
            elif layer.block_type == "full_attention":
                attention = layer.self_attn
                projection_names = ("q_proj", "k_proj", "v_proj", "o_proj")
            else:
                raise ValueError(f"Unsupported attention type: {layer.block_type}")
            if attention.layer_idx != index:
                raise ValueError("Decoder discovery must preserve contiguous original cache indices")
            if any(not hasattr(layer, attr) for attr in ("input_layernorm", "post_attention_layernorm")):
                raise ValueError(f"Unrecognized decoder normalization layout: {name}")
            self.attentions[name] = attention
            self.attention_projections[name] = {attr: getattr(attention, attr) for attr in projection_names}
            projections = []
            for attr, projection in self.attention_projections[name].items():
                base, adapters = _base_and_adapters(projection)
                dense, lora = _projection_macs(projection)
                projections.append({"name": attr, "in_features": base.in_features,
                                    "out_features": base.out_features,
                                    "dense_macs_per_row": dense, "lora_macs_per_row": lora,
                                    "active_lora_ranks": [a.out_features for a, _, _ in adapters]})
            self.attention_projection_macs[name] = tuple(
                sum(projection[f"{kind}_macs_per_row"] for projection in projections)
                for kind in ("dense", "lora"))
            records.append({"index": index, "name": name, "type": layer.block_type,
                            "original_width": self.original_widths[name],
                            "ffn_dense_macs_per_row": self.original_macs[name][0],
                            "ffn_lora_macs_per_row": self.original_macs[name][1],
                            "attention_projection_dense_macs_per_row": self.attention_projection_macs[name][0],
                            "attention_projection_lora_macs_per_row": self.attention_projection_macs[name][1],
                            "attention_projections": projections})
        if set(discovered.layers) != {name + ".mlp" for name in self.layer_names}:
            raise ValueError("The supplied backbone contains additional unsupported FFNs")
        self.metadata = {
            "layer_count": self.layer_count, "layers": records,
            "original_model_parameter_count": sum(p.numel() for p in backbone.parameters()),
            "parameters_removed": 0, "replica_parameter_count": 0, "replica_resident_bytes": 0,
            "phase": "candidate only; outer past_key_values is not None; complete prefix unchanged",
            "definition": "FFN/attention bypass returns zeros before residual addition; block bypass is identity",
            "attention_state_semantics": "Skipped branch cache slots retain prefix state; terminal scoring only, not a continuation cache",
            "continuation_guard_scope": "Caches returned after attention/block bypass cannot be reused through this controller; do not use them outside it",
            "projection_mac_scope": "Dense Linear and active unmerged vanilla LoRA projections only; excludes convolution, recurrence, QK/AV, normalization, activation, residual and other model modules",
        }

    def context(self, layer_indices, component="ffn", inspect=False):
        return _ComponentSkipContext(self, layer_indices, component, inspect)


def _detail():
    return {"block_calls": 0, "skipped_block_calls": 0,
            "attention_calls": 0, "skipped_attention_calls": 0, "attention_module_calls": 0,
            "ffn_calls": 0, "skipped_ffn_calls": 0, "ffn_module_calls": 0,
            "valid_rows": 0, "padded_rows": 0, "executed_rows": 0,
            "actual_attention_rows": 0, "actual_ffn_rows": 0,
            **{f"{execution}_{part}_{kind}_macs": 0
               for execution in ("original", "actual")
               for part in ("ffn", "attention_projection")
               for kind in ("dense", "lora")}}


class _ComponentSkipContext(AbstractContextManager):
    def __init__(self, controller, layer_indices, component, inspect):
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}")
        try:
            indices = list(layer_indices)
        except TypeError as error:
            raise ValueError("layer_indices must be an iterable of integers") from error
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
            raise ValueError("Layer indices must be integers, excluding booleans")
        indices = [int(index) for index in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("Layer indices must not contain duplicates")
        if any(index < 0 or index >= controller.layer_count for index in indices):
            raise ValueError("Layer index is outside the discovered decoder layers")
        self.controller, self.component = controller, component
        self.indices, self.inspect = frozenset(indices), bool(inspect)
        self._restores = []
        self._entered = False
        self._finished = False
        self.stats = {
            "component": component, "phase": "candidate", "layer_indices": sorted(indices),
            "inspect": self.inspect, "backbone_calls": {"prefix": 0, "candidate": 0},
            "phases": {phase: {"calls": 0, "valid_tokens": 0, "padded_tokens": 0,
                               "backbone_rows": 0, **_detail()} for phase in ("prefix", "candidate")},
            "layers": {name: {"index": index, "type": controller.layer_types[index],
                              "selected": index in self.indices,
                              "prefix": _detail(), "candidate": _detail()}
                       for index, name in enumerate(controller.layer_names)},
            "terminal_caches_marked": 0, "restored": False,
        }

    def _patch(self, module, forward):
        self._restores.append((module, module.__dict__.get("forward", _MISSING)))
        module.forward = MethodType(forward, module)

    def _record(self, name, hidden_states, skip_attention, skip_ffn, skip_block):
        frame = self.controller._frame
        if hidden_states.ndim != 3 or tuple(hidden_states.shape[:2]) != frame["shape"]:
            raise ValueError("Decoder rows do not match the backbone token mask")
        part_rows = {"ffn": 0 if skip_ffn else frame["rows"],
                     "attention_projection": 0 if skip_attention else frame["rows"]}
        entry = self.stats["layers"][name][frame["phase"]]
        totals = self.stats["phases"][frame["phase"]]
        for detail in (entry, totals):
            detail["block_calls"] += 1
            detail["skipped_block_calls"] += int(skip_block)
            detail["attention_calls"] += 1
            detail["skipped_attention_calls"] += int(skip_attention)
            detail["attention_module_calls"] += int(not skip_block)
            detail["ffn_calls"] += 1
            detail["skipped_ffn_calls"] += int(skip_ffn)
            detail["ffn_module_calls"] += int(not skip_block)
            detail["valid_rows"] += frame["valid"]
            detail["padded_rows"] += frame["padded"]
            detail["executed_rows"] += frame["rows"]
            detail["actual_attention_rows"] += part_rows["attention_projection"]
            detail["actual_ffn_rows"] += part_rows["ffn"]
            for part, pieces in (("ffn", self.controller.original_macs[name]),
                                 ("attention_projection", self.controller.attention_projection_macs[name])):
                for kind, per_row in zip(("dense", "lora"), pieces):
                    detail[f"original_{part}_{kind}_macs"] += per_row * frame["rows"]
                    detail[f"actual_{part}_{kind}_macs"] += per_row * part_rows[part]

    def __enter__(self):
        c = self.controller
        if self._entered or self._finished:
            raise RuntimeError("Skip contexts are single-use and cannot be re-entered")
        if c._active is not None or c.backbone in _ACTIVE_BACKBONES:
            raise RuntimeError("Nested component skip contexts on the same backbone are unsupported")
        if torch.is_grad_enabled() or c.backbone.training:
            raise RuntimeError("Component skipping requires inference/no_grad and eval()")
        self._entered = True
        c._active = self
        _ACTIVE_BACKBONES[c.backbone] = self
        original_backbone = c.backbone.forward

        def backbone_forward(_backbone, *args, **kwargs):
            if c._frame is not None:
                raise RuntimeError("Recursive backbone calls are unsupported")
            if len(args) > 1:
                raise ValueError("Pass cache and attention mask as keyword arguments")
            cache = kwargs.get("past_key_values")
            phase = "candidate" if cache is not None else "prefix"
            if cache is not None and cache in c._terminal_caches:
                raise ValueError("Cannot continue a cache returned after terminal attention/block skipping")
            self.stats["backbone_calls"][phase] += 1
            frame = {"phase": phase}
            if self.inspect:
                value = kwargs.get("input_ids", args[0] if args else None)
                if value is None:
                    value = kwargs.get("inputs_embeds")
                if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
                    raise ValueError("Expected input_ids[B,T] or inputs_embeds[B,T,H]")
                batch, tokens = value.shape[:2]
                if batch < 1 or tokens < 1:
                    raise ValueError("Empty batch or token sequences are unsupported")
                mask = kwargs.get("attention_mask")
                if mask is None:
                    valid = batch * tokens
                elif not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] < tokens:
                    raise ValueError("Inspection requires a two-dimensional token attention mask")
                else:
                    valid = int(mask[:, -tokens:].bool().sum().item())
                frame.update(shape=(batch, tokens), rows=batch * tokens,
                             valid=valid, padded=batch * tokens - valid)
                detail = self.stats["phases"][phase]
                detail["calls"] += 1
                detail["valid_tokens"] += valid
                detail["padded_tokens"] += frame["padded"]
                detail["backbone_rows"] += frame["rows"]
            c._frame = frame
            try:
                result = original_backbone(*args, **kwargs)
                if phase == "candidate" and self.indices and self.component in ("attention", "block"):
                    # The supplied DynamicCache is modified in place by the
                    # active layers even when return_dict/use_cache is disabled.
                    c._terminal_caches[cache] = True
                    self.stats["terminal_caches_marked"] += 1
                return result
            finally:
                c._frame = None

        try:
            self._patch(c.backbone, backbone_forward)
            for index, (name, layer) in enumerate(c.layers.items()):
                original_layer = layer.forward
                original_mlp = layer.mlp.forward
                attention = c.attentions[name]
                original_attention = attention.forward
                layer_type = layer.block_type

                def layer_forward(_layer, hidden_states, *args, _name=name, _index=index,
                                  _original=original_layer, **kwargs):
                    if c._frame is None:
                        raise RuntimeError("Decoder called outside the explicit backbone phase")
                    selected = c._frame["phase"] == "candidate" and _index in self.indices
                    skip_block = selected and self.component == "block"
                    skip_attention = selected and self.component in ("attention", "block")
                    skip_ffn = selected and self.component in ("ffn", "block")
                    if self.inspect:
                        self._record(_name, hidden_states, skip_attention, skip_ffn, skip_block)
                    if skip_block:
                        return hidden_states
                    return _original(hidden_states, *args, **kwargs)

                def mlp_forward(_mlp, hidden_states, _index=index, _original=original_mlp):
                    if c._frame is None:
                        raise RuntimeError("FFN called outside the explicit backbone phase")
                    if c._frame["phase"] == "candidate" and _index in self.indices and self.component == "ffn":
                        return torch.zeros_like(hidden_states)
                    return _original(hidden_states)

                def attention_forward(_attention, hidden_states, *args, _index=index,
                                      _original=original_attention, _type=layer_type, **kwargs):
                    if c._frame is None:
                        raise RuntimeError("Attention called outside the explicit backbone phase")
                    if c._frame["phase"] == "candidate" and _index in self.indices and self.component == "attention":
                        zeros = torch.zeros_like(hidden_states)
                        return zeros if _type == "linear_attention" else (zeros, None)
                    return _original(hidden_states, *args, **kwargs)

                self._patch(layer, layer_forward)
                self._patch(layer.mlp, mlp_forward)
                self._patch(attention, attention_forward)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if not self._entered or self._finished:
            return False
        for module, previous in reversed(self._restores):
            if previous is _MISSING:
                del module.forward
            else:
                module.forward = previous
        self._restores.clear()
        c = self.controller
        c._frame = None
        c._active = None
        if _ACTIVE_BACKBONES.get(c.backbone) is self:
            del _ACTIVE_BACKBONES[c.backbone]
        self._finished = True
        self.stats["restored"] = True
        return False
