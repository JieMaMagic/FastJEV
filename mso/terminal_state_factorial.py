"""Controlled terminal-state ablation for Qwen3.5 candidate scoring.

Four variants use one recurrent Triton kernel, changing only the initial-state
address (one shared batch or real per-candidate copies) and the final-state store.
The recurrence, dtype preparation, normalization, and launch geometry are fixed.
``chunk_copy_keep`` uses the installed original callables under the same context
wrappers, to measure the separate effect of switching recurrence implementations.

Keep variants preserve the installed cache's real ``copy_`` update: copy/keep
reuses its initial-state buffer, while share/keep allocates a private destination
before that update. Kernel final-state output and retained cache destination are
therefore distinct allocations. All branches are single-use terminal branches.
This patches process globals and is for single-threaded inference experiments.

Kernel derived from FLA 0.5.2's fla/ops/gated_delta_rule/fused_recurrent.py,
https://github.com/fla-org/flash-linear-attention .

MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
import math
from contextlib import AbstractContextManager
from contextvars import ContextVar
from types import MethodType

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp

from mso import branch as BR


MODES = (
    "chunk_copy_keep",
    "recurrent_copy_keep",
    "recurrent_copy_drop",
    "recurrent_share_keep",
    "recurrent_share_drop",
)
_ACTIVE = ContextVar("omnijev_state_factorial", default=None)


@triton.jit(do_not_specialize=["T"])
def _factorial_recurrent_kernel(
    q, k, v, g, beta, o, h0, ht, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NORMALIZE_QK: tl.constexpr,
    SHARE_INITIAL: tl.constexpr, STORE_FINAL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = i_n * T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k, mask_v = o_k < K, o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_beta = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if SHARE_INITIAL:
        p_h0 = h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    else:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if NORMALIZE_QK:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV
        p_o += HV * V

    if STORE_FINAL:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h, mask=mask_h)


def factorial_recurrent_delta_rule(
    query, key, value, g, beta, *, initial_state=None, shared_initial=False,
    output_final_state=False, use_qk_l2norm_in_kernel=False, max_tokens=64, **kwargs,
):
    """Run the same recurrence under the two binary storage choices.

    Inputs q/k[B,T,H,K], v[B,T,HV,V], g/beta[B,T,HV]; 1 <= T <= max_tokens.
    The original experiment defaults to 64. Explicit limits up to 4096 allow
    full datasets with longer options; T remains a runtime loop bound and the
    kernel, numerical operations, and launch geometry do not change.
    Initial state is contiguous [1,HV,K,V] when shared, [B,HV,K,V] otherwise.
    A requested final state is a new FP32 tensor [B,HV,K,V]; neither mode ever
    writes initial_state. Only headwise beta and scalar log-decay are supported.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Factorial terminal recurrence is inference-only")
    if not isinstance(shared_initial, bool) or not isinstance(output_final_state, bool):
        raise TypeError("shared_initial and output_final_state must be booleans")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be an integer in [1,4096]")
    for name in ("cu_seqlens", "cu_seq_lens_q", "gk", "gv", "A_log", "dt_bias"):
        if kwargs.get(name) is not None:
            raise ValueError(f"{name} is not supported by factorial terminal recurrence")
    for name in ("use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval",
                 "state_v_first", "transpose_state_layout", "head_first"):
        if kwargs.get(name, False):
            raise ValueError(f"{name} is not supported by factorial terminal recurrence")
    tensors = (query, key, value, g, beta, initial_state)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("q/k/v/g/beta and initial_state must be tensors")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("Expected q/k[B,T,H,K] and v[B,T,HV,V]")
    batch, length, heads, key_dim = query.shape
    value_heads, value_dim = value.shape[2:]
    if (batch < 1 or not 1 <= length <= max_tokens or heads < 1 or value_heads < 1
            or value_heads % heads or not 1 <= key_dim <= 256 or not 1 <= value_dim <= 256):
        raise ValueError("Unsupported branch count, length, heads, or head dimension")
    if value.shape[:2] != (batch, length):
        raise ValueError("q/k/v batch and sequence dimensions must match")
    if g.shape != (batch, length, value_heads) or beta.shape != g.shape:
        raise ValueError("g and headwise beta must have shape [B,T,HV]")
    state_batch = 1 if shared_initial else batch
    if initial_state.shape != (state_batch, value_heads, key_dim, value_dim):
        raise ValueError(f"initial_state must have shape [{state_batch},HV,K,V]")
    if not initial_state.is_contiguous():
        raise ValueError("initial_state must be contiguous; expanded views are not accepted")
    if query.device.type != "cuda" or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("All tensors must be on the same CUDA device")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("q/k/v must use FP32, FP16 or BF16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("q/k/v dtypes must match")
    if any(tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
           for tensor in (g, beta, initial_state)):
        raise TypeError("g, beta and initial state must be floating point")
    scale = kwargs.get("scale")
    scale = key_dim ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")

    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    decay = g.float().contiguous()
    strength = beta.to(query.dtype).contiguous()
    output = torch.empty_like(v)
    final = (torch.empty((batch, value_heads, key_dim, value_dim), device=query.device,
                         dtype=torch.float32) if output_final_state else None)
    bk = triton.next_power_of_2(key_dim)
    bv = min(8, triton.next_power_of_2(value_dim))
    with torch.cuda.device(query.device):
        _factorial_recurrent_kernel[(triton.cdiv(value_dim, bv), batch * value_heads)](
            q, k, v, decay, strength, output, initial_state, final, scale, length,
            H=heads, HV=value_heads, K=key_dim, V=value_dim, BK=bk, BV=bv,
            NORMALIZE_QK=bool(use_qk_l2norm_in_kernel), SHARE_INITIAL=shared_initial,
            STORE_FINAL=output_final_state, num_warps=1, num_stages=3,
        )
    return output, final


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


class StateFactorialContext(AbstractContextManager):
    """Install equal instrumentation around the five terminal-state controls.

    ``inspect=True`` clones *all* prefix cache tensors and verifies immutability
    at exit; it must not be enabled in timing or memory measurements. Counters
    count cumulative logical bytes over calls, not simultaneously live memory.
    No unsupported-shape fallback is allowed: this experiment fails explicitly.
    """

    def __init__(self, mode, inspect=False, max_tokens=64):
        if mode not in MODES:
            raise ValueError(mode)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be an integer in [1,4096]")
        self.mode = mode
        self.inspect = bool(inspect)
        self.max_tokens = max_tokens
        self.shared_initial = "_share_" in mode
        self.store_final = mode.endswith("_keep")
        self.use_chunk = mode == "chunk_copy_keep"
        self.stats = {
            "mode": mode, "expansions": 0, "dispatch_calls": 0,
            "chunk_calls": 0, "recurrent_calls": 0, "fallback_calls": 0,
            "dense_state_copy_bytes": 0, "dense_state_copy_bytes_avoided": 0,
            "kernel_final_state_allocation_bytes": 0,
            "retained_final_state_bytes": 0, "new_cache_destination_bytes": 0,
            "cache_update_calls": 0, "dropped_cache_update_calls": 0,
            "prefix_snapshot_tensors": 0, "prefix_states_unchanged": True,
            "suffix_shapes": [], "initial_state_shapes": [],
            "retained_state_shapes": [],
        }
        self._snapshots = []
        self._entered = False

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as M

        if self._entered or getattr(BR.expand_cache, "_terminal_branch_patch", False):
            raise RuntimeError("StateFactorialContext cannot be nested")
        self.M = M
        self.old_expand = BR.expand_cache
        self.old_forward = M.Qwen3_5GatedDeltaNet.forward
        self.old_chunk = M.torch_chunk_gated_delta_rule
        self.old_recurrent = M.torch_recurrent_gated_delta_rule

        def expand(cache, copies):
            if torch.is_grad_enabled():
                raise RuntimeError("Factorial terminal cache is inference-only")
            if not isinstance(copies, int) or copies < 1:
                raise ValueError("copies must be a positive integer")
            if getattr(cache, "_omnijev_factorial_owner", None) is not None:
                raise RuntimeError("A terminal branch cache cannot be expanded again")
            if "update_recurrent_state" in vars(cache):
                raise RuntimeError("An instance-patched recurrent cache update is unsupported")
            new = copy.copy(cache)
            new.layers = []
            for layer in cache.layers:
                target = copy.copy(layer)
                for name, value in vars(layer).items():
                    if self.inspect:
                        for tensor in _tensors(value):
                            self._snapshots.append((tensor, tensor.clone()))
                            self.stats["prefix_snapshot_tensors"] += 1
                    if name == "recurrent_states":
                        if not isinstance(value, dict):
                            raise TypeError("Expected installed Qwen3.5 dictionary state cache")
                        for tensor in value.values():
                            if tensor is not None:
                                if tensor.ndim != 4 or tensor.shape[0] != 1 or not tensor.is_contiguous():
                                    raise ValueError("Expected contiguous single-prefix recurrent state")
                                count = copies * tensor.numel() * tensor.element_size()
                                field = ("dense_state_copy_bytes_avoided" if self.shared_initial
                                         else "dense_state_copy_bytes")
                                self.stats[field] += count
                        target.recurrent_states = dict(value) if self.shared_initial else BR._rep(value, copies)
                    else:
                        # Conv state can be updated in-place on T=1; full-attention
                        # KV and all other tensor fields keep the original copies.
                        setattr(target, name, BR._rep(value, copies))
                new.layers.append(target)
            new._omnijev_factorial_owner = self
            new._omnijev_factorial_seen_layers = set()
            new._omnijev_factorial_updated_layers = set()
            new._omnijev_factorial_copies = copies
            self.stats["expansions"] += 1
            return new

        expand._terminal_branch_patch = True
        expand._state_factorial_patch = True

        def forward(layer, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            owner = getattr(cache_params, "_omnijev_factorial_owner", None)
            if owner is None:
                return self.old_forward(layer, hidden_states, cache_params, attention_mask, **kwargs)
            if owner is not self or torch.is_grad_enabled():
                raise RuntimeError("Invalid factorial terminal branch execution")
            if layer.layer_idx in cache_params._omnijev_factorial_seen_layers:
                raise RuntimeError("A terminal branch cache cannot be reused")
            if not cache_params.has_previous_state(layer.layer_idx, state_idx=0):
                raise RuntimeError("Terminal branch requires a populated prefix cache")
            if cache_params.layers[layer.layer_idx].record_past:
                raise RuntimeError("record_past is unsupported for terminal branches")
            if not 1 <= hidden_states.shape[1] <= self.max_tokens:
                raise ValueError("Suffix exceeds the factorial experiment's token limit")
            cache_params._omnijev_factorial_seen_layers.add(layer.layer_idx)
            original_update = type(cache_params).update_recurrent_state

            def update(cache, state, layer_idx, state_idx=0, **update_kwargs):
                if layer_idx != layer.layer_idx or state_idx != 0:
                    raise RuntimeError("Unexpected layer or state index in terminal update")
                if layer_idx in cache._omnijev_factorial_updated_layers:
                    raise RuntimeError("A terminal state cannot be updated twice")
                cache._omnijev_factorial_updated_layers.add(layer_idx)
                if not self.store_final:
                    if state is not None:
                        raise RuntimeError("Drop variant unexpectedly returned a final state")
                    self.stats["dropped_cache_update_calls"] += 1
                    return None
                if not isinstance(state, torch.Tensor) or state.ndim != 4:
                    raise RuntimeError("Keep variant did not return a dense final state")
                current = cache.layers[layer_idx].recurrent_states[state_idx]
                if state.shape != (cache._omnijev_factorial_copies, *current.shape[1:]):
                    raise RuntimeError("Unexpected final-state shape")
                if self.shared_initial:
                    # Preserve the original cache update's real copy_. Assigning
                    # the kernel's final tensor directly would remove this cost.
                    destination = torch.empty_like(state)
                    cache.layers[layer_idx].recurrent_states[state_idx] = destination
                    self.stats["new_cache_destination_bytes"] += destination.numel() * destination.element_size()
                retained = original_update(cache, state, layer_idx, state_idx=state_idx, **update_kwargs)
                self.stats["cache_update_calls"] += 1
                self.stats["retained_final_state_bytes"] += retained.numel() * retained.element_size()
                if self.inspect:
                    if retained.data_ptr() == state.data_ptr():
                        raise RuntimeError("Cache update did not preserve a distinct retained destination")
                    if self.shared_initial and retained.data_ptr() == current.data_ptr():
                        raise RuntimeError("Shared prefix was reused as the final-state destination")
                    self.stats["retained_state_shapes"].append(list(retained.shape))
                return retained

            # Remove the bound method even on exceptions so it cannot create a
            # cache -> bound-method -> cache cycle retaining CUDA allocations.
            cache_params.update_recurrent_state = MethodType(update, cache_params)
            token = _ACTIVE.set(self)
            try:
                return self.old_forward(layer, hidden_states, cache_params, attention_mask, **kwargs)
            finally:
                _ACTIVE.reset(token)
                del cache_params.update_recurrent_state

        def dispatch(original):
            def wrapped(query, key, value, g, beta, **kwargs):
                if _ACTIVE.get() is not self:
                    return original(query, key, value, g=g, beta=beta, **kwargs)
                initial = kwargs.get("initial_state")
                if initial is None:
                    raise RuntimeError("Terminal branch lost its prefix state")
                if not 1 <= query.shape[1] <= self.max_tokens:
                    raise ValueError("Suffix exceeds the factorial experiment's token limit")
                expected_batch = 1 if self.shared_initial else query.shape[0]
                if initial.shape[0] != expected_batch or not initial.is_contiguous():
                    raise RuntimeError("Initial state does not match the selected storage factor")
                kwargs["output_final_state"] = self.store_final
                self.stats["dispatch_calls"] += 1
                if self.inspect:
                    self.stats["suffix_shapes"].append(list(query.shape))
                    self.stats["initial_state_shapes"].append(list(initial.shape))
                if self.use_chunk:
                    # Respect the original T=1 dispatch too: this is the actual
                    # serving implementation, wrapped with matched overhead.
                    self.stats["chunk_calls"] += 1
                    output, final = original(query, key, value, g=g, beta=beta, **kwargs)
                else:
                    self.stats["recurrent_calls"] += 1
                    output, final = factorial_recurrent_delta_rule(
                        query, key, value, g=g, beta=beta,
                        shared_initial=self.shared_initial, max_tokens=self.max_tokens, **kwargs,
                    )
                if self.store_final:
                    if not isinstance(final, torch.Tensor):
                        raise RuntimeError("Keep kernel did not allocate a final state")
                    self.stats["kernel_final_state_allocation_bytes"] += final.numel() * final.element_size()
                elif final is not None:
                    raise RuntimeError("Drop kernel unexpectedly allocated a final state")
                return output, final
            return wrapped

        BR.expand_cache = expand
        M.Qwen3_5GatedDeltaNet.forward = forward
        M.torch_chunk_gated_delta_rule = dispatch(self.old_chunk)
        M.torch_recurrent_gated_delta_rule = dispatch(self.old_recurrent)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered:
            return False
        BR.expand_cache = self.old_expand
        self.M.Qwen3_5GatedDeltaNet.forward = self.old_forward
        self.M.torch_chunk_gated_delta_rule = self.old_chunk
        self.M.torch_recurrent_gated_delta_rule = self.old_recurrent
        self._entered = False
        try:
            if self.inspect:
                self.stats["prefix_states_unchanged"] = all(
                    torch.equal(tensor, before) for tensor, before in self._snapshots
                )
        finally:
            self._snapshots.clear()
        if not self.stats["prefix_states_unchanged"]:
            raise RuntimeError("Prefix cache was modified by a terminal branch")
        return False
