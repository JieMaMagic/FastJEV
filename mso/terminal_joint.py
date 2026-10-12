"""Compose terminal-state storage and candidate-only decoder block skipping.

All optimized modes enter the state patch before the component controller:
the controller captures attention methods on entry, so the reverse order would
silently bypass the patched GatedDeltaNet dispatch. Original is a true unwrapped
control. The other five modes have the same component and terminal-use wrappers.

This single-threaded, inference-only API never produces a continuation cache.
The original prefix may be expanded into multiple independent candidate chunks.
A candidate cache is consumed on its first attempted call, including failures;
reuse through this API is rejected before any decoder layer can read its state.
Do not pass terminal caches to a model outside this API.
"""
from contextlib import AbstractContextManager, ExitStack
from contextvars import ContextVar
import sys
from types import MethodType

import torch

from mso.terminal_state_factorial import StateFactorialContext


MODES = (
    "original",
    "wrapped_original",
    "recurrent_copy_keep",
    "recurrent_share_drop",
    "recurrent_copy_keep_pruned",
    "recurrent_share_drop_pruned",
)
STATE_MODES = {
    "wrapped_original": "chunk_copy_keep",
    "recurrent_copy_keep": "recurrent_copy_keep",
    "recurrent_share_drop": "recurrent_share_drop",
    "recurrent_copy_keep_pruned": "recurrent_copy_keep",
    "recurrent_share_drop_pruned": "recurrent_share_drop",
}
_ACTIVE_JOINT = ContextVar("omnijev_terminal_joint", default=None)


class TerminalJointContext(AbstractContextManager):
    """Use with ``torch.inference_mode()`` and a shared component controller.

    ``layer_indices`` must be empty for unpruned modes and nonempty for pruned
    modes. C and D must receive the same predeclared mask in the experiment.
    ``stats['state']`` and ``stats['component']`` retain the original counters;
    component projection details require ``inspect=True`` and should be omitted
    from performance runs. State counters include eager expansion of every GDN
    slot, including skipped blocks; they do not claim copies were eliminated by
    block skipping alone.
    """

    def __init__(self, controller, mode, layer_indices=(), *, inspect=False, max_tokens=64):
        if mode not in MODES:
            raise ValueError(f"Unknown terminal joint mode: {mode}")
        self.controller = controller
        self.mode = mode
        # Validate the indices before installing any global or instance patch.
        self.component = controller.context(layer_indices, "block", inspect=inspect)
        self.indices = sorted(self.component.indices)
        self.pruned = mode.endswith("_pruned")
        if self.pruned != bool(self.indices):
            raise ValueError("Only pruned modes accept a nonempty layer mask")
        self.state = (None if mode == "original" else
                      StateFactorialContext(STATE_MODES[mode], inspect=inspect, max_tokens=max_tokens))
        self.stats = {
            "mode": mode,
            "layer_indices": self.indices,
            "state": None if self.state is None else self.state.stats,
            "component": None if self.state is None else self.component.stats,
            "terminal_candidate_calls": 0,
            "rejected_terminal_reuses": 0,
            "restored": False,
            "continuation_guard_scope": "This API only; terminal caches must not escape to ordinary generation",
        }
        self._stack = None
        self._finished = False

    def __enter__(self):
        if self._stack is not None or self._finished:
            raise RuntimeError("Terminal joint contexts are single-use")
        if _ACTIVE_JOINT.get() is not None:
            raise RuntimeError("Terminal joint contexts cannot be nested")
        if torch.is_grad_enabled() or self.controller.backbone.training:
            raise RuntimeError("Terminal joint scoring requires inference/no_grad and eval()")
        stack = ExitStack()
        self._stack = stack
        token = _ACTIVE_JOINT.set(self)
        stack.callback(_ACTIVE_JOINT.reset, token)
        try:
            if self.state is not None:
                # Required order: component wrappers capture the state-patched
                # GDN class forward. ExitStack restores instances first.
                stack.enter_context(self.state)
                stack.enter_context(self.component)
                backbone = self.controller.backbone
                captured = backbone.forward

                def terminal_forward(_backbone, *args, **kwargs):
                    cache = kwargs.get("past_key_values")
                    if cache is not None:
                        if getattr(cache, "_omnijev_joint_consumed", False):
                            self.stats["rejected_terminal_reuses"] += 1
                            raise RuntimeError("A terminal candidate cache cannot be reused")
                        if getattr(cache, "_omnijev_factorial_owner", None) is not self.state:
                            raise RuntimeError("Candidate cache must be expanded from a complete prefix inside this context")
                        # Mark before execution: an exception may leave a
                        # partially advanced cache and must not permit retry.
                        cache._omnijev_joint_consumed = True
                        self.stats["terminal_candidate_calls"] += 1
                    return captured(*args, **kwargs)

                backbone.forward = MethodType(terminal_forward, backbone)
                stack.callback(setattr, backbone, "forward", captured)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self._stack is None or self._finished:
            return False
        self._finished = True
        try:
            return self._stack.__exit__(exc_type, exc, traceback)
        finally:
            self.stats["restored"] = True
