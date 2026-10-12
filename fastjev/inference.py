"""Inference-only extraction of the main public-benchmark and LIBERO protocols.

The prompt and readout policies follow the original experiment helpers in
bench/paper_supplement_common.py and bench/fast_jev_libero_protocol.py.
Dataset loading, metrics, analysis and experiment queues are intentionally absent.
"""
from contextlib import nullcontext
import copy
import math
from pathlib import Path
from threading import Lock

import torch
from PIL import Image

from .config import load_config, select_layers, verify_checkpoint
from .runtime import configure_runtime

# The underlying implementation temporarily patches module-level functions.
# Serialize initialization and prediction across instances of this public API.
_INFERENCE_LOCK = Lock()


def model_class(protocol):
    from mso.infer import MSO1
    from mso.head import choice_outputs
    from mso import records

    class MainExperimentModel(MSO1):
        def _prompt(self, question, options):
            if protocol == "libero":
                return super()._prompt(question, options)
            row = self.active_input
            if question["instructions"] != row["question"]["instructions"]:
                raise ValueError("Question text does not match its image layout")
            loaded = []
            for path in row["images"]:
                with Image.open(path) as image:
                    loaded.append(image.convert("RGB"))
            frames, content = [], []
            for item in row["content_layout"]:
                if item["type"] == "image":
                    frames.append(loaded[item["image_index"]])
                    content.append({"type": "image"})
                else:
                    content.append({"type": "text", "text": item["text"]})
            self._vc = (frames, content)
            body = "".join(records.OPT_OPEN + str(option.get("text", "")) + records.OPT_CLOSE
                           for option in options)
            prompt = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                                   tokenize=False, add_generation_prompt=True)
            return prompt + body, frames

        def _finish(self, question, keys, options, mu, latency):
            if isinstance(mu, tuple):
                mu = mu[0]
            kind = question["type"]
            if kind == "noul":
                yes = float(mu[0])
                probabilities = {"yes": yes, "no": 1 - yes}
                prediction = "yes" if yes >= 0.5 else "no"
            else:
                output = choice_outputs(mu, allow_abstain=kind == "choice")
                values = [float(value) for value in output.probs]
                if len(values) != len(keys) or not keys:
                    raise ValueError("Every candidate must produce one score")
                if kind == "score":
                    total = sum(values)
                    if total <= 0:
                        raise ValueError("Invalid ordinal probabilities")
                    values = [value / total for value in values]
                probabilities = dict(zip(keys, values))
                prediction = max(probabilities, key=probabilities.get)
                if kind == "choice":
                    probabilities["__abstain__"] = float(output.abstain)
                if protocol == "libero":
                    prediction = max(probabilities, key=probabilities.get)
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("Invalid decision probabilities")
            if abs(sum(probabilities.values()) - 1) > 1e-5:
                raise ValueError("Decision probabilities must sum to one")
            return {"probabilities": probabilities, "prediction": prediction,
                    "raw_argmax": max(probabilities, key=probabilities.get)}

    return MainExperimentModel


def prepare_request(request, protocol, image_root):
    """Resolve user-supplied images without loading labels or dataset manifests."""
    if protocol not in ("public", "libero"):
        raise ValueError("Choose the public or libero inference protocol")
    row = copy.deepcopy(request)
    source_images = row.get("images") if protocol == "public" else row.get("state", {}).get("images")
    if not isinstance(source_images, list) or not source_images:
        raise ValueError("A nonempty image path list is required")
    resolved = []
    for name in source_images:
        if not isinstance(name, str) or not name:
            raise ValueError("Image paths must be nonempty strings")
        path = (Path(image_root) / Path(name).expanduser()).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
    if protocol == "public":
        question = row.get("question", {})
        if question.get("type") not in ("choice", "noul"):
            raise ValueError("Public benchmark requests use choice or noul questions")
        if not isinstance(question.get("instructions"), str):
            raise ValueError("Question instructions must be a string")
        if question["type"] == "choice":
            options = question.get("options", [])
            if not options or any(not isinstance(o.get("text"), str) for o in options):
                raise ValueError("Public choice questions require text options")
            keys = [option["key"] for option in options]
            if len(set(keys)) != len(keys) or "__abstain__" in keys:
                raise ValueError("Option keys must be unique and cannot be __abstain__")
        row["images"] = resolved
        row.setdefault("content_layout", [*({"type": "image", "image_index": i}
                                            for i in range(len(resolved))),
                                          {"type": "text", "text": question["instructions"]}])
        indices = []
        for item in row["content_layout"]:
            if item.get("type") == "image":
                index = item.get("image_index")
                if not isinstance(index, int) or not 0 <= index < len(resolved):
                    raise ValueError("Image index is outside the supplied images")
                indices.append(index)
            elif item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise ValueError("Layout entries must be image indices or text")
        if sorted(indices) != list(range(len(resolved))):
            raise ValueError("The public layout must include each supplied image exactly once")
    else:
        questions = row.get("questions", {})
        if not isinstance(questions, dict) or len(questions) != 8 or "r2_instr" not in questions:
            raise ValueError("The LIBERO protocol requires r2_instr and seven other questions")
        if any(q.get("type") not in ("choice", "noul", "score") for q in questions.values()):
            raise ValueError("Unsupported LIBERO question type")
        row["state"] = {"images": resolved}
    return row


class FastJEV:
    """Local OmniJev v1.1 inference with the main experiment's fixed masks."""
    def __init__(self, checkpoint, base_model, *, scale="4B", mode="full",
                 pruned_layers=None, protocol="public"):
        if protocol not in ("public", "libero"):
            raise ValueError("Choose the public or libero inference protocol")
        self.config = load_config(scale)
        self.layers = select_layers(self.config, mode, pruned_layers)
        self.mode, self.protocol = mode, protocol
        self.identity = verify_checkpoint(self.config, checkpoint, base_model)
        with _INFERENCE_LOCK:
            configure_runtime()
            self.model = model_class(protocol)(str(Path(checkpoint).expanduser().resolve()),
                                               str(Path(base_model).expanduser().resolve()),
                                               max_pixels=768 * 28 * 28)
            self.model.model.eval()
            from mso.phase_component_skip import PhaseComponentSkipController
            self.controller = PhaseComponentSkipController(self.model.model.backbone)
            if self.controller.layer_count != self.config["layer_count"] or not self.model.branch:
                raise ValueError("Loaded backbone does not match the selected v1.1 scale")

    def _context(self):
        if self.mode == "original":
            return nullcontext()
        if self.mode == "full":
            from mso.hybrid_prefix_trie import HybridTrieContext
            return HybridTrieContext(self.controller, shared=True, layers=self.layers)
        from mso.terminal_joint import TerminalJointContext
        mode = "recurrent_share_drop_pruned" if self.layers else "recurrent_share_drop"
        return TerminalJointContext(self.controller, mode, self.layers, max_tokens=4096)

    def predict(self, request, *, image_root="."):
        row = prepare_request(request, self.protocol, image_root)
        with _INFERENCE_LOCK, torch.inference_mode():
            # Cold image cache per input. LIBERO reuses it within the two groups.
            self.model._cache.clear()
            self.model._vc_key = None
            with self._context():
                if self.protocol == "public":
                    self.model.active_input = row
                    answers = self.model.ask_branch(row["images"][0], None, {"q": row["question"]})
                else:
                    questions = row["questions"]
                    groups = ({"r2_instr": questions["r2_instr"]},
                              {key: value for key, value in questions.items() if key != "r2_instr"})
                    answers = {}
                    for group in groups:
                        answers.update(self.model.system_one(row["state"], group))
        # The serving class's internal timing fields are not synchronized GPU
        # benchmark measurements. This inference-only API does not expose them.
        answers = {qid: {key: value for key, value in answer.items() if not key.startswith("latency")}
                   for qid, answer in answers.items()}
        return {"record_id": row.get("record_id"), "release": "v1.1",
                "scale": self.config["scale"], "mode": self.mode,
                "pruned_layers": self.layers, "protocol": self.protocol, "answers": answers}
