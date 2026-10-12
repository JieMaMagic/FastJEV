"""Frozen inference masks and public checkpoint identities, without weights."""
import hashlib
import json
from pathlib import Path

SCALES = ("0.8B", "2B", "4B")
MODES = ("original", "full", "independent")


def load_config(scale):
    if scale not in SCALES:
        raise ValueError(f"Supported model scales are {SCALES}")
    return json.loads((Path(__file__).parent / "configs" / f"{scale}.json").read_text())


def select_layers(config, mode, pruned_layers=None):
    if mode not in MODES:
        raise ValueError(f"Supported execution modes are {MODES}")
    if mode == "original":
        if pruned_layers not in (None, 0):
            raise ValueError("Original executes all layers; omit --pruned-layers or use 0")
        return []
    budget = config["default_pruned_layers"] if pruned_layers is None else pruned_layers
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("The pruning budget must be an integer")
    if str(budget) not in config["pruning_masks"]:
        raise ValueError(f"Available budgets for {config['scale']}: {list(config['pruning_masks'])}")
    mask = config["pruning_masks"][str(budget)]
    if len(mask) != budget or len(set(mask)) != budget:
        raise ValueError("Invalid pruning mask")
    if any(i < 0 or i >= config["layer_count"] for i in mask):
        raise ValueError("Pruning mask contains an out-of-range layer")
    return list(mask)


def verify_checkpoint(config, checkpoint, base_model):
    """Verify local inference artifacts against the exact experimental release.

    No private download manifest is required. SHA256 checks include model shards,
    adapter weights, decision heads, tokenizer and preprocessing configuration.
    """
    verified = {}
    for role, folder in (("adapter", checkpoint), ("base", base_model)):
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {role} directory: {root}")
        count = 0
        for name, expected in config["artifacts"][role]["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Missing {role} artifact: {name}")
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Unexpected size for {role}/{name}; use the pinned v1.1 artifacts")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected["sha256"]:
                raise ValueError(f"Checkpoint identity mismatch: {role}/{name}")
            count += 1
        verified[role] = {"repo": config["artifacts"][role]["repo"],
                          "revision": config["artifacts"][role]["revision"],
                          "verified_files": count}
    return verified
