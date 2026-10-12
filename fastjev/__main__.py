"""Run main-experiment inference on user-supplied JSON or JSONL requests."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from .config import MODES, SCALES, load_config, select_layers, verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=SCALES, default="4B")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--protocol", choices=("public", "libero"), default="public")
    parser.add_argument("--pruned-layers", type=int, help="Defaults to 6, or 0 for Original")
    parser.add_argument("--checkpoint", type=Path, help="Local OmniJev v1.1 adapter directory")
    parser.add_argument("--base-model", type=Path, help="Local pinned Qwen3.5 backbone directory")
    parser.add_argument("--input", type=Path, help="One JSON request, a JSON list, or JSONL")
    parser.add_argument("--image-root", type=Path, help="Defaults to the input file's directory")
    parser.add_argument("--output", type=Path, help="New output JSONL file; defaults to stdout")
    parser.add_argument("--show-config", action="store_true", help="Show identities and masks without loading a model")
    parser.add_argument("--verify-only", action="store_true", help="Check local model files without loading them")
    args = parser.parse_args()
    config = load_config(args.scale)
    try:
        select_layers(config, args.mode, args.pruned_layers)
        if args.show_config:
            print(json.dumps(config, indent=2))
            return
        if args.checkpoint is None or args.base_model is None:
            parser.error("--checkpoint and --base-model are required")
        if args.verify_only:
            print(json.dumps(verify_checkpoint(config, args.checkpoint, args.base_model), indent=2))
            return
        if args.input is None:
            parser.error("--input is required for inference")
        text = args.input.read_text()
        requests = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if args.input.suffix == ".jsonl" else json.loads(text))
        if isinstance(requests, dict):
            requests = [requests]
        if not isinstance(requests, list) or not requests or any(not isinstance(r, dict) for r in requests):
            raise ValueError("Input must contain at least one request object")
        if args.output and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        image_root = args.image_root if args.image_root is not None else args.input.resolve().parent
        with redirect_stdout(sys.stderr):
            from .inference import FastJEV, prepare_request
            for request in requests:
                prepare_request(request, args.protocol, image_root)
            engine = FastJEV(args.checkpoint, args.base_model, scale=args.scale, mode=args.mode,
                             pruned_layers=args.pruned_layers, protocol=args.protocol)
        stream = sys.stdout
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            stream = args.output.open("x")
        try:
            for request in requests:
                with redirect_stdout(sys.stderr):
                    result = engine.predict(request, image_root=image_root)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        finally:
            if args.output:
                stream.close()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
