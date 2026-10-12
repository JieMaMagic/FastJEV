# Inference reference

[Back to FastJEV](../README.md)

This guide covers request formats, scoring conventions, and the Python API.
See the main README for installation, model downloads, and execution modes.

## Checkpoint verification

The entry point verifies local files against the pinned model configurations
before loading. Verification includes model weights, decision heads, tokenizer,
and preprocessing settings, and runs once per engine initialization.

```bash
python -m fastjev --scale 4B --show-config
python -m fastjev \
  --scale 4B --checkpoint ckpt/4B --base-model base/4B --verify-only
```

Both commands work without importing PyTorch. Use materialized local directories,
as produced by the `hf download --local-dir` commands in the main README.

## Requests and outputs

Input may be a single JSON request, a JSON list, or a `.jsonl` file. Requests are
processed sequentially. The [example request](../examples/request.json) is an
input template; supply your own image.

In the CLI, relative image paths are resolved against `--image-root`, or against
the input file's directory if that flag is omitted. In the Python API,
`image_root` defaults to the current working directory.

Outputs are JSONL. Omit `--output` to write to standard output; existing output
files are never overwritten. Each result includes:

| Field | Meaning |
| --- | --- |
| `record_id` | Input record identifier, when supplied. |
| `release`, `scale` | Model release and model size. |
| `mode`, `protocol` | Execution mode and scoring protocol. |
| `pruned_layers` | Selected zero-based layer indices. |
| `answers` | Predictions, candidate probabilities, and `raw_argmax` for each question. |

The public protocol returns one answer under `answers["q"]`; the LIBERO protocol
uses the question identifiers from the input.

## Input protocols


### Public benchmarks

The default `--protocol public` accepts inputs prepared for AI2D, ScienceQA-IMG,
POPE, BLINK, and ERQA. Supply `images` and one `question`, as in the
[example request](../examples/request.json). Dataset downloaders, converters,
and metric aggregation are not included.

Choice questions use an `options` list with unique candidate keys. The key
`__abstain__` is reserved. To ask a binary question, replace the `question` field
with:

```json
{
  "type": "noul",
  "instructions": "There is a cup."
}
```

By default, all images precede the question text. For native image/text
interleaving, add a top-level `content_layout`, including each image exactly once:

```json
{
  "content_layout": [
    {"type": "image", "image_index": 0},
    {"type": "text", "text": "Which object is visible?"}
  ]
}
```

Layout text is used directly in the prompt, so include the question and any
required text context. Multiple images remain separate in the public protocol.

Public choice predictions select among the supplied options. The probability
record also includes `__abstain__`, and `raw_argmax` may select it. Binary
predictions use the original sigmoid probability with a threshold of `0.5`.
No serving temperature or bias is applied.

### LIBERO offline questions

Use `--protocol libero` with `state.images` and a `questions` dictionary containing
eight questions, including `r2_instr`. Questions use the upstream `choice`,
`noul`, or `score` schema.

The entry point evaluates `r2_instr` first and the other seven questions in a
second request, reusing the image cache within the frame. It retains the upstream
panel layout and ordinal head, and choice predictions include abstention.

This protocol performs offline question answering on supplied images and
questions. The reconstructed LIBERO-10 question set, its generation pipeline,
and robot simulation or control are not included.

## Python API

```python
import json
from pathlib import Path

from fastjev import FastJEV

engine = FastJEV(
    "ckpt/4B",
    "base/4B",
    scale="4B",
    mode="full",
    pruned_layers=6,
    protocol="public",
)

request = json.loads(Path("examples/request.json").read_text())
result = engine.predict(request, image_root="/path/to/your/images")
print(result["answers"])
```

Reuse the engine for multiple requests to avoid reloading and rehashing the
checkpoint. In the Python API, `image_root` defaults to the current working
directory. Calls through this API are serialized because the inference
implementation temporarily patches shared functions.

## Runtime notes

The `full` and `independent` modes use the experimental recurrent kernel. Full
execution additionally uses segmented histories and a bounded vocabulary
projection schedule, so timing differences between modes include those effects.

The patches support inference only. The CLI does not measure synchronized GPU
latency or peak memory. Hardware, kernel scheduling, and floating-point behavior
can affect results; exact checkpoint hashes alone do not guarantee bitwise
reproduction of experiment outputs.
