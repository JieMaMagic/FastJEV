<div align="center">

# FastJEV
### Understanding Redundancy for Compact JEV Inference


**[Paper](https://arxiv.org/abs/2610.11379) · [PDF](https://arxiv.org/pdf/2610.11379) · [Quick Start](#quick-start) · [Citation](#citation)**

</div>

Official inference code for **FastJEV**, a training-free approach to compact
candidate evaluation in JEV models.

## Overview

JEV models answer multimodal questions by directly scoring candidate answers.
Although the common context is encoded once, candidate evaluation can still
repeat the same token computations, duplicate recurrent states, and execute
the full backbone for every candidate. **FastJEV** studies how these sources of
redundancy can be reduced together.

We organize candidate evaluation around a shared context anchor: recurrent
initial states are reused, matching candidate prefixes share computation, and
selected layers are bypassed according to their effect on relative decision
scores. This preserves full context encoding and all answer candidates without
additional training. Our experiments examine both the resulting compactness and
its practical cost, showing how candidate overlap and branching structure affect
latency and memory use.

![FastJEV framework from Figure 1 of the paper](https://arxiv.org/html/2610.11379v1/framework.png)

## Main Results

Reported results on **OmniJev v1.1** across **AI2D, ScienceQA-IMG, POPE, BLINK,
ERQA, and reconstructed LIBERO-10 offline QA**:

| Model | Pruned candidate blocks | Candidate depth reduction | Mean score retention |
| --- | ---: | ---: | ---: |
| 0.8B | 11 / 24 | **45.83%** | **97.52%** |
| 2B | 11 / 24 | **45.83%** | **94.92%** |
| 4B | 14 / 32 | **43.75%** | **93.66%** |

These are the selected larger-budget configurations from [Table 1 of the paper](https://arxiv.org/html/2610.11379v1).
Retention averages six per-dataset accuracy ratios to the matched Original;
it is not absolute accuracy. The CLI defaults to six pruned blocks.

Execution cost also depends on branching and scheduling: the **S+P** ablation
achieves **1.053–1.258×** speedups across the 18 model/dataset comparisons, while
full **T+S+P** can reduce repeated computation yet increase latency; see
[Section 4.3 of the paper](https://arxiv.org/html/2610.11379v1).

## Quick Start

### 1. Installation

The experiment environment uses **Linux, Python 3.12, an NVIDIA CUDA GPU,
PyTorch 2.7.1 / CUDA 12.6, and Transformers 5.17.0**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1
python -m pip install -e .
```

Keep the dependencies pinned in [pyproject.toml](pyproject.toml): the code uses
specific Qwen3.5 interfaces.

### 2. Download the models

Use an OmniJev adapter and its matching Qwen3.5 backbone. Each configuration
records the exact model revisions, checksums, and pruning masks used here.

| Scale | Adapter | Backbone | Configuration |
| --- | --- | --- | --- |
| 0.8B | [OmniJev-0.8B](https://huggingface.co/tinnel123/OmniJev-0.8B) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [0.8B.json](fastjev/configs/0.8B.json) |
| 2B | [OmniJev-2B](https://huggingface.co/tinnel123/OmniJev-2B) | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | [2B.json](fastjev/configs/2B.json) |
| 4B | [OmniJev](https://huggingface.co/tinnel123/OmniJev) | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [4B.json](fastjev/configs/4B.json) |

For **4B**:

```bash
hf download tinnel123/OmniJev \
  --revision 74287cfc256c5198339e82fdac027cb29282703a --local-dir ckpt/4B
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir base/4B
```

For 0.8B or 2B, use the repositories and revisions in the corresponding
configuration. The inference entry point verifies local model files before
loading and never downloads weights automatically.

### 3. Prepare an input

Edit [examples/request.json](examples/request.json) with your image, question,
and candidate answers:

```json
{
  "record_id": "user_image",
  "images": ["your_image.jpg"],
  "question": {
    "type": "choice",
    "instructions": "Which object is visible?",
    "options": [
      {"key": "A", "text": "a cup"},
      {"key": "B", "text": "a chair"},
      {"key": "C", "text": "a bicycle"}
    ]
  }
}
```

### 4. Run FastJEV

```bash
python -m fastjev \
  --scale 4B --checkpoint ckpt/4B --base-model base/4B \
  --mode full --pruned-layers 6 \
  --input examples/request.json --image-root /path/to/your/images \
  --output outputs/full.jsonl
```

Replace `/path/to/your/images` with your image directory. Each JSONL result
contains the predicted answer, candidate probabilities, and the selected layer
mask. Inputs may also be JSON lists or JSONL files.

## Configurations

The CLI exposes the full method and the independent-candidate ablation:

| Mode | Paper configuration | Candidate pruning |
| --- | --- | --- |
| `full` | **T+S+P** — complete FastJEV | Six blocks by default |
| `independent` | **S+P** — without trie sharing | Six blocks by default |
| `original` | **Original** — released OmniJev branch inference | None |

Change `--mode` in the command above to select a configuration. For `original`,
omit `--pruned-layers`. For the headline 4B result, use
`--mode full --pruned-layers 14`.

| Scale | Supported pruning budgets |
| --- | --- |
| 0.8B / 2B | 0, 6, 8, 11, 14 |
| 4B | 0, 6, 10, 14, 18 |

A budget of `0` disables pruning and retains the selected sharing behavior.
Configuration files use zero-based layer indices.

Two scoring protocols are available:

- **`--protocol public`** (default): one question with one or more images;
  choice predictions select among supplied options, and binary predictions use
  a threshold of 0.5.
- **`--protocol libero`**: eight offline questions per frame, evaluated in two
  groups with the upstream panel layout and ordinal head; choice predictions
  include abstention.

See the [inference reference](docs/inference.md) for binary questions, interleaved
image/text inputs, the LIBERO request schema, checkpoint verification, and the
Python API.

## Friendly Links
[Turing Post - Exploring System One AI Models Beyond Jev: 15 Outstanding Models](https://www.turingpost.com/p/exploring-system-one-ai-models-beyond-jev-15-outstanding-models)

## Citation

If you find this work useful, please cite:

```bibtex
@misc{ma2026fastjev,
  title         = {FastJEV: Understanding Redundancy for Compact JEV Inference},
  author        = {Jie Ma and Jie Gao and Yihang Liu and Zhike Qiu and Junle Li and Chongyi Zhuang and Jiayi Ji and Xiaoshuai Sun},
  year          = {2026},
  eprint        = {2610.11379},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.11379}
}
```

## Acknowledgments

We thank the [OmniJev](https://github.com/tinnel123666888/OmniJev) team for open-sourcing their models and code, which form the foundation of this work.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and upstream attribution;
the FLA-derived kernel retains its MIT notice in the source.
