"""Runtime settings used by the main inference experiments."""
import os


def configure_runtime():
    # Set these before loading the model or importing FLA kernels.
    os.environ["MSO_ATTN"] = "sdpa"
    os.environ["MSO_FLA"] = "1"
    os.environ["MSO_BRANCH"] = "1"
    os.environ["MSO_BRANCH_CHUNK"] = "64"
    os.environ["FLA_CACHE_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Main-experiment inference requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    import triton
    from pathlib import Path
    assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    if assembler.is_file():
        os.environ["TRITON_PTXAS_PATH"] = str(assembler)
    import transformers.models.qwen3_5.modeling_qwen3_5 as model_module
    import torch.nn.functional as functional

    # The same PyTorch convolution formula used in the main benchmark.
    def convolution(x, weight, bias=None, activation=None, **kwargs):
        y = functional.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias,
                              padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :x.shape[-1]]
        return (model_module.ACT2FN[activation](y) if activation else y).to(x.dtype)

    def convolution_update(x, state, weight, bias=None, activation=None, **kwargs):
        z = torch.cat([state, x], -1).to(weight.dtype)
        state.copy_(z[..., -state.shape[-1]:])
        y = functional.conv1d(z, weight.unsqueeze(1), bias, groups=x.shape[1])[..., -x.shape[-1]:]
        return (model_module.ACT2FN[activation](y) if activation else y).to(x.dtype)

    model_module.causal_conv1d_fn = convolution
    model_module.causal_conv1d_update = convolution_update
    from mso.fast_kernels import enable_fla
    if not enable_fla(verbose=False):
        raise RuntimeError("FLA is required for the released inference configurations")
