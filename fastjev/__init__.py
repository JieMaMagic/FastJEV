"""FastJEV main-experiment inference on OmniJev v1.1."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "FastJEV":
        from .inference import FastJEV
        return FastJEV
    raise AttributeError(name)
