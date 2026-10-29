"""
skills/lazy.py - load a model only when a skill needs it, keep it, and let it go when asked (step 13).

A model takes hundreds of megabytes of memory. Loading all three at start-up would be slow and
wasteful: "python app.py search" never needs the chat model. So every skill asks for its model
through lazy_load(). The first call loads it; later calls get the same object straight away.
release() deletes the models and asks Python (and the GPU, if any) to free the memory.
"""

import gc

_loaded = {}   # name -> the loaded model object


def lazy_load(name: str, load_function):
    """Return the model called `name`, loading it with load_function() the first time."""
    if name not in _loaded:
        _loaded[name] = load_function()
    return _loaded[name]


def loaded_names() -> list[str]:
    """Which models are in memory right now."""
    return list(_loaded)


def release(name: str | None = None) -> list[str]:
    """Forget one model (or all of them when name is None) and free the memory. Returns what was released."""
    names = [name] if name else list(_loaded)
    released = [n for n in names if _loaded.pop(n, None) is not None]
    gc.collect()                                   # delete the objects nobody points to any more
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()               # NVIDIA GPU memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()                # Apple Silicon GPU memory
    except ImportError:
        pass
    return released
