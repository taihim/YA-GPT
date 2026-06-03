# YA-GPT
Yet another GPT implementation from scratch!

## Development

Enter the Nix development shell, then sync the Python environment with uv:

```sh
nix develop
uv sync
```

The shell provides Python 3.13, uv, the C++ runtime required by PyTorch, and
the NixOS NVIDIA driver library path used by CUDA.
