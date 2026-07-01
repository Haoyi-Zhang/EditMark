# Environment Capture

This de-identified record describes the execution environment used for the frozen evidence release. It is intentionally coarse enough for anonymous review while preserving the information needed to judge reproducibility cost and compatibility.

- Label: `formal-single-host-full`
- System class: Linux x86_64 single-host execution
- Execution mode: `single_host_canonical`
- Python version family: 3.10
- Accelerator class: multi-GPU CUDA host
- Visible accelerator count: 8
- Accelerator memory class: 40 GiB per device
- CUDA compatibility: 12.4
- Code snapshot digest: `e77c05e89cbc7cbb180b1f2504cbcc3d9706ed72dcdd851beb84a45d6e238a6b`
- Execution environment fingerprint: `28ee33d7e551f2adc5aed51a4417e50505dcbba9075f08d772ce69682cd27a7d`

## Package Versions

- `torch`: `2.6.0+cu124`
- `transformers`: `4.57.6`
- `numpy`: `2.2.6`
- `pandas`: not required by the frozen release checks

## Toolchain Availability

- C/C++ compiler: available
- Java compiler/runtime: available
- Node.js runtime: available
- Go toolchain: available
- CUDA device query: available

The review artifact does not include raw host names, kernel build strings, device bus identifiers, private paths, timestamps from infrastructure tools, credential state, or local account information. The shipped summaries and integrity checks are the intended review path; full GPU reruns are optional and require a comparable multi-GPU CUDA host.
