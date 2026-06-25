# Environment Capture

- Label: `formal-single-host-full`
- Hostname: `execution-host`
- FQDN: `execution-host`
- System: `Linux`
- Release: `3.10.0-1160.118.1.el7.x86_64`
- Version: `#1 SMP Wed Apr 24 16:01:50 UTC 2024`
- Machine: `x86_64`
- Python executable: `<release-python>/python`
- Python version: `3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]`
- Execution mode: `single_host_canonical`
- GPU count (physical): `8`
- GPU count (visible execution class): `8`
- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5,6,7`
- Code snapshot digest: `e77c05e89cbc7cbb180b1f2504cbcc3d9706ed72dcdd851beb84a45d6e238a6b`
- Execution environment fingerprint: `28ee33d7e551f2adc5aed51a4417e50505dcbba9075f08d772ce69682cd27a7d`
- GPU driver version: `550.163.01`
- CUDA version (torch build): `12.4`
- CUDA version (nvidia-smi): `12.4`

## Package Versions
- `torch`: `2.6.0+cu124`
- `transformers`: `4.57.6`
- `numpy`: `2.2.6`
- `pandas`: `None`

## GPU Devices
_Git metadata is unavailable in this execution-host work copy; use the recorded code snapshot digest as the release code-identity anchor._
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`
- `NVIDIA A800-SXM4-40GB` | driver `550.163.01` | memory `40960 MiB`

## Toolchain Checks
- `g++`: `ok`
  - stdout: `g++ (Ubuntu 13.4.0-6ubuntu1~22~ppa2) 13.4.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.`
- `javac`: `ok`
  - stdout: `javac 21.0.10`
- `java`: `ok`
  - stderr: `openjdk version "21.0.10" 2026-01-20
OpenJDK Runtime Environment (build 21.0.10+7-Ubuntu-122.04)
OpenJDK 64-Bit Server VM (build 21.0.10+7-Ubuntu-122.04, mixed mode, sharing)`
- `node`: `ok`
  - stdout: `v20.20.0`
- `go`: `ok`
  - stdout: `go version go1.22.12 linux/amd64`
- `nvidia_smi`: `ok`
  - stdout: `Sat Apr 25 01:50:32 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A800-SXM4-40GB          On  |   00000000:10:00.0 Off |                    0 |
| N/A   33C    P0             56W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A800-SXM4-40GB          On  |   00000000:16:00.0 Off |                    0 |
| N/A   31C    P0             54W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A800-SXM4-40GB          On  |   00000000:49:00.0 Off |                    0 |
| N/A   32C    P0             53W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A800-SXM4-40GB          On  |   00000000:4D:00.0 Off |                    0 |
| N/A   32C    P0             58W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A800-SXM4-40GB          On  |   00000000:8C:00.0 Off |                    0 |
| N/A   33C    P0             55W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A800-SXM4-40GB          On  |   00000000:91:00.0 Off |                    0 |
| N/A   32C    P0             52W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A800-SXM4-40GB          On  |   00000000:C7:00.0 Off |                    0 |
| N/A   31C    P0             58W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A800-SXM4-40GB          On  |   00000000:CB:00.0 Off |                    0 |
| N/A   33C    P0             54W /  400W |       1MiB /  40960MiB |      0%   E. Process |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+`
- `git`: `error`
  - error: `command returned non-zero exit status 128`
  - stderr: `fatal: not a git repository (or any parent up to mount point /root)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).`
