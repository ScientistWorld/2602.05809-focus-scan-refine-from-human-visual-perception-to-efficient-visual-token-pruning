#!/bin/bash
# Install Python packages into the container overlay (run by the system with internet).
# Use the system Python; no virtual environments.
#
# torch (CUDA 12.4 wheels bundle the CUDA runtime) runs the frozen LLaVA-1.5-7B and CLIP
# models in the evaluators; pruner programs only need numpy.

set -e

uv pip install --system \
    "numpy==1.26.4" \
    "torch==2.5.1" \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    "sentencepiece==0.2.0" \
    "protobuf==5.28.3" \
    "pillow==11.0.0"
