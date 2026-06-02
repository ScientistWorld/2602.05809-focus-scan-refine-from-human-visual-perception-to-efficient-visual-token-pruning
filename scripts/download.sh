#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This reproduction uses a generated benchmark and has no external downloads.
# The command below is deterministic and safe to run from a fresh clone.

set -e

cd /home/user

python data/generate_token_benchmark.py --output data/token_benchmark.jsonl
