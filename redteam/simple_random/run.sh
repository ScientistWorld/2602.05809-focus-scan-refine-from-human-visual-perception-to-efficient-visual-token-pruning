#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/audit.py run simple_random
