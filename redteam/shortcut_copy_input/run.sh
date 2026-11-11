#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python redteam/audit.py run shortcut_copy_input
