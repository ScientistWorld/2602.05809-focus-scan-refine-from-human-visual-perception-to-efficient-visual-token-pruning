#!/bin/bash
# Fetch every external artifact into data/downloads/ (idempotent: existing files are kept).
#
# Estimated download: ~16 GB
#   - liuhaotian/llava-v1.5-7b @ 4481d270   (~13.5 GB; frozen language model + projector)
#   - openai/clip-vit-large-patch14-336 @ ce19dc91 (~1.7 GB; vision encoder, text encoder)
#   - COCO val2017 images                    (~0.8 GB; benchmark images)
# The evaluators verify every file against pinned hashes.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
D=data/downloads
mkdir -p "$D"

fetch_hf() {
    local repo="$1" rev="$2" dir="$3"; shift 3
    mkdir -p "$dir"
    for f in "$@"; do
        if [ ! -s "$dir/$f" ]; then
            echo "downloading $repo/$f"
            curl -L --fail --retry 3 -o "$dir/$f.part" "https://huggingface.co/$repo/resolve/$rev/$f"
            mv "$dir/$f.part" "$dir/$f"
        fi
    done
}

fetch_hf liuhaotian/llava-v1.5-7b 4481d270cc22fd5c4d1bb5df129622006ccd9234 "$D/llava-v1.5-7b" \
    config.json tokenizer.model tokenizer_config.json special_tokens_map.json \
    pytorch_model.bin.index.json pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin

fetch_hf openai/clip-vit-large-patch14-336 ce19dc912ca5cd21c8a653c79e251e808ccabcd1 "$D/clip-vit-large-patch14-336" \
    config.json preprocessor_config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
    special_tokens_map.json vocab.json merges.txt

if [ ! -d "$D/coco/val2017" ] || [ "$(ls "$D/coco/val2017" | wc -l)" -lt 5000 ]; then
    mkdir -p "$D/coco"
    echo "downloading COCO val2017 images"
    curl -L --fail --retry 3 -o "$D/coco/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
    (cd "$D/coco" && unzip -q -o val2017.zip && rm -f val2017.zip)
fi
echo "downloads ready in $D"
