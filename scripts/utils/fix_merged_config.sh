#!/bin/bash
# Fix merged model config by removing quantization_config
#
# The base model loaded in 4-bit during training retains quantization_config
# in config.json, but merged weights are full precision (bfloat16). vLLM
# doesn't support bitsandbytes quantization and fails to load such models.
#
# Usage:
#   bash scripts/utils/fix_merged_config.sh outputs/sft-overspecific/merged
#   bash scripts/utils/fix_merged_config.sh outputs/*/merged  # Fix all merged models

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <merged_model_path> [<merged_model_path2> ...]"
    echo ""
    echo "Examples:"
    echo "  $0 outputs/sft-overspecific/merged"
    echo "  $0 outputs/*/merged  # Fix all merged models"
    exit 1
fi

for MODEL_PATH in "$@"; do
    CONFIG_PATH="$MODEL_PATH/config.json"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "WARNING: Config not found: $CONFIG_PATH (skipping)"
        continue
    fi

    # Check if quantization_config exists
    if python3 -c "import json; c=json.load(open('$CONFIG_PATH')); exit(0 if 'quantization_config' in c else 1)" 2>/dev/null; then
        echo "Fixing: $CONFIG_PATH"
        python3 -c "
import json
with open('$CONFIG_PATH') as f:
    config = json.load(f)
del config['quantization_config']
with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=2)
"
        echo "  Removed quantization_config"
    else
        echo "OK: $CONFIG_PATH (no quantization_config)"
    fi
done

echo ""
echo "Done!"
