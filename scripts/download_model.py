"""Download official Qwen weights from ModelScope into ignored local storage."""
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/model"))
    args = parser.parse_args()
    from modelscope import snapshot_download

    model_dir = snapshot_download(
        "Qwen/Qwen3-4B-Instruct-2507",
        local_dir=str(args.output),
        allow_file_pattern=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    print(model_dir)
