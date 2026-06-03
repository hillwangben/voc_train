#!/usr/bin/env python3
import configparser
import json
import stat
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required: {exc}")

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "PROFILE.jpg",
    "alg/speech_train_all_in_one/INFO.ini",
    "alg/speech_train_all_in_one/README.md",
    "alg/speech_train_all_in_one/PROFILE.jpg",
    "alg/speech_train_all_in_one/main/config.yaml",
    "alg/speech_train_all_in_one/main/train.sh",
    "alg/speech_train_all_in_one/main/eval.sh",
    "alg/speech_train_all_in_one/main/convert.sh",
    "alg/speech_train_all_in_one/main/service.sh",
    "docs/open_source_cpu_models.json",
    "docs/开源CPU小模型选型.md",
    "docs/开源CPU小模型替换执行报告.md",
    "docs/国内源CPU-Torch安装与复验报告.md",
    "docs/真实CPU小模型测试报告.md",
    "docs/其余9个流程真实CPU测试报告.md",
    "docs/训练平台集成使用说明.md",
    "env/speech_train_all-env/INFO.ini",
    "env/speech_train_all-env/README.md",
    "env/speech_train_all-env/PROFILE.jpg",
    "env/speech_train_all-env/Dockerfile",
    "env/speech_train_all-env/requirements-cpu-open-source.txt",
    "dataset/speech_train_sample-dataset/INFO.ini",
    "dataset/speech_train_sample-dataset/README.md",
    "dataset/speech_train_sample-dataset/PROFILE.jpg",
    "dataset/speech_train_sample-dataset/speech_train_sample-dataset.zip",
    "model/speech_train_base-model/INFO.ini",
    "model/speech_train_base-model/README.md",
    "model/speech_train_base-model/PROFILE.jpg",
    "model/speech_train_base-model/model.zip",
    "scripts/run_all_cpu_tasks.py",
    "scripts/run_real_vosk_asr.py",
    "scripts/run_real_remaining_tasks.py",
]

INFO_FILES = [
    "alg/speech_train_all_in_one/INFO.ini",
    "env/speech_train_all-env/INFO.ini",
    "dataset/speech_train_sample-dataset/INFO.ini",
    "model/speech_train_base-model/INFO.ini",
]

EXECUTABLE_SCRIPTS = [
    "alg/speech_train_all_in_one/main/train.sh",
    "alg/speech_train_all_in_one/main/eval.sh",
    "alg/speech_train_all_in_one/main/convert.sh",
    "alg/speech_train_all_in_one/main/service.sh",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    for rel in REQUIRED_FILES:
        require((ROOT / rel).exists(), f"missing required file: {rel}")

    for rel in INFO_FILES:
        parser = configparser.ConfigParser()
        parser.read(ROOT / rel, encoding="utf-8")
        require(parser.has_section("base"), f"missing [base] in {rel}")
        for key in ["name", "version", "tags", "description"]:
            require(parser.has_option("base", key), f"missing {key} in {rel}")

    config = yaml.safe_load((ROOT / "alg/speech_train_all_in_one/main/config.yaml").read_text(encoding="utf-8"))
    for task in ["train", "evaluate", "adapt", "service"]:
        require(task in config, f"missing {task} in config.yaml")

    json.loads((ROOT / "model/speech_train_base-model/config/model_info.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "dataset/speech_train_sample-dataset/metadata.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "docs/open_source_cpu_models.json").read_text(encoding="utf-8"))
    require(metadata.get("task_count") == 10, "dataset metadata must declare ten tasks")
    require(catalog.get("device_policy") == "cpu_only", "open source catalog must be cpu_only")

    for rel in EXECUTABLE_SCRIPTS:
        mode = (ROOT / rel).stat().st_mode
        require(mode & stat.S_IXUSR, f"script is not executable: {rel}")

    for rel in [
        "dataset/speech_train_sample-dataset/speech_train_sample-dataset.zip",
        "model/speech_train_base-model/model.zip",
    ]:
        with zipfile.ZipFile(ROOT / rel) as package:
            bad = package.testzip()
            require(bad is None, f"corrupt zip entry {bad} in {rel}")

    print("asset-package-ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"asset-package-invalid: {exc}", file=sys.stderr)
        sys.exit(1)
