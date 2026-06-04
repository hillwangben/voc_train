#!/usr/bin/env python3
import configparser
import json
import stat
import sys
import tarfile
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
    "docs/10个CPU流程独立测试说明.md",
    "docs/数据集更新-每类50条说明.md",
    "docs/最终过程文档与运行产物整理说明.md",
    "docs/训练平台集成使用说明.md",
    "env/speech_train_all-env/INFO.ini",
    "env/speech_train_all-env/README.md",
    "env/speech_train_all-env/PROFILE.jpg",
    "env/speech_train_all-env/Dockerfile",
    "env/speech_train_all-env/requirements-cpu-open-source.txt",
    "env/speech_train_all-env/ENVIRONMENT_MANIFEST.json",
    "env/speech_train_all-env/speech_train_all-env.tar.gz",
    "dataset/speech_train_sample-dataset/INFO.ini",
    "dataset/speech_train_sample-dataset/README.md",
    "dataset/speech_train_sample-dataset/PROFILE.jpg",
    "dataset/speech_train_sample-dataset/speech_train_sample-dataset.zip",
    "model/speech_train_base-model/INFO.ini",
    "model/speech_train_base-model/README.md",
    "model/speech_train_base-model/PROFILE.jpg",
    "model/speech_train_base-model/model.zip",
    "scripts/run_all_cpu_tasks.py",
    "scripts/cpu_flows/run_cpu_flow.py",
    "scripts/cpu_flows/run_asr.py",
    "scripts/cpu_flows/run_noise_enhancement.py",
    "scripts/cpu_flows/run_cross_channel_asr.py",
    "scripts/cpu_flows/run_low_resource_asr.py",
    "scripts/cpu_flows/run_code_switch_asr.py",
    "scripts/cpu_flows/run_speaker_verification.py",
    "scripts/cpu_flows/run_speech_interaction.py",
    "scripts/cpu_flows/run_speech_translation.py",
    "scripts/cpu_flows/run_transcription.py",
    "scripts/cpu_flows/run_speech_correction.py",
    "scripts/run_real_vosk_asr.py",
    "scripts/run_real_remaining_tasks.py",
    "scripts/cache_open_source_cpu_models.py",
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
    for backend in catalog.get("backends", []):
        backend_name = backend.get("name")
        manifest_path = ROOT / "model" / "open_source_cpu_models" / backend_name / "manifest.json"
        require(manifest_path.exists(), f"missing frozen open-source model manifest: {backend_name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("backend_name") == backend_name, f"manifest backend mismatch: {backend_name}")
        require(manifest.get("project") == backend.get("project"), f"manifest project mismatch: {backend_name}")
        require(manifest.get("device") == "cpu", f"manifest must be cpu-only: {backend_name}")
        require(manifest.get("engineering_locations"), f"manifest must declare engineering locations: {backend_name}")
        if manifest.get("offline_cached"):
            cache_path = ROOT / manifest.get("offline_cache_path", "")
            require(cache_path.exists(), f"offline model cache path missing: {backend_name}")

    for rel in EXECUTABLE_SCRIPTS:
        mode = (ROOT / rel).stat().st_mode
        require(mode & stat.S_IXUSR, f"script is not executable: {rel}")

    for rel in [
        "env/speech_train_all-env/speech_train_all-env.tar.gz",
        "dataset/speech_train_sample-dataset/speech_train_sample-dataset.zip",
        "model/speech_train_base-model/model.zip",
    ]:
        if rel.endswith(".zip"):
            with zipfile.ZipFile(ROOT / rel) as package:
                bad = package.testzip()
                require(bad is None, f"corrupt zip entry {bad} in {rel}")
        elif rel.endswith(".tar.gz"):
            require(tarfile.is_tarfile(ROOT / rel), f"corrupt tar archive: {rel}")

    print("asset-package-ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"asset-package-invalid: {exc}", file=sys.stderr)
        sys.exit(1)
