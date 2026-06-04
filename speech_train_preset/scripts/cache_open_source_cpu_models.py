#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_MODELS = ROOT / "model" / "real_cpu_models"
FROZEN_MODELS = ROOT / "model" / "open_source_cpu_models"


HF_MODELS = {
    "sherpa_onnx_int8": {
        "repo_id": "csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09",
        "target": "sherpa-onnx-paraformer-zh-small-2024-03-09",
        "allow_patterns": ["*.onnx", "tokens.txt", "README.md", "test_wavs/*"],
    },
    "whisper_tiny_cpu": {
        "repo_id": "openai/whisper-tiny",
        "target": "openai-whisper-tiny",
        "allow_patterns": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "normalizer.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    },
    "speechbrain_ecapa": {
        "repo_id": "speechbrain/spkrec-ecapa-voxceleb",
        "target": "speechbrain-spkrec-ecapa-voxceleb",
        "allow_patterns": ["*.ckpt", "*.yaml", "*.txt", "*.md", "*.json"],
    },
    "minilm_intent": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "target": "sentence-transformers-all-MiniLM-L6-v2",
        "allow_patterns": [
            "config.json",
            "modules.json",
            "sentence_bert_config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "1_Pooling/config.json",
            "README.md",
        ],
    },
    "opus_mt_tiny": {
        "repo_id": "Helsinki-NLP/opus-mt-en-zh",
        "target": "Helsinki-NLP-opus-mt-en-zh",
        "allow_patterns": [
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "source.spm",
            "target.spm",
            "tokenizer_config.json",
            "vocab.json",
            "README.md",
        ],
    },
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copytree(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def cache_hf_model(repo_id, target, allow_patterns):
    from huggingface_hub import snapshot_download

    target_dir = REAL_MODELS / target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    return target_dir


def cache_deepfilternet():
    source = Path.home() / ".cache" / "DeepFilterNet" / "DeepFilterNet3"
    if not source.exists():
        subprocess.run([sys.executable, "-m", "df", "--help"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not source.exists():
        raise FileNotFoundError("DeepFilterNet3 cache not found after df probe")
    target = REAL_MODELS / "DeepFilterNet3"
    copytree(source, target)
    return target


def cache_rnnoise():
    target = REAL_MODELS / "rnnoise"
    target.mkdir(parents=True, exist_ok=True)
    urls = {
        "rnn_data.c": "https://raw.githubusercontent.com/xiph/rnnoise/master/src/rnn_data.c",
        "rnn_data.h": "https://raw.githubusercontent.com/xiph/rnnoise/master/src/rnn_data.h",
    }
    for filename, url in urls.items():
        urllib.request.urlretrieve(url, target / filename)
    write_json(
        target / "OFFLINE_RESOURCE.json",
        {
            "project": "RNNoise",
            "resource_type": "embedded_c_weights",
            "files": sorted(urls),
            "source": "https://github.com/xiph/rnnoise",
        },
    )
    return target


def cache_pycorrector_resources():
    import pycorrector

    package_root = Path(pycorrector.__file__).resolve().parent
    target = REAL_MODELS / "pycorrector-package-resources"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    copied = []
    for rel in ["data", "confusion", "utils"]:
        src = package_root / rel
        if src.exists():
            dst = target / rel
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(rel)
    write_json(
        target / "OFFLINE_RESOURCE.json",
        {
            "project": "pycorrector",
            "resource_type": "installed_package_resources",
            "source_package": str(package_root),
            "copied": copied,
        },
    )
    return target


def update_manifest(backend_name, bundled_path, freeze_status="bundled"):
    manifest_path = FROZEN_MODELS / backend_name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["freeze_status"] = freeze_status
    manifest["bundled_model_path"] = str(bundled_path.relative_to(ROOT))
    manifest["offline_cached"] = True
    manifest["offline_cache_path"] = str(bundled_path.relative_to(ROOT))
    locations = manifest.setdefault("engineering_locations", [])
    cache_rel = str(bundled_path.relative_to(ROOT))
    if cache_rel not in locations:
        locations.insert(0, cache_rel)
    write_json(manifest_path, manifest)


def main():
    REAL_MODELS.mkdir(parents=True, exist_ok=True)
    cached = {}

    vosk_path = REAL_MODELS / "vosk-model-small-en-us-0.15"
    if not vosk_path.exists():
        raise FileNotFoundError(f"missing existing Vosk model: {vosk_path}")
    cached["vosk_small"] = vosk_path

    for backend, spec in HF_MODELS.items():
        print(f"cache:{backend}:{spec['repo_id']}", flush=True)
        cached[backend] = cache_hf_model(**spec)

    print("cache:deepfilternet_cpu:DeepFilterNet3", flush=True)
    cached["deepfilternet_cpu"] = cache_deepfilternet()

    print("cache:rnnoise:embedded-weights", flush=True)
    cached["rnnoise"] = cache_rnnoise()

    print("cache:pycorrector_kenlm:package-resources", flush=True)
    cached["pycorrector_kenlm"] = cache_pycorrector_resources()

    for backend, path in cached.items():
        update_manifest(backend, path)

    summary = {
        "device_policy": "cpu_only",
        "offline_cached_backends": {
            backend: str(path.relative_to(ROOT)) for backend, path in sorted(cached.items())
        },
    }
    write_json(REAL_MODELS / "offline_cache_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
