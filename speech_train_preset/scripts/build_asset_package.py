#!/usr/bin/env python3
"""
语音训练资产包端到端构建脚本(Stage-based)。

Stages:
  code     - 校验算法工程(alg/speech_train_all-alg/)结构、haikit/、源码、tests
  model    - 取消 symlink,复制真实权重到 9 个独立 model 资产
  dataset  - 重新打包 dataset/speech_train_sample-dataset.zip(含 train/val/test 划分)
  env      - 重新打包 env/speech_train_all-env.tar.gz(wheelhouse 完整离线环境)
  package  - 重新压缩为 speech_train_preset.zip(无 symlink,纯真实内容)
  all      - 顺序跑完所有 stages(默认)

用法:
  python3 scripts/build_asset_package.py [STAGE ...]
  python3 scripts/build_asset_package.py all
  python3 scripts/build_asset_package.py code model
  python3 scripts/build_asset_package.py package
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path


# ============================================================
# 路径与常量
# ============================================================
ROOT = Path(__file__).resolve().parents[1]  # speech_train_preset/
PRESET_PARENT = ROOT.parent                  # voc_train/
OUTPUT_ZIP = PRESET_PARENT / "speech_train_preset.zip"

ALG_DIR = ROOT / "alg" / "speech_train_all-alg"
ENV_DIR = ROOT / "env" / "speech_train_all-env"
DATASET_DIR = ROOT / "dataset" / "speech_train_sample-dataset"
MODEL_DIR = ROOT / "model"
REAL_CPU_MODELS = MODEL_DIR / "real_cpu_models"

DATASET_ZIP = DATASET_DIR / "speech_train_sample-dataset.zip"
ENV_TAR = ENV_DIR / "speech_train_all-env.tar.gz"
BASE_MODEL_ZIP = MODEL_DIR / "speech_train_base-model" / "model.zip"

# 9 个独立 model 资产的 backend 名 -> 真实权重目录名 映射
BACKEND_WEIGHT_MAP = {
    "asr_vosk_small":          "vosk-model-small-en-us-0.15",
    "asr_sherpa_paraformer":   "sherpa-onnx-paraformer-zh-small-2024-03-09",
    "asr_whisper_tiny":        "openai-whisper-tiny",
    "mt_opus_mt_en_zh":        "Helsinki-NLP-opus-mt-en-zh",
    "si_minilm_intent":        "sentence-transformers-all-MiniLM-L6-v2",
    "sv_ecapa_voxceleb":       "speechbrain-spkrec-ecapa-voxceleb",
    "ne_deepfilternet3":       "DeepFilterNet3",
    "ne_rnnoise":              "rnnoise",
    "sc_pycorrector":          "pycorrector-package-resources",
}

# 颜色(简化)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def info(msg: str):
    print(f"{BLUE}[INFO]{RESET}  {msg}")


def ok(msg: str):
    print(f"{GREEN}[OK]{RESET}    {msg}")


def warn(msg: str):
    print(f"{YELLOW}[WARN]{RESET}  {msg}")


def fail(msg: str):
    print(f"{RED}[FAIL]{RESET}  {msg}")


def section(title: str):
    print()
    print(f"{GREEN}{'=' * 60}{RESET}")
    print(f"{GREEN}  {title}{RESET}")
    print(f"{GREEN}{'=' * 60}{RESET}")


def hr():
    print(f"{BLUE}{'-' * 60}{RESET}")


def fmt_size(num_bytes: int) -> str:
    """字节数转人类可读格式。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


# ============================================================
# Stage 1: code - 校验算法工程
# ============================================================
def stage_code(quiet: bool = False) -> bool:
    section("[1/5] 校验算法工程 alg/speech_train_all-alg/")
    required = [
        ALG_DIR / "INFO.ini",
        ALG_DIR / "README.md",
        ALG_DIR / "PROFILE.jpg",
        ALG_DIR / "main" / "config.yaml",
        ALG_DIR / "main" / "train.sh",
        ALG_DIR / "main" / "eval.sh",
        ALG_DIR / "main" / "convert.sh",
        ALG_DIR / "main" / "service.sh",
        ALG_DIR / "main" / "src" / "train.py",
        ALG_DIR / "main" / "src" / "evaluate.py",
        ALG_DIR / "main" / "src" / "speech_tasks.py",
        ALG_DIR / "main" / "src" / "backends.py",
        ALG_DIR / "main" / "haikit" / "__init__.py",
        ALG_DIR / "main" / "haikit" / "haikit" / "__init__.py",
        ALG_DIR / "main" / "haikit" / "haikit" / "context.py",
        ALG_DIR / "main" / "haikit" / "haikit" / "reporting.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        fail(f"算法工程缺失关键文件: {[str(p.relative_to(ROOT)) for p in missing]}")
        return False

    # INFO.ini 中 name 字段
    import configparser
    parser = configparser.ConfigParser()
    parser.read(ALG_DIR / "INFO.ini", encoding="utf-8")
    info_name = parser.get("base", "name", fallback="")
    if info_name != "speech_train_all-alg":
        fail(f"INFO.ini name 应为 speech_train_all-alg,实际: {info_name}")
        return False

    # config.yaml 4 节
    try:
        import yaml
    except ImportError:
        warn("PyYAML 未安装,跳过 config.yaml 详细校验")
    else:
        cfg = yaml.safe_load((ALG_DIR / "main" / "config.yaml").read_text(encoding="utf-8"))
        for phase in ["train", "evaluate", "adapt", "service"]:
            if phase not in cfg:
                fail(f"config.yaml 缺 {phase} 节")
                return False

    # 校验 4 个 .sh 可执行
    import stat
    for sh in ["train.sh", "eval.sh", "convert.sh", "service.sh"]:
        mode = (ALG_DIR / "main" / sh).stat().st_mode
        if not (mode & stat.S_IXUSR):
            fail(f"{sh} 不可执行,请 `chmod +x`")
            return False

    ok(f"算法工程完整(INFO.ini={info_name},config.yaml 4 节齐全,4 脚本可执行)")
    return True


# ============================================================
# Stage 2: model - 取消 symlink + 真实权重
# ============================================================
def stage_model(quiet: bool = False) -> bool:
    section("[2/5] 校验 9 个独立 model 资产 + 复制真实权重")

    # 重要:不再删除 real_cpu_models 中的非 json 文件,保持"中央仓库"完整
    # 9 个独立 model 资产以"副本"形式存在,real_cpu_models 是"权威源"
    # 这样后续 re-pull / 验证时仍可从 real_cpu_models 重新分发

    for short in BACKEND_WEIGHT_MAP:
        asset_dir = MODEL_DIR / f"{short}-cpu-model"
        link_path = asset_dir / "model" / short
        real_path = REAL_CPU_MODELS / BACKEND_WEIGHT_MAP[short]

        if not asset_dir.exists():
            fail(f"独立 model 资产不存在: {asset_dir.relative_to(ROOT)}")
            return False

        # 校验 real_cpu_models 中确有真实权重(防止之前被误删)
        if not real_path.exists():
            fail(f"真实权重源不存在: {real_path.relative_to(ROOT)}")
            info("请先运行: python3 scripts/cache_open_source_cpu_models.py")
            return False

        # 检查 real_cpu_models 中是否真的有非 json 权重文件
        has_weights = any(
            f.is_file() and f.suffix.lower() not in {".json", ".md", ".txt"}
            for f in real_path.rglob("*")
        )
        if not has_weights:
            fail(f"real_cpu_models/{real_path.name} 中没有真实权重(只有 json 索引)")
            info("Vosk 等预置模型可能已被清空,需重新下载")
            return False

        # 删除现有(可能是 symlink 或损坏的副本)
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            if link_path.is_dir():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.parent.mkdir(parents=True, exist_ok=True)

        # 复制真实内容(从 real_cpu_models 复制,包含全部文件)
        # symlinks=False 确保 symlink 被解析为真实文件
        shutil.copytree(real_path, link_path, symlinks=False)

        # 校验复制结果有真实内容
        size = sum(f.stat().st_size for f in link_path.rglob("*") if f.is_file())
        if size < 1024:  # 至少 1KB
            fail(f"复制后 model/{short}/ 太小({size} 字节),可能复制失败")
            return False
        ok(f"  {asset_dir.name}: model/{short}/ 已就位 ({fmt_size(size)})")
    ok(f"9 个独立 model 资产已就位(model/<name>/ 全为真实内容,无 symlink)")
    return True


# ============================================================
# Stage 3: dataset - 重新打包 dataset zip
# ============================================================
def stage_dataset(quiet: bool = False) -> bool:
    section("[3/5] 重新打包 dataset/speech_train_sample-dataset.zip")

    # 校验 audio + labels 完整
    audio_dir = DATASET_DIR / "audio"
    labels_dir = DATASET_DIR / "labels"
    if not (audio_dir / "train").exists() or not (audio_dir / "val").exists() or not (audio_dir / "test").exists():
        fail("dataset/audio/ 缺 train/val/test 划分")
        return False
    if not all((labels_dir / f"{s}.jsonl").exists() for s in ["train", "val", "test"]):
        fail("dataset/labels/ 缺 train/val/test.jsonl")
        return False
    metadata = DATASET_DIR / "metadata.json"
    if not metadata.exists():
        fail("dataset/metadata.json 缺失")
        return False

    # 重新打 zip
    if DATASET_ZIP.exists():
        DATASET_ZIP.unlink()

    tmpdir = Path(f"/tmp/dataset_repack_{os.getpid()}_{int(time.time())}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        # 准备 zip 根(不嵌入 dataset/ 前缀,zip 内直接是 audio/ labels/ metadata.json)
        # 符合《数据集规范》要求:解压后可直接作为数据根目录使用
        shutil.copytree(audio_dir, tmpdir / "audio", symlinks=False)
        shutil.copytree(labels_dir, tmpdir / "labels", symlinks=False)
        shutil.copy(metadata, tmpdir / "metadata.json")

        # 在 zip 根目录打 zip
        import zipfile as zf
        with zf.ZipFile(DATASET_ZIP, "w", compression=zf.ZIP_DEFLATED) as zfobj:
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    full = Path(root) / fname
                    rel = full.relative_to(tmpdir)
                    zfobj.write(full, rel.as_posix())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size = DATASET_ZIP.stat().st_size
    ok(f"dataset zip 重新打包完成: {DATASET_ZIP.relative_to(ROOT)} ({fmt_size(size)})")
    return True


# ============================================================
# Stage 4: env - 重新打包 wheelhouse tar.gz
# ============================================================
def stage_env(quiet: bool = False) -> bool:
    section("[4/5] 重新打包 env/speech_train_all-env.tar.gz(wheelhouse)")

    wheels_dir = ENV_DIR / "wheels"
    if not wheels_dir.exists() or not any(wheels_dir.glob("*.whl")):
        warn("wheels/ 目录为空或缺失,需要先下载依赖")
        info("执行: pip download --dest wheels/ --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-backends.txt")
        return False

    # 校验必要文件
    required = [
        ENV_DIR / "Dockerfile",
        ENV_DIR / "ENVIRONMENT_MANIFEST.json",
        ENV_DIR / "INFO.ini",
        ENV_DIR / "requirements-backends.txt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        fail(f"env 目录缺关键文件: {[str(p.relative_to(ROOT)) for p in missing]}")
        return False

    # 重新打 tar.gz
    if ENV_TAR.exists():
        ENV_TAR.unlink()

    tmpdir = Path(f"/tmp/env_repack_{os.getpid()}_{int(time.time())}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        # 准备 tar 根
        tar_root_name = "speech_train_all-env"
        tar_root = tmpdir / tar_root_name
        tar_root.mkdir()

        # 复制 wheels
        shutil.copytree(wheels_dir, tar_root / "wheels", symlinks=False)

        # 复制构建材料
        for fname in [
            "Dockerfile",
            "Dockerfile.cpu",
            "requirements-cpu-open-source.txt",
            "requirements-backends.txt",
            "ENVIRONMENT_MANIFEST.json",
            "INFO.ini",
            "README.md",
            "PROFILE.jpg",
        ]:
            src = ENV_DIR / fname
            if src.exists():
                shutil.copy(src, tar_root / fname)

        with tarfile.open(ENV_TAR, "w:gz") as tar:
            tar.add(tar_root, arcname=tar_root_name)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size = ENV_TAR.stat().st_size
    ok(f"env tar.gz 重新打包完成: {ENV_TAR.relative_to(ROOT)} ({fmt_size(size)})")
    return True


# ============================================================
# Stage 5: package - 重新打总 zip
# ============================================================
def stage_package(quiet: bool = False) -> bool:
    section("[5/5] 重新压缩为 speech_train_preset.zip(无 symlink)")

    if not OUTPUT_ZIP.exists():
        warn(f"{OUTPUT_ZIP.name} 不存在,开始创建")
    else:
        info(f"覆盖 {OUTPUT_ZIP.name} ...")

    # 关键修复:zip 写入期间会读当前目录,可能把目标 zip 自身递归包含。
    # 先写到 /tmp 临时文件,再 mv 到目标位置,避免递归。
    tmp_zip = Path(f"/tmp/speech_train_preset_{os.getpid()}_{int(time.time())}.zip.tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()

    # 不排除任何东西,完整打包(包括 env tar.gz / dataset zip / model.zip 都是平台需要的):
    #   - env/speech_train_all-env/speech_train_all-env.tar.gz  平台侧离线安装需要完整环境包
    #   - dataset/speech_train_sample-dataset/speech_train_sample-dataset.zip  数据集压缩包
    #   - model/speech_train_base-model/model.zip  规范要求的模型压缩包
    #   - wheels/*.tar.gz  sdist,虽然 pip 主要用 .whl,保留不影响功能
    cmd = ["zip", "-qry9", "-y", str(tmp_zip), "speech_train_preset/"]
    info(f"执行: {' '.join(cmd)} (在 {PRESET_PARENT}, 目标 {tmp_zip})")
    result = subprocess.run(cmd, cwd=PRESET_PARENT, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"zip 命令失败 (rc={result.returncode})")
        if result.stderr:
            fail(f"stderr: {result.stderr}")
        if tmp_zip.exists():
            tmp_zip.unlink()
        return False

    # 移动到目标位置
    shutil.move(str(tmp_zip), str(OUTPUT_ZIP))

    # 验证 zip 无 symlink
    result = subprocess.run(
        ["unzip", "-l", str(OUTPUT_ZIP)],
        capture_output=True,
        text=True,
    )
    symlink_count = sum(1 for line in result.stdout.splitlines() if " -> " in line)

    # 统计 zip 内的关键嵌套包(env tar.gz / dataset zip / model.zip)
    env_tar = sum(1 for line in result.stdout.splitlines() if "speech_train_all-env.tar.gz" in line and "wheels/" not in line)
    dataset_zip = sum(1 for line in result.stdout.splitlines() if "speech_train_sample-dataset.zip" in line and "wheels/" not in line)
    model_zip = sum(1 for line in result.stdout.splitlines() if line.endswith(" model.zip") and "wheels/" not in line)

    size = OUTPUT_ZIP.stat().st_size
    ok(f"总 zip 重新打包完成: {OUTPUT_ZIP.name} ({fmt_size(size)})")
    if symlink_count == 0:
        info("zip 中 0 个 symlink ✅")
    else:
        warn(f"zip 中检测到 {symlink_count} 个 symlink")

    info(f"嵌套包清单(平台需要):")
    info(f"  env tar.gz  : {env_tar} 个(离线环境包)")
    info(f"  dataset zip : {dataset_zip} 个(数据集压缩包)")
    info(f"  model zip   : {model_zip} 个(基础模型压缩包)")
    return True

    # 验证 zip 无 symlink
    result = subprocess.run(
        ["unzip", "-l", str(OUTPUT_ZIP)],
        capture_output=True,
        text=True,
    )
    symlink_count = sum(1 for line in result.stdout.splitlines() if " -> " in line)

    # 统计 zip 内的关键嵌套包(env tar.gz / dataset zip / model.zip)
    env_tar = sum(1 for line in result.stdout.splitlines() if "speech_train_all-env.tar.gz" in line and "wheels/" not in line)
    dataset_zip = sum(1 for line in result.stdout.splitlines() if "speech_train_sample-dataset.zip" in line and "wheels/" not in line)
    model_zip = sum(1 for line in result.stdout.splitlines() if line.endswith("model.zip") and "wheels/" not in line)

    size = OUTPUT_ZIP.stat().st_size
    ok(f"总 zip 重新打包完成: {OUTPUT_ZIP.name} ({fmt_size(size)})")
    if symlink_count == 0:
        info("zip 中 0 个 symlink ✅")
    else:
        warn(f"zip 中检测到 {symlink_count} 个 symlink")

    info(f"嵌套包清单(平台需要):")
    info(f"  env tar.gz  : {env_tar} 个(离线环境包)")
    info(f"  dataset zip : {dataset_zip} 个(数据集压缩包)")
    info(f"  model zip   : {model_zip} 个(基础模型压缩包)")
    return True


# ============================================================
# Stage runner
# ============================================================
STAGES = {
    "code":    ("校验算法工程",                                 stage_code),
    "model":   ("取消 symlink + 复制真实权重到 9 个 model 资产", stage_model),
    "dataset": ("重新打包 dataset zip",                          stage_dataset),
    "env":     ("重新打包 env wheelhouse tar.gz",                stage_env),
    "package": ("重新压缩总 zip",                                stage_package),
}


def main():
    parser = argparse.ArgumentParser(
        description="语音训练资产包端到端构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help=f"要执行的 stage,可多选: {', '.join(STAGES.keys())} 或 all",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="跳过最后的 validate_asset_package.py 调用",
    )

    args = parser.parse_args()

    # 解析 stages
    if "all" in args.stages:
        stage_names = list(STAGES.keys())
    else:
        stage_names = []
        for s in args.stages:
            if s not in STAGES:
                fail(f"未知 stage: {s} (合法: {', '.join(STAGES.keys())} 或 all)")
                sys.exit(2)
            stage_names.append(s)

    info(f"将按顺序执行 stages: {stage_names}")
    info(f"工作根: {ROOT}")
    info(f"目标 zip: {OUTPUT_ZIP}")

    start = time.time()
    failed = []
    for name in stage_names:
        title, fn = STAGES[name]
        try:
            stage_ok = fn()
            if not stage_ok:
                failed.append(name)
                # 出错时:code/model 失败停止;dataset/env 失败警告但继续(package 可能仍有效)
                if name in ("code", "model"):
                    fail(f"stage '{name}' 失败且为前置依赖,停止")
                    break
        except Exception as exc:
            import traceback
            fail(f"stage '{name}' 异常: {exc}")
            traceback.print_exc()
            failed.append(name)
            if name in ("code", "model"):
                break

    elapsed = time.time() - start
    section("构建总结")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  成功 stages: {[s for s in stage_names if s not in failed]}")
    if failed:
        print(f"  {RED}失败 stages: {failed}{RESET}")
        sys.exit(1)
    else:
        ok("所有 stages 成功 ✅")

    # 最后跑一次 validate
    if not args.skip_validate:
        hr()
        info("运行 validate_asset_package.py 复核...")
        validate_script = ROOT / "scripts" / "validate_asset_package.py"
        if validate_script.exists():
            result = subprocess.run(
                [sys.executable, str(validate_script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                fail(f"validate 失败: {result.stderr}")
                sys.exit(1)
        else:
            warn("validate_asset_package.py 不存在,跳过")

    sys.exit(0)


if __name__ == "__main__":
    main()
