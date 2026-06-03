#!/usr/bin/env python3
import audioop
import json
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT.parent / "runs" / "real_vosk_asr"
DATA_DIR = RUN_ROOT / "dataset"
MODEL_DIR = ROOT / "model" / "real_cpu_models" / "vosk-model-small-en-us-0.15"

PHRASES = [
    "one zero zero one nine oh two",
    "open the training dashboard",
    "start speech recognition training",
    "check the model evaluation report",
    "deploy the voice correction service",
    "query the speaker verification result",
    "run the noise enhancement test",
    "translate the speech into english",
    "record the meeting transcript",
    "switch to channel number two",
]


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def wer(reference, hypothesis):
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference, hypothesis):
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def synthesize_with_pyttsx3(samples):
    import pyttsx3

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        engine = pyttsx3.init()
        engine.setProperty("rate", 145)
        engine.save_to_file(sample["text"], str(raw_dir / f"{sample['utt_id']}.wav"))
        engine.runAndWait()


def convert_to_16k_mono(src, dst):
    with wave.open(str(src), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    if frame_rate != 16000:
        frames, _ = audioop.ratecv(frames, sample_width, 1, frame_rate, 16000, None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(sample_width)
        writer.setframerate(16000)
        writer.writeframes(frames)


def build_dataset(limit=10):
    if limit > 50:
        raise ValueError("dataset limit must be <= 50")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, phrase in enumerate(PHRASES[:limit], start=1):
        samples.append(
            {
                "utt_id": f"real_vosk_{index:03d}",
                "task_type": "asr",
                "text": phrase,
                "language": "en",
                "speaker_id": "pyttsx3_voice",
                "channel": "tts_mono",
            }
        )

    synthesize_with_pyttsx3(samples)
    for sample in samples:
        raw_path = DATA_DIR / "raw" / f"{sample['utt_id']}.wav"
        audio_path = DATA_DIR / "audio" / "test" / f"{sample['utt_id']}.wav"
        convert_to_16k_mono(raw_path, audio_path)
        sample["audio"] = str(audio_path.relative_to(DATA_DIR))

    labels_dir = DATA_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (labels_dir / "test.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "name": "real_vosk_asr_test",
        "sample_count": len(samples),
        "limit": limit,
        "task_type": "asr",
        "backend": "vosk_small_en_us_0_15",
        "device": "cpu",
        "source": "pyttsx3 offline TTS generated speech",
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return samples


def recognize(sample, model):
    from vosk import KaldiRecognizer

    audio_path = DATA_DIR / sample["audio"]
    with wave.open(str(audio_path), "rb") as reader:
        recognizer = KaldiRecognizer(model, reader.getframerate())
        while True:
            data = reader.readframes(4000)
            if not data:
                break
            recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
    return result.get("text", "")


def run(limit=10):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"missing Vosk model: {MODEL_DIR}")
    samples = build_dataset(limit=limit)

    from vosk import Model, SetLogLevel

    SetLogLevel(-1)
    model = Model(str(MODEL_DIR))
    results = []
    for sample in samples:
        prediction = recognize(sample, model)
        sample_result = {
            **sample,
            "prediction": prediction,
            "wer": wer(sample["text"], prediction),
            "cer": cer(sample["text"], prediction),
        }
        results.append(sample_result)

    report = {
        "backend": "vosk_small_en_us_0_15",
        "device": "cpu",
        "model_dir": str(MODEL_DIR),
        "dataset_dir": str(DATA_DIR),
        "sample_count": len(results),
        "wer": sum(item["wer"] for item in results) / len(results),
        "cer": sum(item["cer"] for item in results) / len(results),
        "results": results,
    }
    report_path = RUN_ROOT / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    print(json.dumps({k: report[k] for k in ["backend", "device", "sample_count", "wer", "cer"]}, indent=2))


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(limit=limit)
