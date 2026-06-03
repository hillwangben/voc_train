import argparse
import json

from speech_tasks import evaluate_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--backend", default="cpu_reference")
    args = parser.parse_args()

    result = evaluate_task(args.task_type, args.data_dir, args.model_dir, args.save_dir, args.backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
