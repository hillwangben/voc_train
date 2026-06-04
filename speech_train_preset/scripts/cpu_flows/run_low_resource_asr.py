#!/usr/bin/env python3
from run_cpu_flow import run_flow
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run independent CPU flow for low_resource_asr")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(run_flow("low_resource_asr", args.output_root))


if __name__ == "__main__":
    main()
