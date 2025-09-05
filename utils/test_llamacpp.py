#!/usr/bin/env python3
"""
Quick test script for LlamaCppModel.

Usage examples:
  python utils/test_llamacpp.py \
    --model-path /path/to/llama-3-instruct.Q4_K_M.gguf \
    --prompt "Explain what a quine is in one sentence." \
    --n-ctx 8192 --max-tokens 256

With a system message and chat mode:
  python utils/test_llamacpp.py \
    --model-path /path/to/model.gguf \
    --prompt "List 3 benefits of unit tests." \
    --system-message "You are a concise assistant." \
    --chat-mode \
    --max-tokens 128

Note: Requires optional dependency 'llama-cpp-python'.
Install CPU default: pip install "eureka-ml-insights[llamacpp]"
GPU wheels/source: see README Installation → Optional: Llama.cpp local runtime.
"""

import argparse
import json
import sys

from eureka_ml_insights.models import LlamaCppModel


def main():
    parser = argparse.ArgumentParser(description="Test LlamaCppModel with a single prompt.")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file.")
    parser.add_argument("--prompt", required=True, help="User prompt text.")
    parser.add_argument("--system-message", default=None, help="Optional system message.")
    parser.add_argument("--chat-mode", action="store_true", help="Use chat API with messages.")

    # Runtime/model args
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context length.")
    parser.add_argument("--n-threads", type=int, default=None, help="Threads for inference (optional).")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size (kv cache).")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers (requires CUDA build).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")

    # Generation args
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling.")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty.")

    args = parser.parse_args()

    try:
        model = LlamaCppModel(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            chat_mode=args.chat_mode,
        )
    except ImportError as e:
        print("Error: llama-cpp-python is not installed.")
        print("Install CPU default: pip install 'eureka-ml-insights[llamacpp]'")
        print("For CUDA, see README Installation → Optional: Llama.cpp local runtime.")
        print(str(e))
        sys.exit(1)

    result = model.generate(
        args.prompt,
        system_message=args.system_message,
        previous_messages=None,
    )

    print("=== LlamaCppModel Result ===")
    print(json.dumps({
        "is_valid": result.get("is_valid"),
        "response_time": result.get("response_time"),
        "n_output_tokens": result.get("n_output_tokens"),
    }, indent=2))
    print("\n--- Model Output ---\n")
    print(result.get("model_output"))


if __name__ == "__main__":
    main()

