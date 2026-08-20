from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def extract_embeddings(args: argparse.Namespace) -> dict:
    data = pd.read_csv(args.data, usecols=["compound_inchikey", "canonical_smiles"])
    data = data.drop_duplicates("compound_inchikey").reset_index(drop=True)
    if args.limit:
        data = data.iloc[: args.limit].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model,
        deterministic_eval=True,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen MoLFormer extraction run")
    device = torch.device("cuda:0")
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    embeddings = None
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            smiles = data["canonical_smiles"].iloc[start : start + args.batch_size].tolist()
            lengths = tokenizer(smiles, add_special_tokens=True, truncation=False, return_length=True)["length"]
            truncated += sum(length > args.max_length for length in lengths)
            inputs = tokenizer(
                smiles,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs).pooler_output.float().cpu().numpy()
            if embeddings is None:
                embeddings = np.empty((len(data), output.shape[1]), dtype=np.float32)
            embeddings[start : start + len(output)] = output
            del inputs, output

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        compound_inchikey=data["compound_inchikey"].to_numpy(dtype="U27"),
        embedding=embeddings,
    )
    manifest = {
        "model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "output": str(output_path.resolve()),
        "compounds": int(len(data)),
        "embedding_shape": list(embeddings.shape),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "truncated_smiles": int(truncated),
        "dtype": "bfloat16 inference; float32 output",
        "device": torch.cuda.get_device_name(device),
        "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "torch": torch.__version__,
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frozen MoLFormer embeddings with bounded GPU memory.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=202)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    manifest = extract_embeddings(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
