"""Download three official GGUF models, verifying their published SHA256."""
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def download_resumable(repo, revision, filename, size):
    target = ROOT / "models" / filename
    if target.exists() and target.stat().st_size == size:
        return target
    parts = ROOT / "models" / ".parts" / filename
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size
    url = hf_hub_url(repo, filename, revision=revision)
    # Resolve once to avoid making a Hub resolver request for every range.
    response = requests.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    url = response.url

    def chunk(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        path = parts / str(index)
        length = end - start + 1
        if path.exists() and path.stat().st_size == length:
            return length
        for attempt in range(5):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity"}, stream=True, timeout=(15, 60)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError(f"Server did not return requested byte range {start}-{end}")
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as output:
                        for data in r.iter_content(1024 * 1024):
                            output.write(data)
                    if temporary.stat().st_size != length:
                        raise RuntimeError("Incomplete range response")
                    temporary.replace(path)
                return length
            except (requests.RequestException, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Download failed")

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(chunk, index) for index in range(count)]
        for n, future in enumerate(as_completed(futures), 1):
            completed += future.result()
            if n % 16 == 0 or n == count:
                print(f"{filename}: {completed / size:.0%} ({completed / 1e9:.2f}/{size / 1e9:.2f} GB)", flush=True)
    temporary = target.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / str(index)).open("rb") as source:
                for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(data)
    temporary.replace(target)
    return target


def main():
    manifest_path = ROOT / "runtime" / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for size, precision in (("7B", "Q4_K_M"), ("1.1B", "Q4_K_M"), ("1.1B", "F16")):
        repo = f"squeeze-ai-lab/TinyAgent-{size}-GGUF"
        filename = f"TinyAgent-{size}-{precision}.gguf"
        info = HfApi().model_info(repo, files_metadata=True)
        metadata = next(item for item in info.siblings if item.rfilename == filename)
        if not metadata.lfs or not metadata.lfs.get("sha256"):
            raise RuntimeError(f"No published SHA256 found for {filename}")
        expected = metadata.lfs["sha256"]
        print(f"Downloading {filename}...", flush=True)
        path = download_resumable(repo, info.sha, filename, metadata.size)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise RuntimeError(f"Not a GGUF file: {path}")
            handle.seek(0)
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        parts = ROOT / "models" / ".parts" / filename
        if parts.exists():
            for part in parts.iterdir():
                part.unlink()
            parts.rmdir()
        # Fine-tuned weight repositories omit tokenizers; use their base tokenizers.
        tokenizer_repo = ("Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct" if size == "1.1B"
                          else "mistralai/Mistral-7B-v0.1")
        tokenizer_info = HfApi().model_info(tokenizer_repo)
        snapshot_download(tokenizer_repo, revision=tokenizer_info.sha,
                          local_dir=ROOT / "models" / f"tokenizer-{size.lower()}",
                          local_dir_use_symlinks=False,
                          allow_patterns=["tokenizer*", "special_tokens_map.json",
                                          "added_tokens.json", "config.json"])
        manifest[f"{size}-{precision}"] = {"repo": repo, "revision": info.sha, "file": filename,
                          "sha256": expected, "bytes": path.stat().st_size,
                          "tokenizer_repo": tokenizer_repo, "tokenizer_revision": tokenizer_info.sha}
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n")
        print(f"Verified {filename}", flush=True)


if __name__ == "__main__":
    main()
