# src/upload_to_hf.py

from huggingface_hub.errors import HfHubHTTPError
import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, logging as hf_logging

# --- Quiet + progress bars (version-safe) ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
hf_logging.set_verbosity_error()

# ====== EDIT THESE (then run: python src/upload_to_hf.py) ======
DEFAULT_GAME_FOLDER = r"data/3p_clips/Syracuse@Pitt"
DEFAULT_REPO_ID = "pietrobarbera/3-Points-Shot-ACC"
DEFAULT_PRIVATE = False
BATCH_SIZE = 25  # clips per commit (safer than 1 clip/commit)
MAX_RETRIES = 8
# ====================================================================

try:
    # works on newer versions
    from huggingface_hub.utils import disable_progress_bars  # type: ignore
    disable_progress_bars()
except Exception:
    pass

try:
    from huggingface_hub import CommitOperationAdd  # newer versions
except Exception:
    CommitOperationAdd = None  # fallback handled later


def _status_code_from_err(e: Exception):
    # Best-effort extraction of HTTP status code
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def _retry(call, *, what: str):
    """Retry wrapper with exponential backoff + jitter for transient HF errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except HfHubHTTPError as e:
            code = _status_code_from_err(e)
            # Retry on transient / throttling / server overload
            if code in (429, 500, 502, 503, 504) or code is None:
                sleep_s = min(60, (2 ** (attempt - 1))) + random.random()
                print(
                    f"⚠️  {what} failed (status={code}). Retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            # If something else odd happens, do a couple retries too
            sleep_s = min(30, attempt * 2) + random.random()
            print(
                f"⚠️  {what} failed ({type(e).__name__}). Retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue
    raise RuntimeError(f"❌ Giving up after {MAX_RETRIES} retries: {what}")


def upload_game_folder(game_folder: Path, repo_id: str, private: bool = False):
    if not game_folder.exists():
        raise FileNotFoundError(f"Game folder not found: {game_folder}")

    csv_path = game_folder / "clips_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing clips_metadata.csv at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "clip_path" not in df.columns:
        raise ValueError(
            "clips_metadata.csv must contain a 'clip_path' column.")

    mp4_files = sorted(game_folder.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 files found in: {game_folder}")

    api = HfApi()

    # Create dataset repo if needed
    _retry(
        lambda: api.create_repo(
            repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private),
        what="create_repo",
    )

    prefix = f"{game_folder.name}/"
    marker_done = f"{prefix}_UPLOAD_COMPLETE.txt"
    marker_csv = f"{prefix}clips_metadata.csv"

    # List repo once (resume logic)
    repo_files = set(_retry(lambda: api.list_repo_files(
        repo_id=repo_id, repo_type="dataset"), what="list_repo_files"))

    if marker_done in repo_files:
        print(
            f"✅ Game '{game_folder.name}' already marked complete in {repo_id}. Nothing to upload.")
        return

    remote_mp4 = {Path(p).name for p in repo_files if p.startswith(
        prefix) and p.endswith(".mp4")}
    local_mp4 = {p.name for p in mp4_files}
    missing = sorted([p for p in mp4_files if p.name not in remote_mp4])

    print(
        f"Remote already has {len(remote_mp4)}/{len(local_mp4)} clips for {game_folder.name}. Missing: {len(missing)}")

    # Upload/overwrite CSV (cheap)
    _retry(
        lambda: api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo=marker_csv,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"[{game_folder.name}] update metadata",
        ),
        what="upload clips_metadata.csv",
    )

    # Upload missing clips in batches
    if missing:
        if CommitOperationAdd is None:
            # Fallback: older hub version — still batch by sleeping and retrying per file
            print("⚠️ Your huggingface_hub is old (no CommitOperationAdd). "
                  "I strongly recommend: pip install -U huggingface_hub")
            for i, f in enumerate(missing, start=1):
                _retry(
                    lambda f=f: api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f"{prefix}{f.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"[{game_folder.name}] add clip {i}/{len(missing)}",
                    ),
                    what=f"upload_file {f.name}",
                )
        else:
            # Preferred: create_commit with multiple operations per commit
            for start in range(0, len(missing), BATCH_SIZE):
                batch = missing[start: start + BATCH_SIZE]
                ops = [
                    CommitOperationAdd(
                        path_in_repo=f"{prefix}{f.name}", path_or_fileobj=str(f))
                    for f in batch
                ]

                msg = f"[{game_folder.name}] add clips {start+1}-{start+len(batch)} of {len(missing)}"
                _retry(
                    lambda ops=ops, msg=msg: api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=ops,
                        commit_message=msg,
                    ),
                    what=f"create_commit batch {start+1}-{start+len(batch)}",
                )
                # tiny pause reduces hammering the API
                time.sleep(0.5)

    # Upload completion marker LAST (so partial uploads don't cause skipping)
    done_text = f"complete: {game_folder.name}\nclips: {len(local_mp4)}\n"
    _retry(
        lambda: api.upload_file(
            path_or_fileobj=done_text.encode("utf-8"),
            path_in_repo=marker_done,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"[{game_folder.name}] mark complete",
        ),
        what="upload completion marker",
    )

    print("\n✅ Upload complete!")
    print(f"Repo: {repo_id}")
    print(f"Folder: {game_folder.name}/")
    print(f"Total clips: {len(local_mp4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_folder", default=DEFAULT_GAME_FOLDER)
    ap.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    ap.add_argument("--private", action="store_true", default=DEFAULT_PRIVATE)
    args = ap.parse_args()

    upload_game_folder(
        game_folder=Path(args.game_folder),
        repo_id=args.repo_id,
        private=bool(args.private),
    )


if __name__ == "__main__":
    main()
