# src/delete_game_from_hf.py
# Run: python src/delete_game_from_hf.py

from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import HfApi
import time
import random
DEFAULT_REPO_ID = "pietrobarbera/3-Points-Shot-ACC"
GAME_FOLDER_NAME = "Louisville@Cal"   # <-- folder to delete (exactly as on HF)
MAX_RETRIES = 8
BATCH_SIZE = 150  # delete operations per commit (safe)


try:
    from huggingface_hub import CommitOperationDelete
except Exception as e:
    raise RuntimeError(
        "Your huggingface_hub is too old (missing CommitOperationDelete). "
        "Run: pip install -U huggingface_hub"
    ) from e


def _status_code_from_err(e: Exception):
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def _retry(call, what: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except HfHubHTTPError as e:
            code = _status_code_from_err(e)
            if code in (429, 500, 502, 503, 504) or code is None:
                sleep_s = min(60, (2 ** (attempt - 1))) + random.random()
                print(
                    f"⚠️ {what} failed (status={code}). Retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            sleep_s = min(30, attempt * 2) + random.random()
            print(
                f"⚠️ {what} failed ({type(e).__name__}). Retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue
    raise RuntimeError(f"❌ Giving up after {MAX_RETRIES} retries: {what}")


def main():
    api = HfApi()
    repo_id = DEFAULT_REPO_ID
    prefix = f"{GAME_FOLDER_NAME}/"

    files = _retry(lambda: api.list_repo_files(
        repo_id=repo_id, repo_type="dataset"), "list_repo_files")
    to_delete = [p for p in files if p.startswith(prefix)]

    if not to_delete:
        print(f"✅ Nothing to delete: '{prefix}' not found in {repo_id}")
        return

    print(
        f"About to delete {len(to_delete)} files under '{prefix}' in dataset {repo_id}")
    print("This is permanent on the repo history (but you can re-upload).")

    # Delete in batches to avoid huge commits
    for start in range(0, len(to_delete), BATCH_SIZE):
        batch = to_delete[start: start + BATCH_SIZE]
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        msg = f"Delete {GAME_FOLDER_NAME} ({start+1}-{start+len(batch)} of {len(to_delete)})"

        _retry(
            lambda ops=ops, msg=msg: api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=ops,
                commit_message=msg,
            ),
            what=f"create_commit delete batch {start+1}-{start+len(batch)}",
        )

        time.sleep(0.5)

    print(f"\n✅ Deleted folder '{GAME_FOLDER_NAME}/' from {repo_id}")


if __name__ == "__main__":
    main()
