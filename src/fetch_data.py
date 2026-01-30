# src/fetch_data.py
import os
import json
import requests
import argparse

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"


def fetch_json(url: str):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_game_data(game_id: str, save_dir: str = "data/metadata"):
    game_dir = os.path.join(save_dir, game_id)
    os.makedirs(game_dir, exist_ok=True)

    url = f"{SUMMARY_URL}?event={game_id}"
    print(f"Fetching: {url}")
    data = fetch_json(url)

    out_path = os.path.join(game_dir, "pbp.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    comps = data.get("header", {}).get("competitions", [])
    if comps:
        teams = [c["team"]["displayName"]
                 for c in comps[0].get("competitors", [])]
        print(f"Game {game_id}: {teams}")
    else:
        print(f"Game {game_id}: (teams not found)")

    print(f"Saved metadata JSON to {out_path}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--espn_id", required=True)
    parser.add_argument("--save_dir", default="data/metadata")
    args = parser.parse_args()
    fetch_game_data(args.espn_id, save_dir=args.save_dir)
