'''

This script will serve the purpose of formatting the dataset to a more comprehensive appearance.

In this project we use PHEME dataset for research purpose.

'''

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from dataClean import clean_text
import sys

# add path to import rumour veracity converter
sys.path.append("/home/zhangshuhao/projects/ys/Graduate/dataset/PHEME")
from convert_veracity_annotations import convert_annotations


DATASET_ROOT = "/home/zhangshuhao/projects/ys/Graduate/dataset/PHEME/all-rnr-annotated-threads"
OUTPUT_CSV = "/home/zhangshuhao/projects/ys/Graduate/dataset/pheme_formatted.csv"

def calc_retweet_intensity(retweet_count: int, favorite_count: int) -> float:
    # use log fraction to calculate the retweet intensity
    return np.log(min(retweet_count, 10000) + 1) / np.log(10000 + 1)

def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_structure(structure: Dict[str, Any]) -> List[Tuple[str, Optional[str], int]]:
    result: List[Tuple[str, Optional[str], int]] = []

    def walk(node_id: str, children: Any, parent_id: Optional[str], depth: int) -> None:
        result.append((node_id, parent_id, depth))
        if isinstance(children, dict):
            for child_id, grand_children in children.items():
                walk(child_id, grand_children, node_id, depth + 1)

    for root_id, children in structure.items():
        walk(root_id, children, None, 0)
    return result


def iter_threads(topic_dir: str) -> List[str]:
    threads: List[str] = []
    for rumour_flag in ["rumours", "non-rumours"]:
        subdir = os.path.join(topic_dir, rumour_flag)
        if not os.path.isdir(subdir):
            continue
        for thread_id in os.listdir(subdir):
            thread_path = os.path.join(subdir, thread_id)
            if os.path.isdir(thread_path):
                threads.append(thread_path)
    return threads


def parse_tweet_json(tweet_json: Dict[str, Any]) -> Dict[str, Any]:
    if tweet_json is None:
        return {}
    user = tweet_json.get("user") or {}
    return {
        "tweet_id": str(tweet_json.get("id_str") or tweet_json.get("id") or ""),
        "created_at": tweet_json.get("created_at") or "",
        "text_raw": tweet_json.get("text") or "",
        "author_id": str(user.get("id_str") or user.get("id") or ""),
        "author_screen_name": user.get("screen_name") or "",
        "retweet_count": tweet_json.get("retweet_count") or 0,
        "favorite_count": tweet_json.get("favorite_count") or 0,
        "lang": tweet_json.get("lang") or "",
    }


def collect_thread_rows(topic: str, thread_dir: str, rumour_label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    annotation_path = os.path.join(thread_dir, "annotation.json")
    structure_path = os.path.join(thread_dir, "structure.json")
    source_dir = os.path.join(thread_dir, "source-tweets")
    reactions_dir = os.path.join(thread_dir, "reactions")

    annotation = safe_read_json(annotation_path) or {}
    structure = safe_read_json(structure_path) or {}

    # compute veracity labels for rumours
    if rumour_label == "rumour":
        veracity_label = convert_annotations(annotation, string=True)
        veracity_label_id = convert_annotations(annotation, string=False)
    else:
        veracity_label = "true"
        veracity_label_id = 1

    flattened = flatten_structure(structure) if structure else []
    parent_map: Dict[str, Tuple[Optional[str], int]] = {}
    for tweet_id, parent_id, depth in flattened:
        parent_map[str(tweet_id)] = (str(parent_id) if parent_id else None, depth)

    # Load source tweet
    source_tweet_files = []
    if os.path.isdir(source_dir):
        source_tweet_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".json")]

    for st in source_tweet_files:
        tj = safe_read_json(st)
        parsed = parse_tweet_json(tj)
        tweet_id = parsed.get("tweet_id") or os.path.splitext(os.path.basename(st))[0]
        parent_id, depth = parent_map.get(tweet_id, (None, 0))
        row = {
            "topic": topic,
            "rumour_flag": rumour_label,
            "veracity_true": annotation.get("true"),
            "misinformation": annotation.get("misinformation"),
            "category": annotation.get("category"),
            "veracity_label": veracity_label,
            "veracity_label_id": veracity_label_id,
            "thread_id": os.path.basename(thread_dir),
            "tweet_id": tweet_id,
            "parent_tweet_id": parent_id or "",
            "is_source": 1,
            "author_screen_name": parsed.get("author_screen_name", ""),
            "created_at": parsed.get("created_at", ""),
            "text_raw": parsed.get("text_raw", ""),
            "text_clean": clean_text(parsed.get("text_raw", "")),
            "retweet_count": parsed.get("retweet_count", 0),
            "favorite_count": parsed.get("favorite_count", 0),
            "retweet_intensity": calc_retweet_intensity(parsed.get("retweet_count", 0), parsed.get("favorite_count", 0)),
            "reply_count": 0,
            "depth": depth,
            "lang": parsed.get("lang", ""),
        }
        rows.append(row)

    # Load reactions
    if os.path.isdir(reactions_dir):
        for f in os.listdir(reactions_dir):
            if not f.endswith(".json"):
                continue
            rp = os.path.join(reactions_dir, f)
            tj = safe_read_json(rp)
            parsed = parse_tweet_json(tj)
            tweet_id = parsed.get("tweet_id") or os.path.splitext(f)[0]
            parent_id, depth = parent_map.get(tweet_id, (None, 1))
            row = {
                "topic": topic,
                "rumour_flag": rumour_label,
                "veracity_true": annotation.get("true"),
                "misinformation": annotation.get("misinformation"),
                "category": annotation.get("category"),
                "veracity_label": veracity_label,
                "veracity_label_id": veracity_label_id,
                "thread_id": os.path.basename(thread_dir),
                "tweet_id": tweet_id,
                "parent_tweet_id": parent_id or "",
                "is_source": 0,
                "author_screen_name": parsed.get("author_screen_name", ""),
                "created_at": parsed.get("created_at", ""),
                "text_raw": parsed.get("text_raw", ""),
                "text_clean": clean_text(parsed.get("text_raw", "")),
                "retweet_count": parsed.get("retweet_count", 0),
                "favorite_count": parsed.get("favorite_count", 0),
                "retweet_intensity": calc_retweet_intensity(parsed.get("retweet_count", 0), parsed.get("favorite_count", 0)),
                "reply_count": 0,
                "depth": depth,
                "lang": parsed.get("lang", ""),
            }
            rows.append(row)

    return rows


def gather_pheme_to_csv(dataset_root: str, output_csv: str) -> None:
    topics = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    all_rows: List[Dict[str, Any]] = []

    for topic_dirname in topics:
        topic_dir = os.path.join(dataset_root, topic_dirname)
        topic_name = topic_dirname.replace("-all-rnr-threads", "")
        for thread_dir in iter_threads(topic_dir):
            rumour_label = "rumour" if os.path.basename(os.path.dirname(thread_dir)) == "rumours" else "non-rumour"
            rows = collect_thread_rows(topic_name, thread_dir, rumour_label)
            all_rows.extend(rows)

    fieldnames = [
        "topic",
        "rumour_flag",
        "veracity_true",
        "misinformation",
        "category",
        "veracity_label",
        "veracity_label_id",
        "thread_id",
        "tweet_id",
        "parent_tweet_id",
        "is_source",
        "author_screen_name",
        "created_at",
        "text_raw",
        "text_clean",
        "retweet_count",
        "favorite_count",
        "retweet_intensity",
        "reply_count",
        "depth",
        "lang",
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


if __name__ == "__main__":
    gather_pheme_to_csv(DATASET_ROOT, OUTPUT_CSV)
    print(f"Wrote CSV to: {OUTPUT_CSV}")