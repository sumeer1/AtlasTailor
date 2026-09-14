#!/usr/bin/env python3
"""List or download a public Aliyun Drive share used by the SPATCH portal."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


API = "https://bj21400.api.aliyunfile.com"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["x-share-token"] = token
    request = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def share_token(share_id: str) -> str:
    return post("/v2/share_link/get_share_token", {"share_id": share_id})["share_token"]


def list_items(share_id: str, token: str, parent: str = "root") -> list[dict]:
    marker = ""
    items: list[dict] = []
    while True:
        result = post(
            "/v2/file/list",
            {
                "share_id": share_id,
                "parent_file_id": parent,
                "limit": 100,
                "marker": marker,
                "fields": "name,size,content_type,download_url,file_id,type,parent_file_id",
                "url_expire_sec": 7200,
            },
            token,
        )
        items.extend(result.get("items", []))
        marker = result.get("next_marker", "")
        if not marker:
            return items


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.stat().st_size if output.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=300) as response, output.open(mode) as handle:
        while block := response.read(8 << 20):
            handle.write(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("share_id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    token = share_token(args.share_id)
    items = list_items(args.share_id, token)
    print(json.dumps(items, indent=2))
    if args.output_dir:
        for item in items:
            if item.get("type") != "file":
                raise RuntimeError("Nested share requires explicit handling: " + item["name"])
            url = item.get("download_url")
            if not url:
                raise RuntimeError("No download URL for " + item["name"])
            download(url, args.output_dir / item["name"])


if __name__ == "__main__":
    main()
