#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Iterable

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check W&B run status.")
    p.add_argument("--entity", default="pedroafonsomalheiros30-aaa")
    p.add_argument("--project", default="codeforge-asm")
    p.add_argument("--run", default="")
    p.add_argument("--runs", default="")
    p.add_argument("--latest", type=int, default=3)
    return p.parse_args()


def normalize_ids(one: str, many: str) -> list[str]:
    if one.strip():
        return [one.strip()]
    if many.strip():
        return [x.strip() for x in many.split(",") if x.strip()]
    return []


def format_run(run) -> dict:
    attrs = getattr(run, "_attrs", {}) or {}
    return {
        "id": run.id,
        "name": run.name,
        "state": getattr(run, "state", None),
        "lastHistoryStep": getattr(run, "lastHistoryStep", None),
        "createdAt": attrs.get("createdAt"),
        "heartbeatAt": attrs.get("heartbeatAt"),
        "url": run.url,
    }


def print_runs(api: wandb.Api, entity: str, project: str, run_ids: Iterable[str]) -> None:
    for rid in run_ids:
        path = f"{entity}/{project}/{rid}"
        try:
            run = api.run(path)
            print(json.dumps(format_run(run), ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"id": rid, "error": str(exc)}, ensure_ascii=False))


def print_latest(api: wandb.Api, entity: str, project: str, n: int) -> None:
    if n <= 0:
        return
    print(json.dumps({"latest": n, "project": f"{entity}/{project}"}, ensure_ascii=False))
    for run in api.runs(f"{entity}/{project}", per_page=n):
        print(json.dumps(format_run(run), ensure_ascii=False))


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    run_ids = normalize_ids(args.run, args.runs)
    if run_ids:
        print_runs(api, args.entity, args.project, run_ids)
    print_latest(api, args.entity, args.project, args.latest)


if __name__ == "__main__":
    main()

