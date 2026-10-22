"""Serve the MatGraphRAG API and dashboard.

Usage:
    uv run python scripts/serve.py [--port 8000]
Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("cognee", "litellm", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    uvicorn.run("matgraph.api:app", host=args.host, port=args.port, log_level="info")
