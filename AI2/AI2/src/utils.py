"""
utils.py — Core helper modules and execution logs
"""

import os, json, logging
from pathlib import Path
from datetime import datetime

LOG_DIR  = "./models/logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")


def get_logger(name="rag_pipeline"):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch  = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        fh  = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def log_query(query, answer, latency, sources, confidence, variant="standard"):
    return {
        "timestamp":  datetime.utcnow().isoformat(),
        "variant":    variant,
        "query":      query,
        "answer":     answer[:300],
        "confidence": confidence,
        "sources":    sources,
        "latency_s":  latency,
    }
