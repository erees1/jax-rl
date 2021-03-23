import pandas as pd
import re
import os
from collections import defaultdict
import logging

logger = logging.getLogger()


def extract_value(entry, field):
    try:
        search = f"{field} (\d+.\d+)"
        val = float(re.search(search, entry).group(1))
    except AttributeError:
        search = f"{field} (\d+)"
        val = int(re.search(search, entry).group(1))
    return val


def parse_logs(
    fp, stage="Training", fields=["Episode", "Total Steps", "Epsilon", "Loss", "Reward"]
):
    result = defaultdict(list)
    with open(fp, "r") as fh:
        lines = fh.readlines()
    test_result = lines[-1]
    lines = [l for l in lines if stage in l]
    for line in lines:
        for field in fields:
            try:
                val = extract_value(line, field)
                result[field].append(val)
            except AttributeError:
                continue
    
    try:
        test_result = extract_value(test_result, "episodes")
    except AttributeError:
        test_result = None
    return pd.DataFrame(result), test_result


def parse_experiments(fp, stage="Training"):
    runs = os.listdir(fp)
    dfs = []
    trs = []
    for run in runs:
        path = os.path.join(fp, run, "log")
        try:
            df, tr = parse_logs(path, stage=stage)
            df["seed"] = run
            dfs.append(df)
            trs.append(tr)
        except FileNotFoundError:
            logger.info(f"Could not load logs from {path}, skipping...")
            continue
    return pd.concat(dfs, axis=0), trs
