import pandas as pd
from typing import List
import numpy as np
from pathlib import Path
from zipfile import ZipFile
import json


_FNAMES = [  # from debug-kevin/triplet_figures_orig/output_embedding.py
    (0, "strangefruit30/i0126.png"),
    (1, "strangefruit30/i0208.png"),
    (2, "strangefruit30/i0076.png"),
    (3, "strangefruit30/i0326.png"),
    (4, "strangefruit30/i0526.png"),
    (5, "strangefruit30/i0322.png"),
    (6, "strangefruit30/i0312.png"),
    (7, "strangefruit30/i0036.png"),
    (8, "strangefruit30/i0414.png"),
    (9, "strangefruit30/i0256.png"),
    (10, "strangefruit30/i0074.png"),
    (11, "strangefruit30/i0050.png"),
    (12, "strangefruit30/i0470.png"),
    (13, "strangefruit30/i0022.png"),
    (14, "strangefruit30/i0430.png"),
    (15, "strangefruit30/i0254.png"),
    (16, "strangefruit30/i0572.png"),
    (17, "strangefruit30/i0200.png"),
    (18, "strangefruit30/i0524.png"),
    (19, "strangefruit30/i0220.png"),
    (20, "strangefruit30/i0438.png"),
    (21, "strangefruit30/i0454.png"),
    (22, "strangefruit30/i0112.png"),
    (23, "strangefruit30/i0494.png"),
    (24, "strangefruit30/i0194.png"),
    (25, "strangefruit30/i0152.png"),
    (26, "strangefruit30/i0420.png"),
    (27, "strangefruit30/i0142.png"),
    (28, "strangefruit30/i0114.png"),
    (29, "strangefruit30/i0184.png"),
]
for k, (idx, fname) in enumerate(_FNAMES):
    assert k == idx
FNAMES = [fname.replace("strangefruit30/", "") for _, fname in _FNAMES]


def __cook_next(raw, byte_keys=False) -> pd.DataFrame:
    try:
        rare = list(raw["participant_responses"].values())
    except:
        rare = list(raw[b"participant_responses"].values())
    mrare = sum(rare, [])
    mwell = []
    for response in mrare:
        if "index_winner" not in response:
            continue
        query = {
            k: response.get(k, -999)
            for k in [
                "alg_label",
                "network_delay",
                "response_time",
                "timestamp_query_generated",
            ]
        }
        query["puid"] = response["participant_uid"]
        objs = response["target_indices"]
        l = [o for o in objs if o["label"] == "left"][0]
        h = [o for o in objs if o["label"] == "center"][0]
        r = [o for o in objs if o["label"] == "right"][0]
        winner = [o for o in objs if o["index"] == response["index_winner"]][0]
        for k, v in [("head", h), ("left", l), ("right", r), ("winner", winner)]:
            query[f"{k}"] = v["index"]
            primary_description = FNAMES[v["index"]]
            query[f"{k}_obj"] = primary_description
            query[f"{k}_smooth"] = int(primary_description.strip("i.png"))
        mwell.append(query)
    df = pd.DataFrame(mwell)
    df["loser"] = df["right"].copy()
    right_wins = df["winner"] == df["right"]
    df.loc[right_wins, "loser"] = df.loc[right_wins, "left"]
    return df


def _process(p: Path) -> pd.DataFrame:
    with ZipFile(p) as zf:
        assert len(zf.filelist) == 1
        with zf.open(zf.filelist[0]) as f:
            raw = json.load(f)

    raw = __cook_next(raw)
    raw = raw.drop(columns=["head", "left", "right"])
    raw = raw.drop(columns=["head_smooth", "left_smooth", "right_smooth"])
    raw = raw.drop(columns=["winner", "winner_smooth", "loser"])
    return raw


if __name__ == "__main__":

    for f in Path("salmon-raw").glob("*.csv"):
        raw = pd.read_csv(f)
        raw = raw.sort_values(by="datetime_received")
        raw = raw.drop(
            columns=[f"{k}_html" for k in ["left", "right", "head", "winner", "loser"]]
        )
        raw = raw.drop(columns=["left", "right", "head", "winner", "loser"])
        raw = raw.drop(
            columns=["start_time", "time_received_since_start", "time_received"]
        )
        raw["timestamp"] = raw["datetime_received"].copy()
        raw = raw.drop(columns=["datetime_received"])
        raw.to_csv(f"salmon-mrare/{f.name}")
    SALMON_COLS = set(raw.columns)

    p = Path("next-fig3.json.zip")
    p_verify = (
        Path("../../orig-next-fig")
        / "fruit_experiment1"
        / "participants.json.zip"
    )

    df = _process(p)
    df_verify = _process(p_verify)
    assert (df == df_verify).all().all()

    df = df.sort_values(by="timestamp_query_generated")
    for alg_label in df.alg_label.unique():
        out = df[df.alg_label == alg_label].copy()
        out = out.sort_values(by="timestamp_query_generated")
        # fmt: off
        assert (out.columns == [
                "alg_label", "network_delay", "response_time",
                "timestamp_query_generated", "puid",
                "head_obj", "left_obj", "right_obj", "winner_obj"]
        ).all()
        out.columns = [
            "alg_ident", "network_latency", "response_time", "timestamp",
            "puid", "head_filename", "left_filename", "right_filename",
            "winner_filename",
        ]
        out["loser_filename"] = out.apply(
            lambda row: row["left_filename"] if row["right_filename"] == row["winner_filename"] else row["right_filename"], axis=1
        )
        # fmt: on
        #  out.columns = ["network_delay",
        out.to_csv(f"next-mrare/{alg_label}.csv")
    NEXT_COLS = set(out.columns)
    assert NEXT_COLS == SALMON_COLS - {"score"}
