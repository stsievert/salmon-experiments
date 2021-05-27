import pandas as pd
from typing import List

def _cook_next(raw: List[dict]) -> pd.DataFrame:
    rare = list(raw["participant_responses"].values())
    mrare = sum(rare, [])
    mwell = []
    for response in mrare:
        if "target_winner" not in response:
            continue
        query = {k: response.get(k, -999) for k in ["alg_id", "network_delay", "response_time", "timestamp_query_generated"]}
        query["puid"] = response["participant_uid"]
        objs = response["target_indices"]
        l = [o for o in objs if o["label"] == "left"][0]
        h = [o for o in objs if o["label"] == "center"][0]
        r = [o for o in objs if o["label"] == "right"][0]
        winner = [o for o in objs if o["target_id"] == response["target_winner"]][0]
        for k, v in [("head", h), ("left", l), ("right", r), ("winner", winner)]:
            query[f"{k}"] = v["target_id"]
            query[f"{k}_obj"] = v["primary_description"]
        mwell.append(query)
    return pd.DataFrame(mwell)