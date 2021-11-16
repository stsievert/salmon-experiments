from pathlib import Path
import pandas as pd
from warnings import warn


def _get_mturk_participants(f):
    df = pd.read_csv(f)
    codes = df["Answer.surveycode"]
    if codes.nunique() != len(codes):
        msg = "codes.nunique == {} != {} == len(codes)"
        warn(msg.format(codes.nunique(), len(codes)))

    ret = codes.tolist()
    ret = {k.replace(" ", "").replace(":", "").replace("Participant", "").replace("ID", "") for k in ret}
    puidi = "eebc3a746db8e5fb"
    ret = {k.replace(puidi * 2, puidi) for k in ret}
    return ret


def _get_salmon_participants(f):
    df = pd.read_csv(f)
    amts = df.groupby("puid")["datetime_received"].count()
    return dict(amts)

def _get_salmon_response_time(f):
    df = pd.read_csv(f)
    amts = df.groupby("puid")["response_time"].median()
    return dict(amts)

# Bad?
# A2ZMBLKMUYM7JY
if __name__ == "__main__":
    DIR = Path(".")
    _mturks = [_get_mturk_participants(f) for f in DIR.glob("mturk*.csv")]
    mturk = set()
    for m in _mturks:
        mturk.update(m)
    _salmons = [_get_salmon_participants(f"m{i}/responses.csv") for i in [1, 2, 3, 4, 5]]
    salmons = {k: v for exp in _salmons for k, v in exp.items()}

    assert sum(len(s) for s in _salmons) == len(salmons)

    extra = set(mturk) - set(salmons.keys())
 # extra =
 #  {'4488',
 #  '5986',
 #  '97660554',
 #  'A1FIPG8TRLFYXS',
 #  'A2PE5OYTZCN8J9',
 #  'A2ZMBLKMUYM7JY',
 #  'A31JRBKCY75IXZ',
 #  'A3J9F00DASNBWF',
 #  'A3MDU0KP6F46FP',
 #  'A3TG3EHZ5Q0U40',
 #  'A3TKUXUTDX6FBF',
 #  'AL6LOYZG0HQTG',
 #  'COMPL3T3',
 #  'R_3efyIX5p7Vl2rbV'}

    _salmons = [_get_salmon_response_time(f"m{i}/responses.csv") for i in [1, 2, 3, 4, 5]]
    salmons = {k: v for exp in _salmons for k, v in exp.items()}

    junk = {k: v for k, v in salmons.items() if v <= 1}
