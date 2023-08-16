import math

import pandas as pd


def get_metrics_lookup(d_p_gpt, d_p_rlfhf, d_p_rlfhf_axes, split):
    lookup = {}
    df = pd.read_csv(
        f"human_allignment_docSumPairs_{split}_portion_"
        f"{d_p_gpt}_{d_p_rlfhf}_{d_p_rlfhf_axes}.csv")
    row_dicts = df.to_dict(orient='records')
    for row in row_dicts:
        if row['doc'] not in lookup.keys():
            lookup[row['doc']] = {}

        for k, v in row.items():
            if k not in ["doc", "summary"] and math.isnan(v):
                print(f"metrics_lookup: Replaced NaN with default 0 for \t{k}")
                row[k] = 0

        lookup[row['doc']][row['summary']] = row
    print(f"created lookup {split}")
    return lookup
