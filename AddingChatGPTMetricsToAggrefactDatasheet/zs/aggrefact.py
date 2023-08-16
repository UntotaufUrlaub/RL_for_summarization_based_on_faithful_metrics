# import re
from add_score import add_score
from chatgpt_fiddle import zs, cot, my_cot, da, star, default_rate

import warnings
warnings.filterwarnings('ignore')


for amount, seed in [
    (None, None)
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "chatgpt-zs": lambda summ, doc: zs(summ, doc, default_value={"defaultFalse": False, "defaultTrue": True}, count_default_share=True, verbose=True),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )

print(f"The default counts are: {default_rate}")
