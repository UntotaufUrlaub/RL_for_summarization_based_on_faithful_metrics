# import re
from add_score import add_score
from chatgpt_improved import star, default_rate

import warnings
warnings.filterwarnings('ignore')


for amount, seed in [
    (None, None)
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "chatgpt-star": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,    # 1 heißt, es wird ein default verwendet.
                retries=None,
                count_default_share=False,
                default_value=0
                ),
            "chatgpt-star_Retries": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=0
                ),
            "chatgpt-star_SumRes": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=None,
                count_default_share=False,
                default_value=0
                ),
            "chatgpt-star_SumRes_Retries": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=0
                ),
            # # now default 3
            "chatgpt-star_D3": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,    # 1 heißt, es wird ein default verwendet.
                retries=None,
                count_default_share=False,
                default_value=3
                ),
            "chatgpt-star_Retries_D3": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=3
                ),
            "chatgpt-star_SumRes_D3": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=None,
                count_default_share=False,
                default_value=3
                ),
            "chatgpt-star_SumRes_Retries_D3": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=3
                ),
            # # now default 5
            "chatgpt-star_D5": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,    # 1 heißt, es wird ein default verwendet.
                retries=None,
                count_default_share=False,
                default_value=5
                ),
            "chatgpt-star_Retries_D5": lambda summ, doc: star(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=5
                ),
            "chatgpt-star_SumRes_D5": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=None,
                count_default_share=False,
                default_value=5
                ),
            "chatgpt-star_SumRes_Retries_D5": lambda summ, doc: star(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=1,
                retries=5,
                count_default_share=False,
                default_value=5
                ),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )

print(f"The default counts are: {default_rate}")
