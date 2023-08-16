# import re
from add_score import add_score
from chatgpt_improved import da, default_rate

import warnings
warnings.filterwarnings('ignore')


for amount, seed in [
    (None, None)
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "chatgpt-da": lambda summ, doc: da(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=2,    # their heuristic. their default: 0
                retries=None,
                count_default_share=False,
                default_value=0),
            "chatgpt-da_Retries": lambda summ, doc: da(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=6,    # their heuristic (but default ignored till final retry.)
                retries=5,
                count_default_share=False,
                default_value=0),
            "chatgpt-da_SumStep": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=5,    # my simple heuristic.
                # Take the first word. Check if it contains a number, to remove stuff like ! and "".
                # Return it.  If not use default.
                retries=None,
                count_default_share=False,
                default_value=0),
            "chatgpt-da_SumStep_Retries": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=5,
                retries=5,
                count_default_share=False,
                default_value=0),
            "chatgpt-da_SumStep_TheirHeuristic": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=2,
                retries=None,
                count_default_share=False,
                default_value=0),
            "chatgpt-da_SumStep_Retries_TheirHeuristic": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=6,
                retries=5,
                count_default_share=False,
                default_value=0),

            "chatgpt-da_D1": lambda summ, doc: da(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=2,    # their heuristic. their default: 0
                retries=None,
                count_default_share=False,
                default_value=100),
            "chatgpt-da_Retries_D1": lambda summ, doc: da(
                summ, doc, sum_res=False,
                verbose=[],
                interpret_res=6,    # their heuristic (but default ignored till final retry.)
                retries=5,
                count_default_share=False,
                default_value=100),
            "chatgpt-da_SumStep_D1": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=5,    # my simple heuristic.
                # Take the first word. Check if it contains a number, to remove stuff like ! and "".
                # Return it.  If not use default.
                retries=None,
                count_default_share=False,
                default_value=100),
            "chatgpt-da_SumStep_Retries_D1": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=5,
                retries=5,
                count_default_share=False,
                default_value=100),
            "chatgpt-da_SumStep_TheirHeuristic_D1": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=2,
                retries=None,
                count_default_share=False,
                default_value=100),
            "chatgpt-da_SumStep_Retries_TheirHeuristic_D1": lambda summ, doc: da(
                summ, doc, sum_res=True,
                verbose=[],
                interpret_res=6,
                retries=5,
                count_default_share=False,
                default_value=100),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )

print(f"The default counts are: {default_rate}")
