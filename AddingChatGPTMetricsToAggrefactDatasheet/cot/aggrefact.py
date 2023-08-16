# import re
from add_score import add_score
from chatgpt_improved import cot, default_rate

import warnings
warnings.filterwarnings('ignore')


for amount, seed in [
    (None, None)
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
#            "chatgpt-cot_simple": lambda summ, doc: cot(
#                summ, doc, sum_res=False, try_understand_bevore_sum=False,
#                verbose=[],
#                interpret_res=False,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
#            "chatgpt-cot_heuristic": lambda summ, doc: cot(
#                summ, doc, sum_res=False, try_understand_bevore_sum=False,
#                # verbose=[1, 2, 3, 4, 5, 7],
#                verbose=[],
#                interpret_res=True,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
#            "chatgpt-cot_Sum_simple": lambda summ, doc: cot(
#                summ, doc, sum_res=True, try_understand_bevore_sum=False,
#                verbose=[],
#                interpret_res=False,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
            "chatgpt-cot_Sum_heuristic": lambda summ, doc: cot(
                summ, doc, sum_res=True, try_understand_bevore_sum=False,
                verbose=[],
                interpret_res=True,
                default_value={"defaultFalse": False, "defaultTrue": True}),
#            "chatgpt-cot_Sum_OnNeed_simple": lambda summ, doc: cot(
#                summ, doc, sum_res=True, try_understand_bevore_sum=True,
#                verbose=[],
#                interpret_res=False,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
            "chatgpt-cot_Sum_OnNeed_heuristic": lambda summ, doc: cot(
                summ, doc, sum_res=True, try_understand_bevore_sum=True,
                verbose=[],
                interpret_res=True,
                default_value={"defaultFalse": False, "defaultTrue": True}),

#            "chatgpt-cot_simple_Retries15": lambda summ, doc: cot(
#                summ, doc, sum_res=False, try_understand_bevore_sum=False,
#                verbose=[],
#                interpret_res=False, retries=15,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
#            "chatgpt-cot_heuristic_Retries15": lambda summ, doc: cot(
#                summ, doc, sum_res=False, try_understand_bevore_sum=False,
#                # verbose=[1, 2, 3, 4, 5, 7],
#                verbose=[],
#                interpret_res=True, retries=15,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
#            "chatgpt-cot_Sum_simple_Retries15": lambda summ, doc: cot(
#                summ, doc, sum_res=True, try_understand_bevore_sum=False,
#                verbose=[],
#                interpret_res=False, retries=15,
#                default_value={"defaultFalse": False, "defaultTrue": True}),
            "chatgpt-cot_Sum_heuristic_Retries15": lambda summ, doc: cot(
                summ, doc, sum_res=True, try_understand_bevore_sum=False,
                verbose=[],
                interpret_res=True, retries=15,
                default_value={"defaultFalse": False, "defaultTrue": True}),
            "chatgpt-cot_Sum_OnNeed_simple_Retries15": lambda summ, doc: cot(
                summ, doc, sum_res=True, try_understand_bevore_sum=True,
                verbose=[],
                interpret_res=False, retries=15,
                default_value={"defaultFalse": False, "defaultTrue": True}),
            "chatgpt-cot_Sum_OnNeed_heuristic_Retries15": lambda summ, doc: cot(
                summ, doc, sum_res=True, try_understand_bevore_sum=True,
                verbose=[],
                interpret_res=True, retries=15,
                default_value={"defaultFalse": False, "defaultTrue": True}),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )

print(f"The default counts are: {default_rate}")
