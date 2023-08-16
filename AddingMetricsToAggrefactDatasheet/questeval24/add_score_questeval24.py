import sys

from questeval.questeval_metric import QuestEval
from add_score import add_score
import warnings
warnings.filterwarnings('ignore')


questeval = QuestEval(use_cache=False,
                      #qg_batch_size=30,
                      #clf_batch_size=30,
                      )


def my_questeval_v024(summs, docs):
    scores = questeval.corpus_questeval(hypothesis=summs, sources=docs)
    return scores['ex_level_scores']


for amount, seed in [
    # (1, 0),
    # (40, 0),
    (None, None),
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "my_questeval024_batched": lambda summs, docs: my_questeval_v024(summs, docs),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )
