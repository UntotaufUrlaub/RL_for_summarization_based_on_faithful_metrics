from questeval.questeval_metric import QuestEval
from add_score import add_score
import warnings
warnings.filterwarnings('ignore')


questeval = QuestEval(task="summarization", do_weighter=True,
                      isCuda=True,
                      #qg_batch_size=16,
                      #clf_batch_size=16
                      )


def my_questeval_v011(summ, doc):
    # doc = doc.replace("\n", " ")
    # summ = summ.replace("\n", " ")

    score = questeval.compute_all(summ, doc)
    return str(score['scores'])


for amount, seed in [
#    (1, 0),
#    (40, 0),
    (None, None),
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
#        "aggre_fact_timebenchmark_wip.csv",
        {
            "my_questeval011": lambda summ, doc: my_questeval_v011(summ, doc),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )
