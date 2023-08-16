from add_score import add_score
from summac.model_summac import SummaCZS
from summac.model_summac import SummaCConv
import warnings
warnings.filterwarnings('ignore')

# device: ether "cpu" or "cuda"
device = "cuda"

modelZS = SummaCZS(granularity="sentence", model_name="vitc",
                   device=device)

modelConv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e",
                       device=device, start_file="default", agg="mean")


for amount, seed in [
    # (1, 0),
    # (40, 0),
    (None, None),
#    (9307, None),
#    (1, None),
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_final.csv",
        {
            "my_summacZS_batched": lambda summs, docs: modelZS.score(docs, summs)['scores'],
            "my_summacConv_batched": lambda summs, docs: modelConv.score(docs, summs)['scores'],
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )
