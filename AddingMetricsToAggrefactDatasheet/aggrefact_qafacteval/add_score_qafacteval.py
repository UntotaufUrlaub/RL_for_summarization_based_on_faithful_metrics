from add_score import add_score
from qafacteval import QAFactEval


lerc = True # False
kwargs = {"cuda_device": 0, "use_lerc_quip": lerc, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}

# kwargs = {"cuda_device": 0, "use_lerc_quip": lerc, \
#         "verbose": True, "generation_batch_size": 1, \
#         "answering_batch_size": 1, "lerc_batch_size": 1}


model_folder = "models"     # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)


for amount, seed in [
    (1, 0),
    (40, 0),
]:
    add_score(
        "aggre_fact_final.csv",
        "aggre_fact_timebenchmark_wip.csv",
        {
            "my_qafacteval": lambda summ, doc: metric.score_batch_qafacteval([doc], [[summ]], return_qa_pairs=False)[0][0]['qa-eval']['lerc_quip'],
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )
