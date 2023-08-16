from qafacteval import QAFactEval

kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}


model_folder = "models"     # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)

doc = "A man was walking down the road."
summ = "A man was walking."

metric.score_batch_qafacteval([doc], [[summ]], return_qa_pairs=False)[0][0]['qa-eval']['lerc_quip']
