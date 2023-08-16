# This is an adaption of the file final_result.ipynb from https://github.com/Liyan06/AggreFact/blob/main/final_result.ipynb

from newsroom.analyze import Fragments
from blanc import BlancHelp, BlancTune
from blanc import Shannon
from disco_score import DiscoScorer
from bert_score import BERTScorer
import re
from blanc import Estime
from ctc_score import SummarizationScorer

import evaluate

from add_score import add_score
# from add_score_batched import add_score as add_score_batched
# from chatgpt_fiddle import zs, cot, my_cot, da, star, default_rate

import warnings
warnings.filterwarnings('ignore')


scorers = {}


def add_scorer(name, scorer):
    if single_scorer_cache:
        # print("clear scorers")
        scorers.clear()
    scorers[name] = scorer
    # print("initialised", name)


def grusky(summ, doc):
    fragments = Fragments(summ, doc)
    return str({
        "Coverage":    fragments.coverage(),
        "Density": fragments.density(),
        "Compression": fragments.compression(),
    })


def blanc(summ, doc, metric_type, device='cpu'):
    if metric_type == "help":
        if "blanc_help" not in scorers.keys():
            add_scorer("blanc_help", BlancHelp(device=device, show_progress_bar=False))
        return scorers["blanc_help"].eval_once(doc, summ)

    elif metric_type == "tune":
        if "blanc_tune" not in scorers.keys():
            add_scorer("blanc_tune", BlancTune(device=device, finetune_mask_evenly=False, show_progress_bar=False))
        return scorers["blanc_tune"].eval_once(doc, summ)

    else:
        raise ValueError("Unknown type for Blanc")


def blanc_batched(summs, docs, metric_type, device='cpu'):
    if metric_type == "help":
        if "blanc_help" not in scorers.keys():
            add_scorer("blanc_help", BlancHelp(device=device, show_progress_bar=False))
        return scorers["blanc_help"].eval_pairs(docs, summs)

    elif metric_type == "tune":
        if "blanc_tune" not in scorers.keys():
            add_scorer("blanc_tune", BlancTune(device=device, finetune_mask_evenly=False, show_progress_bar=False))
        return scorers["blanc_tune"].eval_pairs(docs, summs)

    else:
        raise ValueError("Unknown type for Blanc")


def disco_score(summ, doc, metric_type, truncate, use_longformer):
    # "For DS-FOCUS, we use Conpono (Iter et al., 2020) that finetuned BERT with a novel discourse-level objective
    # regarding sentence ordering. For DS-SENT, we use BERT-NLI.
    # This is because we find this configuration performs best after initial trials"

    if use_longformer:
        model_name = 'allenai/longformer-base-4096'
        if 'disco_scorers_' + model_name not in scorers.keys():
            add_scorer('disco_scorers_' + model_name, DiscoScorer(
                # device='cpu',
                device='cuda:0',
                model_name='allenai/longformer-base-4096',
                truncation=truncate,
            ))
    elif metric_type.startswith("SENT"):
        model_name = 'conpono/'
        if 'disco_scorers_' + model_name not in scorers.keys():
            add_scorer('disco_scorers_' + model_name, DiscoScorer(
                device='cuda:0',
                model_name='conpono/',
                truncation=truncate,
            ))
    elif metric_type.startswith("FOCUS"):
        model_name = 'MNLI_BERT/'
        if 'disco_scorers_' + model_name not in scorers.keys():
            add_scorer('disco_scorers_' + model_name, DiscoScorer(
                device='cuda:0',
                model_name='MNLI_BERT/',
                truncation=truncate,
            ))
    else:
        ValueError("Unknown type for disco_score")

    disco_scorer = scorers['disco_scorers_' + model_name]

    summ = summ.lower()
    doc = doc.lower()

    try:
        if metric_type == "SENT_NN":
            res = disco_scorer.DS_SENT_NN(summ, [doc])
        elif metric_type == "SENT_Entity":
            res = disco_scorer.DS_SENT_Entity(summ, [doc])
        elif metric_type == "FOCUS_NN":
            res = disco_scorer.DS_Focus_NN(summ, [doc])
        elif metric_type == "FOCUS_ENTITY":
            res = disco_scorer.DS_Focus_Entity(summ, [doc])
        else:
            ValueError("Unknown type for disco_score")
        return res
    except RuntimeError as e:
        # raise e
        print("error disco")
        return 0


def bert_score(summ, doc):
    # Model	    Best Layer	"WMT16 To-English Pearson Correlation"	Rank	Max Length
    # facebook/bart-large-mnli	11	0.7532	7	1022
    # allenai/longformer-large-4096-finetuned-triviaqa	14	0.7366	21	4094
    # microsoft/deberta-v3-base	9	0.7262	24	1000000000000000019884624838654

    if "bert_scorer" not in scorers.keys():
        add_scorer("bert_scorer", BERTScorer(lang="en", rescale_with_baseline=True))

    summ = re.sub(r'\s+', ' ', summ)
    doc = re.sub(r'\s+', ' ', doc)
    p, r, f1 = scorers["bert_scorer"].score([summ], [doc])
    return str({"Precision": float(p[0]), "Recall": float(r[0]), "F1": float(f1[0])})


def estim(summ, doc, include_all_tokens, metric_type=['alarms', 'soft', 'coherence'], device='cuda'):
    key = "estim_"+str(tuple(metric_type))+str(include_all_tokens)
    if key not in scorers.keys():
        add_scorer(key, Estime(output=metric_type, device=device, include_all_tokens=include_all_tokens))
    estimator = scorers[key]
    # this was only in a problem in blanc V0.3.1 but should be fixed now in version V0.3.2
    # try:
    #     res = estimator.evaluate_claims(doc, [summ])
    # except ZeroDivisionError:
    #     res = [[0] * len(metric_type)]   # set 0 for every metric type as (very naive) default value
    res = estimator.evaluate_claims(doc, [summ])
    res_dict = {}
    for i in range(len(metric_type)):
        if res[0][i] == res[0][i]:
            value = res[0][i]
        else:
            # print(f"-- doc --\n{doc}\n-- summ -\n{summ}\n -- res: {res[0]}")
            value = 0   # set 0 as default if Nan is returned. (x == x is only false if x == NaN)
            # see https://github.com/PrimerAI/blanc/issues/52#issuecomment-1650309244
        res_dict[metric_type[i]] = value
    # print(str(res_dict))
    return str(res_dict)


def ctc(summ, doc, model, aspect, device='cuda'):
    # "The xsum scorers don't support aspect='relevance' for now"
    # https://github.com/tanyuqian/ctc-gen-eval/issues/8
    assert model in ['D-cnndm', 'R-cnndm', 'D-xsum', 'R-xsum']

    doc = " ".join(doc.split())
    summ = " ".join(summ.split())

    if "ctc_"+model not in scorers.keys():
        add_scorer("ctc_"+model, SummarizationScorer(align=model, device=device))
    scorer = scorers["ctc_"+model]

    aspects = ['consistency', 'relevance']
    if aspect is None:
        res = {}
        for a in aspects:
            refs = []
            if a == "relevance":
                refs = [doc]
            res[a] = scorer.score(doc=doc, refs=refs, hypo=summ, aspect=a)
        return str(res)
    else:
        assert aspect in aspects
        refs = []
        if aspect == "relevance":
            refs = [doc]
        return scorer.score(doc=doc, refs=refs, hypo=summ, aspect=aspect)


def shannon_without_wrapper(summ, doc):
    if "sgflm" not in scorers.keys():
        add_scorer("sgflm", Shannon())

    try:
        ll_base, ll_help, ll_full, S, _, _ = scorers["sgflm"].go(doc, summ)
    except:
        # to many \n result in an error. I dont know why.
        # https://github.com/PrimerAI/blanc/issues/51
        # In that case we retry without the newline.
        try:
            ll_base, ll_help, ll_full, S, _, _ = scorers["sgflm"].go(re.sub(r'\n', ' ', doc), summ)
        except:
            # if this preprocessing didn't fix the error, we fall back to 0s as default.
            print(f"---- Shannon error:\n---- Doc ----\n{doc}\n---- Summ ---\n{summ}")
            return str({
                "ShannonScore": 0,
                "InfoDiff": 0,
                "BlancShannon": 0,
            })

    return str({
        "ShannonScore": (ll_help - ll_base) / (ll_full - ll_base),
        "InfoDiff": ll_help - ll_base,
        "BlancShannon": (S[0][1] - S[1][0]) / (S[0][0] + S[0][1] + S[1][0] + S[1][1]),
    })


def mfma(summ, doc):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = "henry931007/mfma"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # "The pair of summary and the corresponding article are concatenated and then fed into the classification model as an input." [Lee2022]


rouge = evaluate.load('rouge')


single_scorer_cache = True


for amount, seed in [
    #(1,0),
    (None, None),
]:
    add_score(
#        "human_allignment_docSumPairs_train_portion_1_0.1_0.5.csv",
#        "human_allignment_docSumPairs_train_portion_1_0.1_0.5_withMetrics.csv",
        "human_allignment_docSumPairs_val_portion_1_0.5_1.csv",
        "human_allignment_docSumPairs_val_portion_1_0.5_1_withMetrics.csv",
        {
            "rouge1": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge1'])['rouge1'],
            "rouge2": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge2'])['rouge2'],
            "rougeL": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rougeL'])['rougeL'],
            "rougeLsum": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rougeLsum'])['rougeLsum'],
            "rouge1_stem": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge1'], use_stemmer=True)['rouge1'],
            "rouge2_stem": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge2'], use_stemmer=True)['rouge2'],
            "rougeL_stem": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rougeL'], use_stemmer=True)['rougeL'],
            "rougeLsum_stem": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rougeLsum'], use_stemmer=True)['rougeLsum'],
            "length": lambda summ, doc: len(summ.split()),
            "negative_length": lambda summ, doc: -len(summ.split()),
            "length_share": lambda summ, doc: len(summ.split()) / len(doc.split()),
            "compression": lambda summ, doc: len(doc.split()) / len(summ.split()),
            # # "rouge1_big_words_3": lambda summ, doc: rouge.compute(predictions=[" ".join([e for e in summ.split() if len(e) > 3])], references=[doc], rouge_types=['rouge1'])['rouge1'],
            # # "rouge1_big_words_4": lambda summ, doc: rouge.compute(predictions=[" ".join([e for e in summ.split() if len(e) > 4])], references=[doc], rouge_types=['rouge1'])['rouge1'],
            # # "rouge1_big_words_5": lambda summ, doc: rouge.compute(predictions=[" ".join([e for e in summ.split() if len(e) > 5])], references=[doc], rouge_types=['rouge1'])['rouge1'],
            # # "rouge1_big_words_6": lambda summ, doc: rouge.compute(predictions=[" ".join([e for e in summ.split() if len(e) > 6])], references=[doc], rouge_types=['rouge1'])['rouge1'],
            "grusky": grusky,
            "blanc_help": lambda summ, doc: blanc(summ, doc, "help", device="cuda"),
            # "blanc_tune": lambda summ, doc: blanc(summ, doc, "tune", device="cuda"),
            "disco_score_SENT_NN_truncate": lambda summ, doc: disco_score(summ, doc, "SENT_NN", truncate=512, use_longformer=False),
            "disco_score_SENT_Entity_truncate": lambda summ, doc: disco_score(summ, doc, "SENT_Entity", truncate=512, use_longformer=False),
            "disco_score_FOCUS_NN_truncate": lambda summ, doc: disco_score(summ, doc, "FOCUS_NN", truncate=512, use_longformer=False),
            "disco_score_FOCUS_ENTITY_truncate": lambda summ, doc: disco_score(summ, doc, "FOCUS_ENTITY", truncate=512, use_longformer=False),
            "disco_score_SENT_NN_longformer": lambda summ, doc: disco_score(summ, doc, "SENT_NN", truncate=3500, use_longformer=True),
            "disco_score_SENT_Entity_longformer": lambda summ, doc: disco_score(summ, doc, "SENT_Entity", truncate=3500, use_longformer=True),
            "disco_score_FOCUS_NN_longformer": lambda summ, doc: disco_score(summ, doc, "FOCUS_NN", truncate=3500, use_longformer=True),
            "disco_score_FOCUS_ENTITY_longformer": lambda summ, doc: disco_score(summ, doc, "FOCUS_ENTITY", truncate=3500, use_longformer=True),
            "BertScore": bert_score,
            "estim_allTokens": lambda summ, doc: estim(summ, doc, include_all_tokens=True, metric_type=['alarms']),
            "estim_notAllTokens": lambda summ, doc: estim(summ, doc, include_all_tokens=False, metric_type=['alarms']),
            "estim": lambda summ, doc: estim(summ, doc, include_all_tokens=True, metric_type=['soft', 'coherence']),
#            "ctc_D-cnndm": lambda summ, doc: ctc(summ, doc,  "D-cnndm", None),
#            "ctc_D-xsum_consistency": lambda summ, doc: ctc(summ, doc, "D-xsum", "consistency"),
            "SGfLM": shannon_without_wrapper,
            # "chatgpt-zs": lambda summ, doc: zs(summ, doc, default_value={"defaultFalse": False, "defaultTrue": True}, count_default_share=False, verbose=False),
            #
            # "chatgpt-cot_simple": lambda summ, doc: cot(
            #     summ, doc, sum_res=False, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=False,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_heuristic": lambda summ, doc: cot(
            #     summ, doc, sum_res=False, try_understand_bevore_sum=False,
            #     # verbose=[1, 2, 3, 4, 5, 7],
            #     verbose=[],
            #     interpret_res=True,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_simple": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=False,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_heuristic": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=True,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_OnNeed_simple": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=True,
            #     verbose=[],
            #     interpret_res=False,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_OnNeed_heuristic": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=True,
            #     verbose=[],
            #     interpret_res=True,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            #
            # "chatgpt-cot_simple_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=False, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=False, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_heuristic_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=False, try_understand_bevore_sum=False,
            #     # verbose=[1, 2, 3, 4, 5, 7],
            #     verbose=[],
            #     interpret_res=True, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_simple_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=False, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_heuristic_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=False,
            #     verbose=[],
            #     interpret_res=True, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_OnNeed_simple_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=True,
            #     verbose=[],
            #     interpret_res=False, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            # "chatgpt-cot_Sum_OnNeed_heuristic_Retries15": lambda summ, doc: cot(
            #     summ, doc, sum_res=True, try_understand_bevore_sum=True,
            #     verbose=[],
            #     interpret_res=True, retries=15,
            #     default_value={"defaultFalse": False, "defaultTrue": True}),
            #
            # # Should I also think about testing the metric without default value for unparsable chatgpt responses.
            # # Atm I am unsure how aggrefact and human eval benchmark handle None entries.
            # "chatgpt-da": lambda summ, doc: da(summ, doc, sum_res=False,
            #                                    # verbose=[1, 2, 7],
            #                                    verbose=[],
            #                                    interpret_res=2,
            #                                    retries=None, count_default_share=False),
            # # # making their metric more robust,
            # # "chatgpt-da_Retries15": lambda summ, doc: da(summ, doc, sum_res=False,
            # #                                              # verbose=[1, 2, 7],
            # #                                              verbose=[8],
            # #                                              interpret_res=6,
            # #                                              retries=15, count_default_share=False),
            # # this replaces their heuristic with a sum step, simple conversion and retries.
            # "chatgpt-da_SumStep_Retries15": lambda summ, doc: da(summ, doc, sum_res=True,
            #                                                      # verbose=[1, 2, 7],
            #                                                      verbose=[6, 9],
            #                                                      interpret_res=5,
            #                                                      retries=15, count_default_share=False),
            # # (mainly for ablation) sum step as replacment for their heuristic, without repetition.
            # "chatgpt-da_SumStep": lambda summ, doc: da(summ, doc, sum_res=True,
            #                                            # verbose=[1, 2, 7],
            #                                            verbose=[6],
            #                                            interpret_res=5,
            #                                            retries=None),
            # "chatgpt-da_SumStep_TheirHeuristic": lambda summ, doc: da(summ, doc, sum_res=True,
            #                                                                   # verbose=[1, 2, 7],
            #                                                                   verbose=[6],
            #                                                                   interpret_res=2,
            #                                                                   retries=None),
            # "chatgpt-da_SumStep_Retries15": lambda summ, doc: da(summ, doc, sum_res=True,
            #                                                      # verbose=[1, 2, 7],
            #                                                      # verbose=[6, 9],
            #                                                      verbose=[],
            #                                                      interpret_res=5,
            #                                                      retries=15, count_default_share=False),
            # # # (mainly for ablation) sum step as replacment for their heuristic, without repetition.
            # # "chatgpt-da_SumStep": lambda summ, doc: da(summ, doc, sum_res=True,
            # #                                            # verbose=[1, 2, 7],
            # #                                            verbose=[6],
            # #                                            interpret_res=5,
            # #                                            retries=None),
            # # "chatgpt-da_SumStep_TheirHeuristic": lambda summ, doc: da(summ, doc, sum_res=True,
            # #                                                                   # verbose=[1, 2, 7],
            # #                                                                   verbose=[6],
            # #                                                                   interpret_res=2,
            # #                                                                   retries=None),
            # # "chatgpt-da_SumStep_Retries15": lambda summ, doc: da(summ, doc, sum_res=True,
            # #                                                     # verbose=[1, 2, 7],
            # #                                                     verbose=[6],
            # #                                                     interpret_res=5,
            # #                                                     retries=15),
            # # "chatgpt-da_SumStep_Retries15_TheirHeuristic": lambda summ, doc: da(summ, doc, sum_res=True,
            # #                                                                            # verbose=[1, 2, 7],
            # #                                                                            verbose=[6],
            # #                                                                            interpret_res=2,
            # #                                                                            retries=15),
            # "chatgpt-star": lambda summ, doc: star(summ, doc, sum_res=False,
            #                                        # verbose=[1, 2, 4, 5],
            #                                        # verbose=[3],
            #                                        verbose=[],
            #                                        interpret_res=1,
            #                                        retries=None,
            #                                        count_default_share=False,
            #                                        ),
            # # "chatgpt-star_Retries15": lambda summ, doc: star(summ, doc, sum_res=False,
            # #                                                  verbose=[1, 2, 4, 5],
            # #                                                  # verbose=[3],
            # #                                                  interpret_res=1,
            # #                                                  retries=15,
            # #                                                  count_default_share=False,
            # #                                                  ),
            # # "chatgpt-star_SumRes": lambda summ, doc: star(summ, doc, sum_res=True,
            # #                                               verbose=[1, 2, 4, 5],
            # #                                               # verbose=[3],
            # #                                               interpret_res=1,
            # #                                               retries=None,
            # #                                               count_default_share=False,
            # #                                               ),
            # "chatgpt-star_SumRes_Retries15": lambda summ, doc: star(summ, doc, sum_res=True,
            #                                                         # verbose=[1, 2, 4, 5],
            #                                                         # verbose=[3],
            #                                                         verbose=[],
            #                                                         interpret_res=1,
            #                                                         retries=15,
            #                                                         count_default_share=False,
            #                                                         ),
        },
        use_caching=False, caching_interval=60, print_timing=True, data_amount=amount, data_selection_seed=seed,
    )

#
#     # add_score_batched(
#     #     d,
#     #     {
#     #         "blanc_help": lambda summs, docs: blanc_batched(summs, docs, "help", device="cuda"),
#     #         "blanc_tune": lambda summs, docs: blanc_batched(summs, docs, "tune", device="cuda"),
#     #     },
#     #     use_caching=False, caching_interval=60, print_timing=True
#     # )

# print(f"The default counts are: {default_rate}")

# # print_scores([# 'length', 'negative_length', 'length_share', 'compression',
# #               # 'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
# #               # 'rouge1_stem', 'rouge2_stem', 'rougeL_stem', 'rougeLsum_stem',
# #               # 'rouge1_big_words_3', 'rouge1_big_words_4', 'rouge1_big_words_5', 'rouge1_big_words_6',
# #               # "my_dae",
# #               # "my_questeval011_fscore", "my_questeval011_precision", "my_questeval011_recall",
# #               # "my_questeval024", "my_summacZS", "my_summacConv", "my_qafacteval",
# #               # "grusky_Compression", "grusky_Density", "grusky_Coverage",
# #               'DAE', 'QuestEval', 'SummaC-ZS', 'SummaC-Conv', 'QAFactEval'])
