# This is an adaption of the file final_result.ipynb from https://github.com/Liyan06/AggreFact/blob/main/final_result.ipynb
import pandas as pd
import numpy as np
import sklearn
from utils import choose_best_threshold
from utils import SOTA, XFORMER, OLD, MAPPING
from utils import resample_balanced_acc

import warnings
warnings.filterwarnings('ignore')


# atm werden bei thresholds gegen das human label erzeugt. Sollte man das auch anpassen bei other_target_system?
# und wenn wie? es gibt halt nur das als val set.
def print_scores(systems, file_path, other_target_system=None, boolean_columns=None, rename=None):
    # read data where
    # 1. duplication has been removed.
    # 2. examples that annotated in different datasets with different factual consistency labels are manually corrected based on our judgment.
    df = pd.read_csv(file_path)

    if boolean_columns:
        for c in boolean_columns:
            c_name = c+"_score"
            print(f"Convert {c_name} from boolean to int")
            df[c_name] = df[c_name].apply(lambda val: 1 if val else 0)

    if rename:
        print("renaming")
        # print(df.columns.values)
        rename_adjusted = {}
        for k, v in rename.items():
            rename_adjusted[k+"_score"] = v + "_score"
        df.rename(columns=rename_adjusted, inplace=True)
        # print(df.columns.values)
        for i in range(len(systems)):
            if systems[i] in rename.keys():
                systems[i] = rename[systems[i]]

    # split data
    df_val = df[df.cut == 'val']
    df_val_sota = df_val[df_val.model_name.isin(SOTA)]
    df_test = df[df.cut == 'test']
    df_test_sota = df_test[df_test.model_name.isin(SOTA)]

    dataset_list = ['XSumFaith', 'Polytope', 'FactCC', 'SummEval', 'FRANK', 'Wang20', 'CLIFF', 'Goyal21', 'Cao22']
    origins = ['cnndm', 'xsum']

    main_df = pd.DataFrame(
        columns=['system', 'origin', 'count', 'dataset', 'category', 'bl_acc']
    )

    results = []

    if other_target_system is not None:
        # wir überschreiben einfach die huma label im df_test und df_test_sota mit den target labeln.
        # unsere target label werden vorher erstellt
        # in dem analog zum restlichen code die scores in ein boolean umgewandelt werden.
        # Dazu wird auf dem val set der beste threshold gegen das human label ermittelt.
        # Also die inter metrik accuracy ist nicht ganz unabhängig von den human labeln.
        for origin in origins:
            for dataset in dataset_list:
                for i, model_novelty in enumerate([SOTA, XFORMER, OLD]):
                    df_val_temp = df_val[
                        (df_val.dataset == dataset) & (df_val.origin == origin) & (
                            df_val.model_name.isin(model_novelty))]
                    df_test_temp = df_test[(df_test.dataset == dataset) & (df_test.origin == origin) & (
                        df_test.model_name.isin(model_novelty))]

                    if len(df_val_temp) > 0 and len(df_test_temp) > 0:
                        best_thresh, best_f1 = choose_best_threshold(df_val_temp.label.values,
                                                                     df_val_temp[f'{other_target_system}_score'].values)

                        scores_test = df_test_temp[f'{other_target_system}_score'].values
                        preds_test = [1 if score > best_thresh else 0 for score in scores_test]
                        df_test.loc[df_test_temp.index, f'label'] = preds_test

        for origin in origins:
            df_val_temp = df_val_sota[(df_val_sota.origin == origin)]
            df_test_temp = df_test_sota[(df_test_sota.origin == origin)]

            best_thresh, best_f1 = choose_best_threshold(df_val_temp.label.values,
                                                         df_val_temp[f'{other_target_system}_score'].values)

            scores_test = df_test_temp[f'{other_target_system}_score'].values
            preds_test = [1 if score > best_thresh else 0 for score in scores_test]
            df_test_sota.loc[df_test_temp.index, f'label'] = preds_test

    for system in systems:
        df[f'{system}_label'] = None

    for system in systems:
        for origin in origins:
            for dataset in dataset_list:
                for i, model_novelty in enumerate([SOTA, XFORMER, OLD]):
                    df_val_temp = df_val[
                        (df_val.dataset == dataset) & (df_val.origin == origin) & (df_val.model_name.isin(model_novelty))]
                    df_test_temp = df_test[(df_test.dataset == dataset) & (df_test.origin == origin) & (
                        df_test.model_name.isin(model_novelty))]

                    if len(df_val_temp) > 0 and len(df_test_temp) > 0:
                        best_thresh, best_f1 = choose_best_threshold(df_val_temp.label.values,
                                                                     df_val_temp[f'{system}_score'].values)

                        scores_test = df_test_temp[f'{system}_score'].values
                        preds_test = [1 if score > best_thresh else 0 for score in scores_test]
                        df.loc[df_test_temp.index, f'{system}_label'] = preds_test

                        balanced_acc = sklearn.metrics.balanced_accuracy_score(df_test_temp.label.values, preds_test)

                        main_df.loc[len(main_df.index)] = [
                            system, origin, len(preds_test), dataset, MAPPING[i], balanced_acc
                        ]

                        results.append({"system": system, "dataset_name": dataset, 'origin': origin,
                                        'count': len(scores_test), 'cat': MAPPING[i], "labels": df_test_temp.label.values,
                                        "preds": preds_test, "scores": scores_test})

    # df = df.reindex(
    #     columns=['dataset', 'origin', 'id', 'doc', 'summary', 'model_name', 'label',
    #        'cut', 'DAE_score', 'DAE_label', 'QuestEval_score', 'QuestEval_label',
    #        'SummaC-ZS_score', 'SummaC-ZS_label', 'SummaC-Conv_score', 'SummaC-Conv_label',
    #        'QAFactEval_score' , 'QAFactEval_label'],
    # )

    # Table 8
    print("Table 8:")
    print("# Dataset-wise comparsion between factuality metrics. (Since DAE is trained with human annotated data from XsumFaith, we remove DAE for a fair comparison)")
    main_df_pivot_bacc = main_df.pivot(index=['origin', 'dataset', 'category', 'count'], columns='system', values='bl_acc')
    main_df_pivot_bacc = main_df_pivot_bacc.reindex(columns=systems)
    main_df_pivot_bacc.round(3)
    print(main_df_pivot_bacc.to_string())


    # Table 4
    print("\nTable 4, cnndm:")
    print("# Weighted evaluation (balanced accuracy) on AGGREFACT-CNN and AGGREFACT-XSUM across factuality systems (threshold-per-dataset setting). Note that a baseline that simply predict all examples as factually (in)consistent can reach a balanced accuracy of 50%. Since DAE was trained on the human-annotated XSumFaith data (Goyal and Durrett, 2021) that includes summaries generated from XFORMER and OLD, we exclude these summaries for a fair comparison.")
    scores = []
    for cat in MAPPING.values():
        score = []
        for system in systems:
            system_df = main_df[(main_df.system == system) & (main_df.category == cat) & (main_df.origin == 'cnndm')]
            value = sum(system_df['count'] * system_df['bl_acc']) / sum(system_df['count'])
            score.append(round(value, 3))
        scores.append(score)

    weighted_df = pd.DataFrame(
        scores,
        columns=systems,
        index=['SOTA', 'XFORMER', 'OLD']
    )
    print(weighted_df.to_string())


    # Table 4
    print("\nTable 4, xsum:")
    print("# Weighted evaluation (balanced accuracy) on AGGREFACT-CNN and AGGREFACT-XSUM across factuality systems (threshold-per-dataset setting). Note that a baseline that simply predict all examples as factually (in)consistent can reach a balanced accuracy of 50%. Since DAE was trained on the human-annotated XSumFaith data (Goyal and Durrett, 2021) that includes summaries generated from XFORMER and OLD, we exclude these summaries for a fair comparison.")
    scores = []
    for cat in MAPPING.values():
        score = []
        for system in systems:
            system_df = main_df[(main_df.system == system) & (main_df.category == cat) & (main_df.origin == 'xsum')]
            value = sum(system_df['count'] * system_df['bl_acc']) / sum(system_df['count'])
            score.append(round(value, 3))
        scores.append(score)

    weighted_df = pd.DataFrame(
        scores,
        columns=systems,
        index=['SOTA', 'XFORMER', 'OLD']
    )
    print(weighted_df.to_string())


    # Table 5
    print("\nTable 5:")
    print("#  Balanced binary accuracy using a single threshold on the SOTA subset (single-threshold setting). We show 95% confidence intervals.")

    main_sota_df = pd.DataFrame(
        columns=['system', 'origin', 'bl_acc']
    )

    results = []

    for system in systems:
        for origin in origins:
            df_val_temp = df_val_sota[(df_val_sota.origin == origin)]
            df_test_temp = df_test_sota[(df_test_sota.origin == origin)]

            best_thresh, best_f1 = choose_best_threshold(df_val_temp.label.values, df_val_temp[f'{system}_score'].values)

            scores_test = df_test_temp[f'{system}_score'].values
            preds_test = [1 if score > best_thresh else 0 for score in scores_test]

            f1_score = sklearn.metrics.balanced_accuracy_score(df_test_temp.label.values, preds_test)

            main_sota_df.loc[len(main_sota_df.index)] = [
                system, origin, f1_score
            ]

            results.append({"system": system, 'origin': origin, "labels": df_test_temp.label.values,
                            "preds": preds_test, "scores": scores_test})

    # standard deviation may differ due to randomness

    # from https://github.com/tingofurro/summac/
    P5 = 5 / 2  # Correction due to the fact that we are running 2 tests with the same data
    P1 = 1 / 2  # Correction due to the fact that we are running 2 tests with the same data

    for origin in origins:
        sampled_batch_preds = {res["system"]: [] for res in results}

        for res in results:
            if res['origin'] == origin:
                samples = resample_balanced_acc(res["preds"], res["labels"])
                sampled_batch_preds[res["system"]].append(samples)
                low5, high5 = np.percentile(samples, P5), np.percentile(samples, 100 - P5)
                low1, high1 = np.percentile(samples, P1), np.percentile(samples, 100 - P1)
                bacc = sklearn.metrics.balanced_accuracy_score(res["labels"], res["preds"])

                print(res['origin'].center(6), res["system"].center(20), " - %.3f, %.3f" % (bacc, bacc - low5))
        print()


# print_scores([# 'length', 'negative_length', 'length_share', 'compression',
#               # 'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
#               # 'rouge1_stem', 'rouge2_stem', 'rougeL_stem', 'rougeLsum_stem',
#               # 'rouge1_big_words_3', 'rouge1_big_words_4', 'rouge1_big_words_5', 'rouge1_big_words_6',
#               # "my_dae",
#               # "my_questeval011_fscore", "my_questeval011_precision", "my_questeval011_recall",
#               # "my_questeval024", "my_summacZS", "my_summacConv", "my_qafacteval",
#               # "grusky_Compression", "grusky_Density", "grusky_Coverage",
#               'DAE', 'QuestEval', 'SummaC-ZS', 'SummaC-Conv', 'QAFactEval'], "data/aggre_fact_final.csv")
#               # 'DAE', 'QuestEval', 'SummaC-ZS', 'SummaC-Conv', 'QAFactEval'], "data/aggre_fact_final.csv", "DAE")

# print_scores([
#     'DAE', 'QuestEval', 'SummaC-ZS', 'SummaC-Conv', 'QAFactEval',
#     'my_dae_batched',
#     'my_questeval024_batched', 'my_questeval011_fscore',
#     'my_questeval011_precision', 'my_questeval011_recall',
#     'my_summacZS_batched', 'my_summacConv_batched',
#     'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
#     'rouge1_stem', 'rouge2_stem', 'rougeL_stem', 'rougeLsum_stem',
#     'length', 'negative_length', 'length_share', 'compression',
#     'grusky_Coverage', 'grusky_Density', 'grusky_Compression',
#     'blanc_help', 'blanc_tune',
#     'disco_score_SENT_NN_truncate', 'disco_score_SENT_Entity_truncate',
#     'disco_score_FOCUS_NN_truncate', 'disco_score_FOCUS_ENTITY_truncate',
#     'disco_score_SENT_NN_longformer', 'disco_score_SENT_Entity_longformer',
#     'disco_score_FOCUS_NN_longformer', 'disco_score_FOCUS_ENTITY_longformer',
#     'BertScore_Precision', 'BertScore_Recall', 'BertScore_F1',
#     'ctc_D-cnndm_consistency', 'ctc_D-cnndm_relevance', 'ctc_D-xsum_consistency',
#     'SGfLM_ShannonScore', 'SGfLM_InfoDiff', 'SGfLM_BlancShannon',
#     'psudo_perplexity_bert_lefthandContext_normal',
#     'psudo_perplexity_bert_lefthandContext_negated',
#     'psudo_perplexity_bert_fullContext_normal',
#     'psudo_perplexity_bert_fullContext_negated',
#     'psudo_perplexity_bert_lefthandContext_summaryOnly_normal',
#     'psudo_perplexity_bert_lefthandContext_summaryOnly_negated',
#     'psudo_perplexity_bert_fullContext_summaryOnly_normal',
#     'psudo_perplexity_bert_fullContext_summaryOnly_negated',
#     'psudo_perplexity_bert_lefthandContext_swapped_normal',
#     'psudo_perplexity_bert_lefthandContext_swapped_negated',
#     'psudo_perplexity_bert_fullContext_swapped_normal',
#     'psudo_perplexity_bert_fullContext_swapped_negated',
#     'estim_soft', 'estim_coherence',
#     'estim_allTokens_alarms', 'estim_notAllTokens_alarms',
#     'chatgpt-zs_defaultFalse', 'chatgpt-zs_defaultTrue'
# ],
#     "../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res.csv",
#     boolean_columns=['chatgpt-zs_defaultFalse', 'chatgpt-zs_defaultTrue', ],
#     rename={
#         "my_questeval024_batched": "my_questeval024",
#         "my_questeval011_fscore": "my_questeval_fscore",
#         "my_questeval011_precision": "my_questeval_preci",
#         "my_questeval011_recall": "my_questeval_recall",
#         "my_summacZS_batched": "my_summacZS",
#         "my_summacConv_batched": "my_summacConv",
#         "disco_score_SENT_NN_truncate": "disco_S_NN_trunc",
#         "disco_score_SENT_Entity_truncate": "disco_S_Ent_trunc",
#         "disco_score_FOCUS_NN_truncate": "disco_F_NN_trunc",
#         "disco_score_FOCUS_ENTITY_truncate": "disco_F_ENT_trunc",
#         "disco_score_SENT_NN_longformer": "disco_S_NN_long",
#         "disco_score_SENT_Entity_longformer": "disco_S_ENT_long",
#         "disco_score_FOCUS_NN_longformer": "disco_F_NN_long",
#         "disco_score_FOCUS_ENTITY_longformer": "disco_F_ENT_long",
#         "ctc_D-cnndm_consistency": "ctc_CD_consistency",
#         "ctc_D-cnndm_relevance": "ctc_CD_relevance",
#         "ctc_D-xsum_consistency": "ctc_XSUM_consistency",
#         "psudo_perplexity_bert_lefthandContext_normal": "perplex_norm_l_+",
#         "psudo_perplexity_bert_lefthandContext_negated": "perplex_norm_l_-",
#         "psudo_perplexity_bert_fullContext_normal": "perplexity_norm_f_+",
#         "psudo_perplexity_bert_fullContext_negated": "perplexity_norm_f_-",
#         "psudo_perplexity_bert_lefthandContext_summaryOnly_normal": "perplexity_summ_l_+",
#         "psudo_perplexity_bert_lefthandContext_summaryOnly_negated": "perplexity_summ_l_-",
#         "psudo_perplexity_bert_fullContext_summaryOnly_normal": "perplexity_summ_f_+",
#         "psudo_perplexity_bert_fullContext_summaryOnly_negated": "perplexity_summ_f_-",
#         "psudo_perplexity_bert_lefthandContext_swapped_normal": "perplexity_swap_l_+",
#         "psudo_perplexity_bert_lefthandContext_swapped_negated": "perplexity_swap_l_-",
#         "psudo_perplexity_bert_fullContext_swapped_normal": "perplexity_swap_f_+",
#         "psudo_perplexity_bert_fullContext_swapped_negated": "perplexity_swap_f_-",
#         "estim_allTokens_alarms": "estim_alarms_allT",
#         "estim_notAllTokens_alarms": "estim_alarms_notAllT",
#         "chatgpt-zs_defaultFalse": "chatgpt-zs_False",
#         "chatgpt-zs_defaultTrue": "chatgpt-zs_True",
#     }
# )


# print_scores([f"random_{i}" for i in range(100)], "data/aggre_fact_randoms.csv")


# cot_metrics = [f"{s}_{d}" for d in ["defaultFalse", "defaultTrue"] for s in [
#     "chatgpt-cot_simple",
#     "chatgpt-cot_heuristic",
#     "chatgpt-cot_Sum_simple",
#     "chatgpt-cot_Sum_heuristic",
#     "chatgpt-cot_Sum_OnNeed_simple",
#     "chatgpt-cot_Sum_OnNeed_heuristic",
#
#     "chatgpt-cot_simple_Retries15",
#     "chatgpt-cot_heuristic_Retries15",
#     "chatgpt-cot_Sum_simple_Retries15",
#     "chatgpt-cot_Sum_heuristic_Retries15",
#     "chatgpt-cot_Sum_OnNeed_simple_Retries15",
#     "chatgpt-cot_Sum_OnNeed_heuristic_Retries15",
# ]]
# print_scores(cot_metrics, "../../../metric_result_saves/aggrefact_final_ChatGPT/aggre_fact_final_zs_cot.csv",
#              boolean_columns=cot_metrics)

# chatgpt_metrics_bool = [
#     'chatgpt-zs_defaultFalse_score',
#     'chatgpt-zs_defaultTrue_score',
#     'chatgpt-cot_simple_defaultFalse_score',
#     'chatgpt-cot_simple_defaultTrue_score',
#     'chatgpt-cot_heuristic_defaultFalse_score',
#     'chatgpt-cot_heuristic_defaultTrue_score',
#     'chatgpt-cot_Sum_simple_defaultFalse_score',
#     'chatgpt-cot_Sum_simple_defaultTrue_score',
#     'chatgpt-cot_Sum_heuristic_defaultFalse_score',
#     'chatgpt-cot_Sum_heuristic_defaultTrue_score',
#     'chatgpt-cot_Sum_OnNeed_simple_defaultFalse_score',
#     'chatgpt-cot_Sum_OnNeed_simple_defaultTrue_score',
#     'chatgpt-cot_Sum_OnNeed_heuristic_defaultFalse_score',
#     'chatgpt-cot_Sum_OnNeed_heuristic_defaultTrue_score',
#     'chatgpt-cot_simple_Retries15_defaultFalse_score',
#     'chatgpt-cot_simple_Retries15_defaultTrue_score',
#     'chatgpt-cot_heuristic_Retries15_defaultFalse_score',
#     'chatgpt-cot_heuristic_Retries15_defaultTrue_score',
#     'chatgpt-cot_Sum_simple_Retries15_defaultFalse_score',
#     'chatgpt-cot_Sum_simple_Retries15_defaultTrue_score',
#     'chatgpt-cot_Sum_heuristic_Retries15_defaultFalse_score',
#     'chatgpt-cot_Sum_heuristic_Retries15_defaultTrue_score',
#     'chatgpt-cot_Sum_OnNeed_simple_Retries15_defaultFalse_score',
#     'chatgpt-cot_Sum_OnNeed_simple_Retries15_defaultTrue_score',
#     'chatgpt-cot_Sum_OnNeed_heuristic_Retries15_defaultFalse_score',
#     'chatgpt-cot_Sum_OnNeed_heuristic_Retries15_defaultTrue_score']
# chatgpt_metrics_scale = [
#     'chatgpt-da_score', 'chatgpt-da_Retries_score',
#     'chatgpt-da_SumStep_score', 'chatgpt-da_SumStep_Retries_score',
#     'chatgpt-da_SumStep_TheirHeuristic_score',
#     'chatgpt-da_SumStep_Retries_TheirHeuristic_score',
#     'chatgpt-da_D1_score', 'chatgpt-da_Retries_D1_score',
#     'chatgpt-da_SumStep_D1_score',
#     'chatgpt-da_SumStep_Retries_D1_score',
#     'chatgpt-da_SumStep_TheirHeuristic_D1_score',
#     'chatgpt-da_SumStep_Retries_TheirHeuristic_D1_score',
#     'chatgpt-star_score', 'chatgpt-star_Retries_score',
#     'chatgpt-star_SumRes_score', 'chatgpt-star_SumRes_Retries_score',
#     'chatgpt-star_D3_score', 'chatgpt-star_Retries_D3_score',
#     'chatgpt-star_SumRes_D3_score',
#     'chatgpt-star_SumRes_Retries_D3_score', 'chatgpt-star_D5_score',
#     'chatgpt-star_Retries_D5_score', 'chatgpt-star_SumRes_D5_score',
#     'chatgpt-star_SumRes_Retries_D5_score']
# chatgpt_metrics_bool = [s[:-6] for s in chatgpt_metrics_bool]
# chatgpt_metrics_scale = [s[:-6] for s in chatgpt_metrics_scale]
# # print(chatgpt_metrics)
# chatgpt_metrics = []
# chatgpt_metrics.extend(chatgpt_metrics_bool)
# chatgpt_metrics.extend(chatgpt_metrics_scale)
# print_scores(chatgpt_metrics, "../../../metric_result_saves/aggrefact_final_ChatGPT/aggre_fact_final_chatgpt.csv",
#              boolean_columns=chatgpt_metrics_bool)


# print_scores(
#     ["ensemble", "ensemble_negated"],
#     "../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res_ensemble.csv")


# disco_metrics = [
#     'disco_score_SENT_NN_truncate', 'disco_score_SENT_Entity_truncate',
#     'disco_score_FOCUS_NN_truncate', 'disco_score_FOCUS_ENTITY_truncate',
#     'disco_score_SENT_NN_longformer', 'disco_score_SENT_Entity_longformer',
#     'disco_score_FOCUS_NN_longformer', 'disco_score_FOCUS_ENTITY_longformer',
# ]
# df = pd.read_csv("../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res.csv")
# disco_metrics_negated = []
# for c in disco_metrics:
#     name = c + "_negated"
#     df[name + "_score"] = [-v for v in df[c + "_score"]]
#     disco_metrics_negated.append(name)
# df.to_csv("../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res_temp_negatedDiscoScore.csv")
#
# print_scores(
#     disco_metrics_negated,
#     "../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res_temp_negatedDiscoScore.csv",
# )


sgwlm_metrics = [
    'SGfLM_ShannonScore', 'SGfLM_InfoDiff', 'SGfLM_BlancShannon',
]
df = pd.read_csv("../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res.csv")
sgwlm_metrics_negated = []
for c in sgwlm_metrics:
    name = c + "_negated"
    df[name + "_score"] = [-v for v in df[c + "_score"]]
    sgwlm_metrics_negated.append(name)
df.to_csv("../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res_temp_negatedsgwlm.csv")

print_scores(
    sgwlm_metrics_negated,
    "../../../metric_result_saves/aggrefact_final_withoutChatGpt/aggrefact_res_temp_negatedsgwlm.csv",
)