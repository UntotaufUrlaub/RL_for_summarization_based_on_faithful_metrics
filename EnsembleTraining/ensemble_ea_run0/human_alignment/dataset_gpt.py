import random
import statistics
from copy import copy

import datasets
import pandas as pd

from data_processing import load_data
from human_alignment.helper import eval_metric_system_level_matches as eval_sl, create_match, \
    eval_metric_example_level as eval_el


def convert_gpt_data_to_doc_text_system_format(data):
    """"""
    systems = ['t0', 'gpt3', 'brio']

    data_set = None
    for d in data.values():
        if len(d) > 0:
            temp_set = datasets.Dataset.from_pandas(pd.DataFrame(data=list(d.values())))
            temp_set = temp_set.map(lambda examples:
                                    # the tuples should match as the order is preserved.
                                    # https://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed
                                    {
                                        'system': [s for s in systems for _ in examples['article']],
                                        'text': [t['text'] for s in systems for t in examples[s]],
                                        'doc': [a for _ in systems for a in examples['article']],
                                    }
                                    , batched=True, remove_columns=temp_set.column_names)
            if data_set is None:
                data_set = temp_set
            else:
                data_set = datasets.concatenate_datasets([data_set, temp_set])
    return datasets.Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())


def convert_gpt_data_to_doc_text_id_format(data):
    systems = ['t0', 'gpt3', 'brio']

    data_set = datasets.Dataset.from_pandas(pd.DataFrame(data=data['data']))
    data_set = data_set.map(
        lambda examples: {
            'system': [examples['article'][i] + examples[s][i]['text'] for s in systems for i in range(len(examples['article']))],
            'text': [t['text'] for s in systems for t in examples[s]],
            'doc': [a for _ in systems for a in examples['article']],
        }
        , batched=True, remove_columns=data_set.column_names)
    res = datasets.Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())
    # print("gpt doc_text_id length", len(res))
    return res


def convert_to_matches(data, candidates):
    matches = []
    for i in data:
        for element in data[i]["annotators"]:
            candidates_temp = copy(candidates)
            best_summaries = copy(element['best_summary'])
            worst_summaries = copy(element['worst_summary'])

            # summaries rated best and worst at the same time are removed/ignored.
            # Rating as best and worst at the same time seems like no information at count.
            # Is this the right handling though?
            to_remove = []
            for s in best_summaries:
                if s in worst_summaries:
                    to_remove.append(s)
            for s in to_remove:
                candidates_temp.remove(s)
                best_summaries.remove(s)
                worst_summaries.remove(s)

            for b_s in best_summaries:
                candidates_temp.remove(b_s)
            for better in best_summaries:
                for s in candidates_temp:
                    matches.append(create_match(better, s))

            for w_s in worst_summaries:
                candidates_temp.remove(w_s)
            for worse in worst_summaries:
                for s in candidates_temp:
                    matches.append(create_match(s, worse))

    return matches


def convert_to_macthes_examplelevel(data, candidates):
    candidates = ['t0', 'gpt3', 'brio']

    matches = []
    if len(data) != 1:
        raise ValueError("unexpected subgraph size")
    for element in data[0]["annotators"]:
        candidates_temp = copy(candidates)
        best_summaries = copy(element['best_summary'])
        worst_summaries = copy(element['worst_summary'])

        # summaries rated best and worst at the same time are removed/ignored.
        # Rating as best and worst at the same time seems like no information at count.
        # Is this the right handling though?
        to_remove = []
        for s in best_summaries:
            if s in worst_summaries:
                to_remove.append(s)
        for s in to_remove:
            candidates_temp.remove(s)
            best_summaries.remove(s)
            worst_summaries.remove(s)

        for b_s in best_summaries:
            candidates_temp.remove(b_s)
        for better in best_summaries:
            for s in candidates_temp:
                matches.append(create_match(
                    data[0]['article'] + data[0][better]['text'],
                    data[0]['article'] + data[0][s]['text']
                ))

        for w_s in worst_summaries:
            candidates_temp.remove(w_s)
        for worse in worst_summaries:
            for s in candidates_temp:
                matches.append(create_match(
                    data[0]['article'] + data[0][s]['text'],
                    data[0]['article'] + data[0][worse]['text']
                ))

    return matches


def eval_metric_system_level(metric_functions, data_split):
    return eval_sl(
        metric_functions,
        data_split,
        load_data,
        convert_gpt_data_to_doc_text_system_format,
        convert_to_matches,
    )


def load_data_example_level(data_split, data_portion):
    data = load_data(data_split, data_portion)
    data_res = []
    for data_part in data.values():
        for doc in data_part.values():
            data_res.append([doc])
    return data_res


eval_cache_ex_lev = {
    "convert_format": {},
    "load": {},
    "convert_to_matches": {}
}


# the gpt data is already organised into connected sub graphs. The summaries are grouped by document.
# All summaries of one document are compared by each annotator, so there is no chance for more graph splits.
def eval_metric_example_level(metric_functions, data_split, data_portion, cache=True):
    if cache:
        def load_data(data_split_param, data_portion_param):
            key = str(data_split_param) + str(data_portion_param)
            if key in eval_cache_ex_lev["load"]:
                return eval_cache_ex_lev["load"][key]
            res = load_data_example_level(data_split_param, data_portion_param)
            eval_cache_ex_lev["load"][key] = res
            return res

        def convert_format(data_param):
            key = str(data_param)
            if key in eval_cache_ex_lev["convert_format"]:
                return eval_cache_ex_lev["convert_format"][key]
            res = convert_gpt_data_to_doc_text_id_format(data_param)
            eval_cache_ex_lev["convert_format"][key] = res
            return res

        def convert_to_matches_cached(data_param, candidates):
            key = str(data_param)
            if key in eval_cache_ex_lev["convert_to_matches"]:
                return eval_cache_ex_lev["convert_to_matches"][key]
            res = convert_to_macthes_examplelevel(data_param, candidates)
            eval_cache_ex_lev["convert_to_matches"][key] = res
            return res
    else:
        load_data = load_data_example_level
        convert_format = convert_gpt_data_to_doc_text_id_format
        convert_to_matches_cached = convert_to_macthes_examplelevel

    return eval_el(
        metric_functions,
        data_split,
        load_data,
        convert_format,
        convert_to_matches_cached,
        None,
        matches=True,
        data_portion=data_portion,
    )
