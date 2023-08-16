import random
import statistics
from copy import copy

import datasets
from datasets import Dataset
from datasets import *
import pandas as pd

from data_processing import load_data_rl_from_feedback, load_rlff_data_with_graphsplits_examplelevel, get_id
from human_alignment.helper import create_bw_scores_system_level, correlate
from human_alignment.helper import eval_metric_system_level_matches as eval_sl, create_match, eval_metric_example_level as eval_el


def convert_to_matches_rlff(data, candidates):
    matches = []
    for i in data:
        if len(i['summaries']) != 2:
            raise NotImplementedError("Comparing more than two summaries at once is not supported for this dataset.")
        if i['summaries'][0]['policy'] != i['summaries'][1]['policy']:
            matches.append(
                create_match(
                    i['summaries'][(0 + i['choice']) % 2]['policy'],
                    i['summaries'][(1 + i['choice']) % 2]['policy']
                ))
    return matches


def convert_to_matches_rlff_examplelevel(data, candidates):
    matches = []
    for i in data:
        if len(i['summaries']) != 2:
            raise NotImplementedError("Comparing more than two summaries at once is not supported for this dataset.")
        # im gegensatz zu convert_to_matches_rlff gibts hier keine sachen die nicht mit aufgenommen werden sollten.from
        #  auch summaries der gleichen policy können auf example level verglichen werden.
        matches.append(
            create_match(
                get_id(i, i['summaries'][(0 + i['choice']) % 2]),
                get_id(i, i['summaries'][(1 + i['choice']) % 2])
            ))
    return matches


def convert_feedback_data_to_doc_text_system_format(data):
    """
    extracts unique document-text-system tuples from the comparison based data set.
    (In a comparison based data set each tuple is needed multiple times.
    Each time is part of a match witch compares it.
    As our metrics are not based on direct compares we dont need this pairing.
    Wach element should be present only once, to not have a bias in averaging the ratings.)
    """
    data_set = None
    for d in data.values():
        if len(d) > 0:
            temp_set = Dataset.from_pandas(pd.DataFrame(data=d))
            temp_set = temp_set.map(lambda examples:
                                    {
                                        'system': [s['policy'] for e in examples['summaries'] for s in e],
                                        'text': [s['text'] for e in examples['summaries'] for s in e],
                                        'doc': [e['post'] for e in examples['info'] for _ in range(2)]
                                    }
                                    , batched=True, remove_columns=temp_set.column_names)
            if data_set is None:
                data_set = temp_set
            else:
                data_set = datasets.concatenate_datasets([data_set, temp_set])
    return Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())


def convert_feedback_data_to_doc_text_id_format(data):
    data_set = Dataset.from_pandas(pd.DataFrame(data=data['data']))
    data_set = data_set.map(lambda examples:
                            {
                                # creating of the id has to fit to schema of get_id in data_processing.
                                'system': [examples['info'][i]['title'] + examples['info'][i]['post'] + examples['summaries'][i][j]['text'] for i in range(len(examples['info'])) for j in range(2)],  # used as id.
                                'text': [s['text'] for e in examples['summaries'] for s in e],
                                'doc': [e['post'] for e in examples['info'] for _ in range(2)]
                            }
                            , batched=True, remove_columns=data_set.column_names)
    res = Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())
    # print("rlfhf len doc_text_id", len(res))
    return res


def eval_metric_system_level(metric_functions, data_split):
    return eval_sl(
        metric_functions,
        data_split,
        load_data_rl_from_feedback,
        convert_feedback_data_to_doc_text_system_format,
        convert_to_matches_rlff,
    )


eval_cache_ex_lev = {
    "convert_format": {},
    "load": {},
    "convert_to_matches": {}
}


def eval_metric_example_level(metric_functions, data_split, data_portion, cache=True):
    if cache:
        def load_data(data_split_param, data_portion_param):
            key = str(data_split_param) + str(data_portion_param)
            if key in eval_cache_ex_lev["load"]:
                return eval_cache_ex_lev["load"][key]
            res = load_rlff_data_with_graphsplits_examplelevel(data_split_param, data_portion_param)
            eval_cache_ex_lev["load"][key] = res
            return res

        def convert_format(data_param):
            key = str(data_param)
            if key in eval_cache_ex_lev["convert_format"]:
                return eval_cache_ex_lev["convert_format"][key]
            res = convert_feedback_data_to_doc_text_id_format(data_param)
            eval_cache_ex_lev["convert_format"][key] = res
            return res

        def convert_to_matches(data_param, candidates):
            key = str(data_param)
            if key in eval_cache_ex_lev["convert_to_matches"]:
                return eval_cache_ex_lev["convert_to_matches"][key]
            res = convert_to_matches_rlff_examplelevel(data_param, candidates)
            eval_cache_ex_lev["convert_to_matches"][key] = res
            return res
    else:
        load_data = load_rlff_data_with_graphsplits_examplelevel
        convert_format = convert_feedback_data_to_doc_text_id_format
        convert_to_matches = convert_to_matches_rlff_examplelevel

    return eval_el(
        metric_functions,
        data_split,
        load_data,
        convert_format,
        convert_to_matches,
        None,
        matches=True,
        data_portion=data_portion
    )
