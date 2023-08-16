import random
import statistics
from copy import deepcopy

import datasets
import pandas as pd

from data_processing import load_data_rlfhf_axis
from human_alignment.helper import eval_metric_system_level_rating as eval_sl, \
    create_match, eval_metric_example_level as eval_el


def convert_data_to_doc_text_system_format(data, level="system"):
    """"""
    data_set = None
    for d in data.values():
        if len(d) > 0:
            # The values in the 'axis' column have sometimes issues.
            #  They need to be cleaned or if not used, as here, to be removed.
            cleaned_data = []
            for e in d:
                cleaned_data.append({
                    'info': e['info'],
                    'summary': {
                        'text': e['summary']['text'],
                        'policy': e['summary']['policy'],
                    },
                })

            temp_set = datasets.Dataset.from_pandas(pd.DataFrame(data=cleaned_data))
            temp_set = temp_set.map(lambda examples:
                                    {
                                        'system': [s['policy'] for s in examples['summary']] if level == 'system' else
                                                  [examples['info'][i]['post'] + examples['summary'][i]['text'] for i in range(len(examples['summary']))],
                                        'text': [s['text'] for s in examples['summary']],
                                        'doc': [i['post'] for i in examples['info']],
                                    }
                                    , batched=True, remove_columns=temp_set.column_names)
            if data_set is None:
                data_set = temp_set
            else:
                data_set = datasets.concatenate_datasets([data_set, temp_set])
    return datasets.Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())


eval_cache_ex_lev = {
    "convert_data_to_doc_text_system_rating_format": {},
    "data_load_function": {},
    "data_convert_function": {}
}


def get__convert_data_to_doc_text_system_rating_format__function(axis_name, level, cache):
    """"""
    def convert_data_to_doc_text_system_rating_format(data):
        if cache:
            key = str(data)
            if key in eval_cache_ex_lev["convert_data_to_doc_text_system_rating_format"]:
                return eval_cache_ex_lev["convert_data_to_doc_text_system_rating_format"][key]

        data_set = None
        for d in data.values():
            if len(d) > 0:
                cleaned_data = []
                for e in d:
                    if axis_name in e['summary']['axes']:
                        cleaned_data.append({
                            'info': e['info'],
                            'summary': {
                                'text': e['summary']['text'],
                                'policy': e['summary']['policy'],
                                axis_name: int(e['summary']['axes'][axis_name]),
                            },
                            # the worker is included to perform the detection of duplicated entries correctly.
                            #  one summary-doc pair might be included more than once with the same rating,
                            #  but performed by different persons. This should not be removed.
                            #  If one person rates the same tupel twice, with the same res, it will be ignored.
                            'worker_performing_the_rating': e['worker']
                        })

                temp_set = datasets.Dataset.from_pandas(pd.DataFrame(data=cleaned_data))
                temp_set = temp_set.map(lambda examples:
                                        {
                                            'system':
                                                [s['policy'] for s in examples['summary']] if level == 'system' else
                                                [examples['info'][i]['post'] + examples['summary'][i]['text'] for i in range(len(examples['summary']))],
                                            'text': [s['text'] for s in examples['summary']],
                                            'doc': [i['post'] for i in examples['info']],
                                            'human_rating': [s[axis_name] for s in examples['summary']],
                                            'worker_performing_the_rating': examples['worker_performing_the_rating']
                                        }
                                        , batched=True, remove_columns=temp_set.column_names)
                if data_set is None:
                    data_set = temp_set
                else:
                    data_set = datasets.concatenate_datasets([data_set, temp_set])
        res = datasets.Dataset.from_pandas(pd.DataFrame(data_set).drop_duplicates())
        # print("rlfhf axes len doc_text_system_rating", len(res))

        if cache:
            eval_cache_ex_lev["convert_data_to_doc_text_system_rating_format"][key] = res

        return res

    return convert_data_to_doc_text_system_rating_format


def eval_metric_system_level(metric_functions, data_split):
    return eval_sl(
        metric_functions,
        data_split,
        load_data_rlfhf_axis,
        convert_data_to_doc_text_system_format,
        get__convert_data_to_doc_text_system_rating_format__function('overall', 'system'),
    )


def eval_metric_example_level(metric_functions, data_split, data_portion, cache=True):
    if cache:
        def data_load_function(data_split_param, data_portion_param):
            key = str(data_split_param) + str(data_portion_param)
            if key in eval_cache_ex_lev["data_load_function"]:
                return eval_cache_ex_lev["data_load_function"][key]
            res = [[e for batch_values in load_data_rlfhf_axis(data_split_param, data_portion_param).values() for e in batch_values]]
            eval_cache_ex_lev["data_load_function"][key] = res
            return res

        def data_convert_function(data):
            key = str(data)
            if key in eval_cache_ex_lev["data_convert_function"]:
                return eval_cache_ex_lev["data_convert_function"][key]
            res = convert_data_to_doc_text_system_format(data, level='example')
            eval_cache_ex_lev["data_convert_function"][key] = res
            return res
    else:
        def data_load_function(data_split_param, data_portion_param):
            res = [[e for batch_values in load_data_rlfhf_axis(data_split_param, data_portion_param).values() for e in
                    batch_values]]
            return res

        def data_convert_function(data):
            res = convert_data_to_doc_text_system_format(data, level='example')
            return res

    return eval_el(
        metric_functions,
        data_split,
        # because ratings are used here everything can be compared to everything. So the whole graph is connected.
        # No need to split into connected sub graphs. Just wrap into the expected list of sub graphs.
        # So here this list has only one element, which contains all the information about the graph
        data_load_function,
        data_convert_function,
        None,
        get__convert_data_to_doc_text_system_rating_format__function('overall', 'example', cache),
        matches=False,
        data_portion=data_portion,
    )
