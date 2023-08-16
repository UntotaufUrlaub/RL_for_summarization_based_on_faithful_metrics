import statistics

#import evaluate
import pandas as pd
from human_alignment import dataset_gpt, dataset_rlfhf, dataset_rlfhf_axes

import random


def eval_metric(metric_function, data_split, level,
                data_portion_gpt, data_portion_rlfhf, data_portion_rlfhf_axes):
    """
    data_portion_gpt: mini: 0.1;
    data_portion_rlfhf: mini: 0.005; small: 0.01
    data_portion_rlfhf_axes: mini: 0.01; small: 0.09
    """

    if level == 'example':
        res = [
            dataset_gpt.eval_metric_example_level(metric_function, data_split, data_portion_gpt),
            dataset_rlfhf.eval_metric_example_level(metric_function, data_split, data_portion_rlfhf),
            dataset_rlfhf_axes.eval_metric_example_level(metric_function, data_split, data_portion_rlfhf_axes),
        ]
    else:
        res = [
            dataset_gpt.eval_metric_system_level(metric_function, data_split),
            dataset_rlfhf.eval_metric_system_level(metric_function, data_split),
            dataset_rlfhf_axes.eval_metric_system_level(metric_function, data_split),
            ]
    print("---- combine datasets:")
    print("res list:", res)
    grouped_res_by_metric = {}
    for dataset_res in res:
        for metric_name, metric_res in dataset_res:
            if metric_name not in grouped_res_by_metric:
                grouped_res_by_metric[metric_name] = []
            grouped_res_by_metric[metric_name].append(metric_res)
    print("metrics_grouped:", grouped_res_by_metric)
    for k, v in grouped_res_by_metric.items():
        grouped_res_by_metric[k] = statistics.mean(v)

    return grouped_res_by_metric


# rouge = evaluate.load('rouge')
#
# print("example level correlations", eval_metric({
#     "length": lambda summ, doc: len(summ.split()),
#     "length ratio": lambda summ, doc: len(summ.split()) / len(doc.split()),
#     "random": lambda summ, doc: random.random(),
#     "rouge1": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge1'])['rouge1']
# }, ['val', ], 'example'))
#
# print("system level correlations", eval_metric({
#     "length": lambda summ, doc: len(summ.split()),
#     "length ratio": lambda summ, doc: len(summ.split()) / len(doc.split()),
#     "random": lambda summ, doc: random.random(),
#     "rouge1": lambda summ, doc: rouge.compute(predictions=[summ], references=[doc], rouge_types=['rouge1'])['rouge1']
# }, ['val', ], 'system'))

# print("example level correlations", eval_metric({
#     "length": lambda summ, doc: len(summ.split()),
# }, ['train', ], 'example'))
