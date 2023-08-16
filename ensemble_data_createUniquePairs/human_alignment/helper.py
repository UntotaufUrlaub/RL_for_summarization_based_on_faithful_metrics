import random
import statistics
from copy import copy
import datasets
import pandas as pd
from tqdm import tqdm
# import inspect


def create_match(winner, loser):
    return {
        'winner': winner,
        'loser': loser
    }


elo_cache = {}


def stabilised_value_learning(learning_rate, matches, candidates, iterations_for_learning=100, iterations_to_stabilise=10, cache=True):
    """
    The value_learning algorithm is dependent on randomness.
    The effekt of this randomness can probably be controlled by learning_rate, and iterations.
    Taking the mean over iterations_to_stabilise may additionally stabilise the result.
    """
    if cache:
        cache_key = str(matches)
        if cache_key in elo_cache:
            # print("using elo cache")
            return elo_cache[cache_key]

    values = [value_learning(learning_rate, matches, candidates, iterations_for_learning) for _ in range(iterations_to_stabilise)]

    grouped_values = {}
    for entry in values:
        for k, v in entry.items():
            if k not in grouped_values:
                grouped_values[k] = []
            grouped_values[k].append(v)

    for k, v in grouped_values.items():
        grouped_values[k] = statistics.mean(v)

    if cache:
        elo_cache[cache_key] = grouped_values

    return grouped_values


def value_learning(learning_rate, matches, candidates, iterations=100):
    initial_value = 0.5     # todo: what is the right initialisation.

    values = {}
    if candidates is not None:
        for c in candidates:
            values[c] = initial_value

    for i in range(iterations):
        learning_rate_i = learning_rate / (i + 1)
        for m in random.sample(matches, len(matches)):
            for res in random.sample([0, 1], 2):
                if res == 1:
                    key_a = 'winner'
                    key_b = 'loser'
                else:
                    key_a = 'loser'
                    key_b = 'winner'

                if candidates is None:
                    if m['winner'] not in values:
                        values[m['winner']] = initial_value
                    if m['loser'] not in values:
                        values[m['loser']] = initial_value

                # if a winner/loser in the dataset is not known as a candidate an exception will be raised here,
                #  as an unknown candidate lacks initialisation.
                #  This is fine as the candidates are expected to be consistent within one dataset.
                o = {
                    'winner': values[m['winner']] / (1.0 - values[m['winner']]),
                    'loser': values[m['loser']] / (1.0 - values[m['loser']])
                }

                # alpha is the saliency. Means how expected was the given match-outcome.
                alpha = 1.0 - o['winner'] / (o['winner'] + o['loser'])
                # todo: does alpha need any special treatment in some cases?
                #  like mentioned in the paper for none player ever winning before

                values[m[key_a]] += alpha * learning_rate_i * (res - values[m[key_b]])
                # print(m[key_a], values[m[key_a]], 'saliency:', alpha)
        # print(i, values)
    return values


def correlate(score_a_dict, score_b_dict):
    if score_b_dict.keys() != score_b_dict.keys():
        raise AttributeError("Both dicts are supposed to have the same keys."
                             "An identical candidate set is supposed to be evaluated in both groups.")
    x = []
    y = []
    for sys in score_a_dict.keys():
        x.append(score_a_dict[sys])
        y.append(score_b_dict[sys])

    try:
        return statistics.correlation(x, y)
    except statistics.StatisticsError as e:
        return None


def create_bw_scores_system_level(data, candidates, convert):
    matches = []
    for key in data.keys():
        matches.extend(convert(data[key], candidates))

    return stabilised_value_learning(0.5, matches, candidates)


def create_metric_score_system_level(metric_scores):
    metric_scores_system_level = {}
    for sys in metric_scores.keys():
        metric_scores_system_level[sys] = statistics.mean(metric_scores[sys])
    return metric_scores_system_level


human_score_cache = {}


def extract_human_scores_system_level(data, convert_data_to_doc_text_system_rating_format, cache=True):
    data_set = convert_data_to_doc_text_system_rating_format(data)

    if cache:
        cache_key = str(data_set)
        if cache_key in human_score_cache:
            return human_score_cache[cache_key]

    human_scores_grouped_by_system = {}

    filterbar_enabled = datasets.is_progress_bar_enabled()
    if filterbar_enabled:
        datasets.disable_progress_bar()
    systems = data_set.unique("system")
    # for policy in tqdm(systems, desc="grouping human scores"):
    for policy in systems:
        human_scores_grouped_by_system[policy] = data_set.filter(lambda x: x['system'] == policy)['human_rating']
    if filterbar_enabled:
        datasets.enable_progress_bar()

    res = create_metric_score_system_level(human_scores_grouped_by_system)

    if cache:
        human_score_cache[cache_key] = res

    return res


def get_function_add_metric_results(metric_functions):
    def function(examples):
        """
        (This function is supposed to be called by dataset.map)

        iterates over all functions in the dict metric_functions.
        For each it creates and entry in the returned dict,
        which is the list of the metric results created by applying into to every doc-summary-pair.

        (The res dict entries will be added as columns of the dataset)

        :param examples: a batch as list of the elements of the dataset;
            Creating this and passing is usually handled by dataset.map
        :return: a dict with keys (new column names) according to the keys (metric names) of metric_functions and
            values the results of the metrics.
        """
        res = {}
        for name, metric_func in metric_functions.items():
            res[name] = [metric_func(examples['text'][i], examples['doc'][i]) for i in range(len(examples['text']))]
        return res
    return function


def create_metrics_score_system_level(metrics_scores):
    metric_scores_system_level = {}
    for name in metrics_scores.keys():
        metric_scores_system_level[name] = {}
        for sys in metrics_scores[name].keys():
            metric_scores_system_level[name][sys] = statistics.mean(metrics_scores[name][sys])
    return metric_scores_system_level


def eval_metric_system_level_matches(metric_functions, data_split, data_load_function, data_convert_function, convert_to_matches_function, verbose=True, num_proc=44):
    data, metric_scores_system_level = get_data_and_metric_scores(data_convert_function, data_load_function, data_split,
                                                                  metric_functions, verbose, num_proc)

    human_scores_system_level = create_bw_scores_system_level(data, list(metric_scores_system_level[list(metric_scores_system_level.keys())[0]].keys()), convert_to_matches_function)
    if verbose:
        print_human_scores(human_scores_system_level)

    return correlations(human_scores_system_level, metric_functions, metric_scores_system_level)


def eval_metric_system_level_rating(metric_functions, data_split, data_load_function, data_convert_function, convert_data_to_doc_text_system_rating_format):
    data, metric_scores_system_level = get_data_and_metric_scores(data_convert_function, data_load_function,
                                                                  data_split, metric_functions)

    human_scores_system_level = extract_human_scores_system_level(data, convert_data_to_doc_text_system_rating_format)
    print_human_scores(human_scores_system_level)

    return correlations(human_scores_system_level, metric_functions, metric_scores_system_level)


def get_data_and_metric_scores(data_convert_function, data_load_function, data_split, metric_functions, verbose=True, num_proc=44, use_filter=True):
    if verbose:
        print("---- eval")

    data = data_load_function(data_split)
    data_set = data_convert_function(data)
    data_set = data_set.map(get_function_add_metric_results(metric_functions), batched=True, num_proc=1,
                            batch_size=100, desc="calculating the metric scores for each summary-doc pair")

    progress_bar_was_enabled = datasets.is_progress_bar_enabled()
    if progress_bar_was_enabled and not verbose:
        datasets.disable_progress_bar()

    metrics_scores = {}

    for name in metric_functions:
        metric_scores = {}

        if use_filter:
            for policy in tqdm(data_set.unique("system"), desc="group scores for the metric " + name):
                metric_scores[policy] = data_set.filter(lambda x: x['system'] == policy, desc="Filter metric scores by system")[name]
        else:   # this should be faster in the case of a lot of systems as usual during example level correlation.
            def function(example):
                sys = example['system']
                if sys not in metric_scores:
                    metric_scores[sys] = []
                metric_scores[sys].append(example[name])
            data_set.map(function, desc="calc metric scores ["+name+"] for each system/example")

        metrics_scores[name] = metric_scores

    if progress_bar_was_enabled and not verbose:
        datasets.enable_progress_bar()

    metric_scores_system_level = create_metrics_score_system_level(metrics_scores)
    if verbose:
        print("metric_scores_system_level \t", metric_scores_system_level)

    return data, metric_scores_system_level


def print_human_scores(human_scores_system_level):
    print("human_scores_system_level \t\t", human_scores_system_level)
    print("human_scores_system_level sorted \t\t",
          sorted(((v, k) for k, v in human_scores_system_level.items()), reverse=True))


def correlations(human_scores_system_level, metric_functions, metric_scores_system_level):
    return [(name, correlate(metric_scores_system_level[name], human_scores_system_level)) for name in
            metric_functions.keys()]


# todo: maybe: theoretisch kann auch die elo auf dem gesamten datensatz berechnet werden.
#  Die sub graphen beeinflussen sich halt nicht gegenseitig.
#  Nur die correlations müssen dann nach sub graphen gemacht werden.
def eval_metric_example_level(metric_functions, data_split, data_load_function, data_convert_function,
                              convert_to_matches_function, convert_data_to_doc_text_id_rating_format, matches,
                              data_portion, null_cor_for_const=True):
    data = data_load_function(data_split, data_portion)

    data_flat = {'data': [e for els in data for e in els]}
    _, metric_scores = get_data_and_metric_scores(data_convert_function, lambda _: data_flat, data_split, metric_functions, use_filter=False, verbose=False)

    correlations_list_for_each_connected_subgraph = []
    size_of_each_connected_subgraph = []
    for data_part in tqdm(data, desc="calculating the elo for each subgraph and then the correlations for the metrics"):
        if matches:
            human_scores = create_bw_scores_system_level({'data': data_part}, None, convert_to_matches_function)
        else:
            human_scores = extract_human_scores_system_level(
                {'data': data_part},
                convert_data_to_doc_text_id_rating_format
            )

        metrics_scores_for_subgraph = {}
        for metric_name in metric_scores:
            metric_scores_for_subgraph = {}
            for key in human_scores:
                metric_scores_for_subgraph[key] = metric_scores[metric_name][key]
            metrics_scores_for_subgraph[metric_name] = metric_scores_for_subgraph

        correlations_list_for_each_connected_subgraph.append(
            correlations(human_scores, metric_functions, metrics_scores_for_subgraph))
        # to simply perform the mean: insert 1 instead of the graph size
        size_of_each_connected_subgraph.append(len(data_part))

    correlations_for_each_metrik = []
    for name in metric_functions.keys():
        correlations_list = []
        for cor_list in correlations_list_for_each_connected_subgraph:
            for name_id, value in cor_list:
                if name == name_id:
                    correlations_list.append(value)

        weighted_cor_sum = 0
        weight_sum = 0
        for i in range(len(correlations_list)):
            if correlations_list[i] is None:
                if null_cor_for_const:
                    weighted_cor_sum += 0 * size_of_each_connected_subgraph[i]
                    weight_sum += size_of_each_connected_subgraph[i]
                else:
                    pass
            else:
                weighted_cor_sum += correlations_list[i] * size_of_each_connected_subgraph[i]
                weight_sum += size_of_each_connected_subgraph[i]
        correlations_for_each_metrik.append((name, weighted_cor_sum/weight_sum))
    return correlations_for_each_metrik


def eval_metric_example_level_rating(metric_functions, data_split, data_load_function, data_convert_function, convert_data_to_doc_text_system_rating_format):
    data, metric_scores = get_data_and_metric_scores(data_convert_function, data_load_function,
                                                     data_split, metric_functions)

    human_scores_system_level = extract_human_scores_system_level(data, convert_data_to_doc_text_system_rating_format)
    print_human_scores(human_scores_system_level)

    return correlations(human_scores_system_level, metric_functions, metric_scores)
