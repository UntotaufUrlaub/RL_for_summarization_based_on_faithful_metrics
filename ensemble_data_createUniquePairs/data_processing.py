import json
import math
import statistics
from copy import copy
import hashlib
from tqdm import tqdm
import pickle
import os

# todo: maybe: sollte ich die daten iwie cachen, im RAM und/oder in files?


def pseudo_random(text):
    # random_value = hash(text) / sys.maxsize
    # random_value = abs(random_value)
    m = hashlib.sha256()
    m.update(text.encode())
    value = int(m.hexdigest(), 16)
    value_in_range = value / math.pow(2, 256)
    # print(value, value_in_range)
    # print(value_in_range)
    if value_in_range >= 1 or value_in_range < 0:
        raise NotImplementedError("The hash (" + str(value_in_range) + ") is out of range")
    return value_in_range


def load_data(share, data_portion):
    """data_portion: mini: 0.1;"""

    # todo: den rest zuweißen, so dass am ende die summe 1 ergibt
    shares = {'train': 0.6, 'val': 0.2, 'test': 0.2}

    for s in share:
        if s not in shares:
            raise ValueError('This share (' + s + ') is not known! possible options are ' + str(list(shares.keys())))
    if len(share) == 0:
        raise ValueError('The list of passed shares needs to at least contain one element!')

    bounds = []
    if 'train' in share:
        low_bound = 0
        up_bound = shares['train']
        bounds.append((low_bound, up_bound))
    if 'val' in share:
        low_bound = 0 + shares['train']
        up_bound = 0 + shares['train'] + shares['val']
        bounds.append((low_bound, up_bound))
    if 'test' in share:
        low_bound = 0 + shares['train'] + shares['val']
        up_bound = 0 + shares['train'] + shares['val'] + shares['test']
        bounds.append((low_bound, up_bound))

    keys = ['bbc', 'cnn']
    # keyword dataset is ignored as we are not interested in guided summaries atm

    datasets = {}
    for key in keys:
        path = "./human_annotated_data/Era_of_GPT3/human_annotations/" + key + "_human.json"
        with open(path, 'r') as f:
            datasets[key] = json.load(f)

    for data_batch in datasets.values():
        for k in copy(data_batch):
            random_value = pseudo_random(data_batch[k]['article'])

            if data_portion is not None and random_value > data_portion:
                del data_batch[k]
                continue

            keep = False
            for b in bounds:
                if not (random_value < b[0] or random_value >= b[1]):
                    keep = True
            if not keep:
                del data_batch[k]
    return datasets


def load_data_rlfhf_axis(share, data_portion, filter_by_metricmap=None):
    """data_portion: mini: 0.01; small: 0.09"""

    keys = [
        'cnndm1',
        'cnndm3',
        'cnndm4',
        'tldraxis1',
        'tldraxis2',
    ]

    # todo: mapping rausfinden. github issue ausstehend
    share_map = {'train': 'valid2', 'val': 'valid1', 'test': 'test'}

    if len(share) == 0:
        raise ValueError('The list of passed shares needs to at least contain one element!')

    datasets = {}
    for key in keys:
        path = "./human_annotated_data/summarize-from-feedback/axis_evals/" + key + ".json"
        with open(path, 'r') as f:

            data = []
            for json_str in list(f):
                data.append(json.loads(json_str))
            datasets[key] = data

    splits = []
    for s in share:
        splits.append(share_map[s])

    start_size = 0
    end_size = 0
    for batch_key in datasets:
        length = len(datasets[batch_key])
        start_size += length
        for i in range(length):
            k = length - 1 - i  # progress in reverse to not shift the indexes of yet unhandled items in the list,

            # if filter_by_metricmap is not None:
            #     key = 'article' if 'article' in datasets[batch_key][k]['info'].keys() else 'post'
            #     if datasets[batch_key][k]['info'][key] not in filter_by_metricmap.keys() or \
            #             datasets[batch_key][k]['summary']['text'] not in \
            #             filter_by_metricmap[datasets[batch_key][k]['info'][key]].keys():
            #         del datasets[batch_key][k]
            #         continue
            if data_portion is not None:
                key = 'article' if 'article' in datasets[batch_key][k]['info'].keys() else 'post'
                if pseudo_random(datasets[batch_key][k]['info'][key]) > data_portion:
                    del datasets[batch_key][k]
                    continue

            if datasets[batch_key][k]['split'] not in splits:
                del datasets[batch_key][k]
        end_size += len(datasets[batch_key])

    print(share, start_size, end_size, end_size / start_size)
    # anteile: train: 44%; val: 14%; test: 42%
    #   vorsicht das mapping ist noch unklar.

    return datasets
    # todo: muss ich checken ob es duplicate gibt?
    #  also ob ein paar 2mal bewertet wurde (gleiche oder abweichende bewertung).
    #  muss ich das auch an anderen stellen überprüfen?
    #  ich vermute mal das wäre erst relevant wenn es um example level correlation geht.


def load_data_rl_from_feedback(share):
    keys = [
        'batch0_cnndm',
        'batch3',
        'batch4',
        'batch5',
        'batch6',
        'batch7',
        'batch8',
        'batch9',
        'batch10',
        'batch11',
        'batch12',
        'batch13',
        'batch14',
        'batch15',
        'batch16',
        'batch17',
        'batch18',
        'batch19',
        'batch20',
        'batch22',
        'cnndm0',
        'cnndm2', ]

    # this is the mapping according to the readme in the data repo
    share_map = {'train': 'train', 'val': 'valid1', 'test': 'valid2'}

    if len(share) == 0:
        raise ValueError('The list of passed shares needs to at least contain one element!')

    datasets = {}
    for key in keys:
        path = "./human_annotated_data/summarize-from-feedback/comparisons/" + key + ".json"
        with open(path, 'r') as f:
            data = []
            for json_str in list(f):
                data.append(json.loads(json_str))
            datasets[key] = data

    splits = []
    for s in share:
        splits.append(share_map[s])

    start_size = 0
    end_size = 0
    for batch_key in datasets:
        length = len(datasets[batch_key])
        start_size += length
        for i in range(length):
            k = length - 1 - i  # progress in reverse to not shift the indexes of yet unhandled items in the list,
            if datasets[batch_key][k]['split'] not in splits:
                del datasets[batch_key][k]
        end_size += len(datasets[batch_key])
    # the splits are grouped in batches. Aka not every batch has elements of every split.
    # For example there are batches which contain no element of the validation split.
    # todo: is this a problem?
    #  One could check the paper if the splits are representative nevertheless.
    #  One could check if all systems are in all splits. (with similar shares)
    #  One could check if all splits have similar ranges of metric values.

    print(share, start_size, end_size, end_size / start_size)
    # anteile: train: 52%; val: 18%; test: 30%
    # from the analysis is known that one summary appears in such a way in multiple compares
    # that is included in 2 data parts.
    # This is ignored for now, as it is only one element the positive bias should be neglectable.

    return datasets


def get_text(entry):
    if 'article' in entry['info']:
        text = entry['info']['article']
    elif 'post' in entry['info']:
        text = entry['info']['post']
    else:
        raise KeyError('Unknown structure of dataset entry')
    return entry['info']['title'] + text
    # todo: fließt der title mit ein in die zusammenfassungen? wenn nicht kann das raus


def get_id(entry, s):
    return get_text(entry) + s['text']


def graph_rlff(data):
    summary_compare_graph = {}

    def add_edge(a_k, b_k):
        if b_k in summary_compare_graph:
            summary_compare_graph[b_k].add(a_k)
        else:
            summary_compare_graph[b_k] = {a_k}

    for data_part in data.values():
        for entry in data_part:
            for a in entry['summaries']:
                for b in entry['summaries']:
                    a_id = get_id(entry, a)
                    b_id = get_id(entry, b)
                    if b_id != a_id:
                        add_edge(a_id, b_id)
                        add_edge(b_id, a_id)

    visited = set()
    graphs = {}

    for k in summary_compare_graph:
        if k not in visited:
            graphs[k] = set()

            old_nodes = {k}
            while len(old_nodes) != 0:
                new_nodes = set()
                for x in old_nodes:
                    if x not in visited:
                        visited.add(x)
                        graphs[k].add(x)
                        for i in summary_compare_graph[x]:
                            new_nodes.add(i)
                old_nodes = copy(new_nodes)

    return graphs


# caching="rlfhf_examplelevel_data_cache.pkl"):
def load_rlff_data_with_graphsplits_examplelevel(data_split, data_portion, caching="rlfhf_examplelevel_data_cache"):
    """
    data_portion: mini: 0.005; small: 0.01
    only used when no cache file is used.
    """

    data_split_str = ""
    for s in data_split:
        data_split_str += s
    caching = f"{caching}_{data_split_str}_{data_portion}.pkl"

    if caching is not None and os.path.exists(caching):
        print(f"Used cache of load_rlff_data_with_graphsplits_examplelevel. File name is {caching}")
        with open(caching, 'rb') as file:
            return pickle.load(file)

    data = load_data_rl_from_feedback(data_split)
    graphs = graph_rlff(data)

    print(f"Anzahl der subgraphen: {len(graphs.keys())}")
    keys_to_remove = []
    for k in graphs.keys():
        if pseudo_random(k) > data_portion:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del graphs[k]
    print(f"Anzahl der subgraphen nach Reduktion: {len(graphs.keys())}")

    sub_graph_data = []
    for k in tqdm(graphs, desc="converting to sub graphs"):
        compares = []
        for data_part in data.values():
            for entry in data_part:
                if get_id(entry, entry['summaries'][0]) in graphs[k]:
                    compares.append(entry)
        sub_graph_data.append(compares)

        # # can be temporarily uncommented to enable faster debugging
        # if len(sub_graph_data) > 9:
        #     break

    if caching is not None:
        print(f"cached load_rlff_data_with_graphsplits_examplelevel. File name is {caching}")
        with open(caching, 'wb') as file:
            pickle.dump(sub_graph_data, file)

    return sub_graph_data


def load_squality_human_eval_data(share):
    # todo: den rest zuweißen, so dass am ende die summe 1 ergibt
    shares = {'train': 0.2, 'val': 0.2, 'test': 0.2}

    for s in share:
        if s not in shares:
            raise ValueError('This share (' + s + ') is not known! possible options are ' + str(list(shares.keys())))
    if len(share) == 0:
        raise ValueError('The list of passed shares needs to at least contain one element!')

    bounds = []
    if 'train' in share:
        low_bound = 0
        up_bound = shares['train']
        bounds.append((low_bound, up_bound))
    if 'val' in share:
        low_bound = 0 + shares['train']
        up_bound = 0 + shares['train'] + shares['val']
        bounds.append((low_bound, up_bound))
    if 'test' in share:
        low_bound = 0 + shares['train'] + shares['val']
        up_bound = 0 + shares['train'] + shares['val'] + shares['test']
        bounds.append((low_bound, up_bound))

    human_data = []
    path = "./../human_annotated_data/SQuALITY/all-responses.jsonl"
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        human_data.append(json.loads(json_str))

    # splitting on text level, opposed to on summary level, seems less mixed and so as a cleaner cut.
    # the list is traversed in reverse so the indexes of unhandled elements are not altered.
    length = len(human_data)
    for i in range(length):
        k = length - 1 - i
        random_value = pseudo_random(human_data[k]['passage-id'])

        keep = False
        for b in bounds:
            if not (random_value < b[0] or random_value >= b[1]):
                keep = True
        if not keep:
            del human_data[k]

    return human_data


def analyse_rlff_data(data_split):
    print("--- analyse_rlff_data for", data_split)
    policy_map = {}
    data = load_data_rl_from_feedback(data_split)
    for data_part in data.values():
        for entry in data_part:
            for s in entry['summaries']:
                p = s['policy']
                if p in policy_map:
                    policy_map[p] += 1
                else:
                    policy_map[p] = 1
    print("amount of different policies:", len(policy_map))
    print("how many entries per policy:", policy_map)
    v = policy_map.values()
    print("minimum amount of entries:", min(v))
    print("mean amount of entries:", statistics.mean(v))

    # connectivity
    policy_compare_graph = {}
    for data_part in data.values():
        for entry in data_part:
            for a in entry['summaries']:
                for b in entry['summaries']:
                    a_p = a['policy']
                    b_p = b['policy']
                    if b_p != a_p:
                        def add_edge(a_k, b_k):
                            if b_k in policy_compare_graph:
                                policy_compare_graph[b_k].add(a_k)
                            else:
                                policy_compare_graph[b_k] = {a_k}
                        add_edge(a_p, b_p)
                        add_edge(b_p, a_p)

    print("\npolicy graph:")
    print(policy_compare_graph)
    print("\npolicy graph, edges per node:")
    for k in policy_compare_graph:
        print(k, policy_compare_graph[k])

    visited = set()
    graphs = {}

    for k in policy_compare_graph:
        if k not in visited:
            graphs[k] = set()

            old_nodes = {k}
            while len(old_nodes) != 0:
                new_nodes = set()
                for x in old_nodes:
                    if x not in visited:
                        visited.add(x)
                        for i in policy_compare_graph[x]:
                            new_nodes.add(i)
                for a in old_nodes:
                    graphs[k].add(a)
                old_nodes = copy(new_nodes)

    print("\nlist of connected sub graphs")
    for k in graphs:
        print(k, graphs[k])

    # der graph ist komplett zusammenhängend und
    #  auch die Anzahl der Matches pro Edge macht einen genügend guten Eindruck
    # das bleibt auch so für die data splits (train, val, test)

    whole_amount_matches_within_one_policy = 0
    whole_amount_matches = 0
    for key in data.keys():
        amount_matches_within_one_policy = 0
        amount_matches = 0
        for entry in data[key]:
            amount_matches += 1
            if entry['summaries'][0]['policy'] == entry['summaries'][1]['policy']:
                amount_matches_within_one_policy += 1
        print("share of compares within one policy for batch", key, ":",
              amount_matches_within_one_policy / amount_matches if amount_matches > 0 else "empty batch")
        whole_amount_matches_within_one_policy += amount_matches_within_one_policy
        whole_amount_matches += amount_matches
    print("share of compares within one policy for whole dataset:",
          whole_amount_matches_within_one_policy / whole_amount_matches)

    # alot of compares are between examples of the same policy/summarieser.
    # They can be ignored for system level ranking based on human judgment. (For example level they are still important)
    # Whether they should still be used for computing the system level scoring using the metrics is open.
    #  There is a paper that claims one should use them
    #  Using more data for the metrics might be unfair.
    #  But one downside of human judgment is that it is hardly available for large data sets.
    #   But as large dataset are easy to handle for metrics,
    #    it seems unfair to transfer the limited data burden to metrics.
    # Es sind zwar viele compares innerhalb einer policy aber der anteil ist nicht zu hoch,
    #  auch innerhalb der data  split (train, val, test).
    #  Also auch wenn man die von der Anzahl abzieht sieht es nach einer soliden Aussagekraft aus.


def analyse_rlff_data_for_examplelevel(data_split):
    print("--- analyse_rlff_data_for_examplelevel for", data_split)
    summaries_map = {}
    data = load_data_rl_from_feedback(data_split)
    for data_part in data.values():
        for entry in data_part:
            for s in entry['summaries']:
                identifier = get_id(entry, s)

                if identifier in summaries_map:
                    summaries_map[identifier] += 1
                else:
                    summaries_map[identifier] = 1
    print("amount of unique summary-doc pairs:", len(summaries_map))
    v = summaries_map.values()
    print("amount of entries per unique pair: min, mean, max:", min(v), statistics.mean(v), max(v))

    # connectivity
    graphs = graph_rlff(data)
    print("-----")
    print(len(graphs.values()))

    lengths = []
    for k in graphs.values():
        lengths.append(len(k))
    print(min(lengths), statistics.mean(lengths), max(lengths))

    # --> the graph is very not connected. It consists of very many small sub graphs.
    # node stats of the sub graphs: number of edges: min: 2 mean: 4.590897608622432 max:22


def analyse_rlff_axes_for_exanplelevel(data_splits):
    data = load_data_rlfhf_axis(data_splits)


def analyse_data_split_distribution_rlfhf_comps_data():
    print("--- analyse_data_split_distribution_rlfhf_comps_data")
    text_to_split = {}
    splits = {}
    data = load_data_rl_from_feedback(['train', 'val', 'test'])
    for data_batch in data.values():
        for element in data_batch:
            split = element['split']
            if split in splits:
                splits[split] += 1
            else:
                splits[split] = 1
            for s in element['summaries']:
                ident = get_id(element, s)
                if ident not in text_to_split:
                    text_to_split[ident] = set()
                text_to_split[ident].add(split)

    print(splits)

    shares = {}
    count = 0
    for v in splits.values():
        count += v
    for key in splits:
        shares[key] = splits[key] / count
    print(shares)

    amount_of_multis = 0
    text_to_split_amounts = {}
    for key in text_to_split:
        text_to_split_amounts[key] = len(text_to_split[key])
        if len(text_to_split[key]) > 1:
            amount_of_multis += 1
            print("summary which is in more than one split.",
                  "split:", text_to_split[key],
                  "text:", key.replace("\r", "").replace("\n", ""))

    # print(statistics.mean(text_to_split_amounts.values()))
    print("Amount of text-summary pairs rated in more than one split", amount_of_multis)
    # print(amount_of_multis / len(text_to_split_amounts) * 100)

    # there is only one element which is rated in more than 1 split. If this is removed the splits can be used.
    # the fact the on example level the dataset is shattered into many sub graphs
    #  was a hint to expect such a low connectivity between the data splits.


def analyse_data_split_distribution_rlfhf_axis_data():
    print("--- analyse_data_split_distribution_rlfhf_axis_data")
    text_to_split = {}
    splits = {}
    data = load_data_rlfhf_axis(['train', 'val', 'test'])
    for data_batch in data.values():
        for element in data_batch:
            split = element['split']
            if split in splits:
                splits[split] += 1
            else:
                splits[split] = 1

            ident = get_id(element, element['summary'])
            if ident not in text_to_split:
                text_to_split[ident] = set()
            text_to_split[ident].add(split)

    print(splits)

    shares = {}
    count = 0
    for v in splits.values():
        count += v
    for key in splits:
        shares[key] = splits[key] / count
    print(shares)
    print('amount of entries:', count)

    amount_of_multis = 0
    text_to_split_amounts = {}
    for key in text_to_split:
        text_to_split_amounts[key] = len(text_to_split[key])
        if len(text_to_split[key]) > 1:
            amount_of_multis += 1
            print("summary which is in more than one split.",
                  "split:", text_to_split[key],
                  "text:", key.replace("\r", "").replace("\n", ""))

    # print(statistics.mean(text_to_split_amounts.values()))
    print("Amount of text-summary pairs rated more than once and so in different splits", amount_of_multis)
    # print(amount_of_multis / len(text_to_split_amounts) * 100)

    # amount of entries: 14897
    # shares: ~ {'test': 0.42, 'valid2': 0.44, 'valid1': 0.14}
    # keine überschneidungen zwischen den splits.


def analyse_gpt3_data():
    print("--- analyse_gpt3_data")
    data = load_data(['test', 'val', 'test'])

    text_counts = {}
    for batch in data.values():
        for entry in batch.values():
            article = entry['article']
            if article not in text_counts:
                text_counts[article] = 0
            text_counts[article] += 1

    print("max occurrence of one text:", max(list(text_counts.values())))
    # each text occurs only once, so splitting the texts into train/val/test splits based on the texts is fine.


# analyse_data_split_distribution_rlfhf_comps_data()
# analyse_data_split_distribution_rlfhf_axis_data()
# analyse_gpt3_data()
# for data_split_temp in [['train'], ['val'], ['test'], ['train', 'val', 'test']]:
#     analyse_rlff_data(data_split_temp)

# analyse_rlff_data_for_examplelevel(['train', 'val', 'test'])
# analyse_rlff_axes_for_exanplelevel(['train', 'val', 'test'])
