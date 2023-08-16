from human_allignment import eval_metric
import pandas as pd
import random


# To run this function you need to set num_proc = 1 in /Code/human_alignment/helper.py.
# Otherwise, multiprocessing will mess with things
def create_csv_of_unique_doc_sum_pairs(
        data_portion_gpt, data_portion_rlfhf, data_portion_rlfhf_axes,
        target_path="human_allignment_metric_scores.csv",
        data_split=['train', 'val'],
        levels=['example', 'system'],):

    target_path = f"{target_path}_portion_{data_portion_gpt}_{data_portion_rlfhf}_{data_portion_rlfhf_axes}.csv"

    docs = {}

    def dummy(summ, doc):
        #print("in dummy")
        if doc in docs.keys():
            #print("add")
            docs[doc].add(summ)
        else:
            #print("skip")
            docs[doc] = {summ}
        return random.random()

    for level in levels:
        eval_metric({
            "random_dummy": dummy,
        }, data_split, level, data_portion_gpt, data_portion_rlfhf, data_portion_rlfhf_axes)

        count = 0
        for summs in docs.values():
            count += len(summs)
        print(f"Count {level}: ", count)

    # convert dict to df and store it.
    doc_array = []
    summ_array = []
    for doc, summs in docs.items():
        for summ in summs:
            doc_array.append(doc)
            summ_array.append(summ)
    df = pd.DataFrame()
    df['doc'] = doc_array
    df['summary'] = summ_array
    print("df length: ", len(df))
    df.to_csv(target_path)


data_portion_gpt = 1            # __400 ->  400
data_portion_rlfhf = 0.1        # 79000 -> 7900
data_portion_rlfhf_axes = 0.5   # _6000 -> 3000
#                               gesammt:  11300

#create_csv_of_unique_doc_sum_pairs(
#    data_portion_gpt, data_portion_rlfhf, data_portion_rlfhf_axes,
#    target_path="human_allignment_docSumPairs_train",
#    data_split=['train'],
#    levels=['example'])
# train:
# dataset_gpt: Count example:  ursprünglich: 363; mini: 42
# dataset_rlfhf: small: 705, mini: 365
# dataset_rlfhf_axes: ursprünglich: 5986; mini: 48
# gesammt: mini: 455

data_portion_gpt_val = 1
data_portion_rlfhf_val = 0.5
data_portion_rlfhf_axes_val = 1

create_csv_of_unique_doc_sum_pairs(
    data_portion_gpt_val, data_portion_rlfhf_val, data_portion_rlfhf_axes_val,
    target_path="human_allignment_docSumPairs_val",
    data_split=['val'],
    levels=['example'])
# # mini war zu klein
# # small: 458
