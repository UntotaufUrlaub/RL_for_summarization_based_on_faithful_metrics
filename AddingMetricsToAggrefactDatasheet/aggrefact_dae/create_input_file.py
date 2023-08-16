import pandas as pd
import re

path = 'sample_cache.txt'


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)


def append_sample(summ, doc):
    doc = doc.replace("\n", " ").lower()
    doc = remove_non_ascii(doc)
    summ = summ.replace("\n", " ").lower()
    summ = remove_non_ascii(summ)
    with open(path, 'a') as f:
        f.write(doc + '\n' + summ + '\n\n')
        f.close()


df = pd.read_csv("aggre_fact_final.csv")

# clear input file
with open(path, 'w') as f:
    f.write("")
    f.close()

# print("dactivate the sample selection")
# df = df.sample(n=40, random_state=0)
df.apply(lambda row: append_sample(row['summary'], row['doc']), axis=1)
