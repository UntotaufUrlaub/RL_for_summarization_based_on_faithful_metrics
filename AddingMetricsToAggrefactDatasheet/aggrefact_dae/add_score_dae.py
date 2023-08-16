import pandas as pd

array = []
with open('cache.txt', 'r') as file:
    for line in file:
        array.append(int(line.split("Sent-level pred:\t")[1].split('\n')[0]))

source_path = "aggre_fact_final.csv"
df = pd.read_csv(source_path)
# print("dactivate the sample selection")
# df = df.sample(n=40, random_state=0)
df["my_dae_batched_score"] = array
df.to_csv(source_path, index=False)
