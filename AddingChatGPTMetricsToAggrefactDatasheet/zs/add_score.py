import pandas as pd
from datetime import datetime, timedelta
import ast
# import warnings

# warnings.filterwarnings('ignore')


def add_score(source_path, target_path, metric_functions, use_caching, caching_interval, print_timing=False,
              data_amount=None, data_selection_seed=None):
    while True:
        res = None

        df = pd.read_csv(source_path)

        if data_amount is not None:
            if data_selection_seed is None:
                df = df.head(data_amount)
            else:
                df = df.sample(n=data_amount, random_state=data_selection_seed)

        # print(df[[col for col in df.columns if col != "doc" and col != "summary"]].to_string())

        later = datetime.now() + timedelta(minutes=caching_interval)

        for name, function in metric_functions.items():
            if use_caching:
                column_found = None
                for n in df.columns.tolist():
                    if n.startswith(name+"_") and n.endswith("_score"):
                        column_found = n
                        break
                if column_found is not None:
                    print(f"known metric: (identified by colum {column_found})")
                    row_filter = df[column_found].isna()
                else:
                    print("new metric:")
                    row_filter = df.index

                print(f"{name}: {len(df.loc[row_filter])} entries to go.")
                if (len(df.loc[row_filter])) == 0:
                    print("skip")
                    continue
                print(f"next caching at around {later.strftime('%H:%M (%Y-%m-%d)')}")

                def apply_function(row):
                    if datetime.now() > later:
                        return None
                    return function(row['summary'], row['doc'])
            else:
                if print_timing:
                    start = datetime.now()
                row_filter = df.index

                def apply_function(row):
                    return function(row['summary'], row['doc'])

            res = df.loc[row_filter].apply(apply_function, axis=1)
            # print(res)

            first_res_item = res[res.index[0]]

            if isinstance(first_res_item, str) and isinstance(ast.literal_eval(first_res_item), dict):
                series_dict = {}
                for k in ast.literal_eval(first_res_item).keys():
                    series_dict[k] = []
                for e in res:
                    for k, v in ast.literal_eval(e).items():
                        series_dict[k].append(v)
                for k, v in series_dict.items():
                    df.loc[row_filter, name + '_' + k + '_score'] = pd.Series(v, index=res.index)
            else:
                df.loc[row_filter, name + '_score'] = res

            if use_caching and res.isna().any():
                # if this metric still has not all results,
                # we break, to come to writing/caching, bevor performing the next metric.
                break

            if not use_caching and print_timing:
                elapsed_time = datetime.now() - start

                # Format the elapsed time
                hours = int(elapsed_time.total_seconds() // 3600)
                minutes = int((elapsed_time.total_seconds() % 3600) // 60)
                seconds = elapsed_time.total_seconds() % 60

                print("{}: Elapsed time: {} hours, {} minutes, {} seconds".format(name, hours, minutes, seconds))

                elapsed_time_per_sample = elapsed_time / len(df)

                # Format the elapsed time
                hours = int(elapsed_time_per_sample.total_seconds() // 3600)
                minutes = int((elapsed_time_per_sample.total_seconds() % 3600) // 60)
                seconds = elapsed_time_per_sample.total_seconds() % 60

                print("{}: Elapsed time per sample: {} hours, {} minutes, {} seconds".format(name, hours, minutes, seconds))

        df.to_csv(target_path, index=False)

        if not use_caching or res is None or not res.isna().any():
            break

    print(df)
    # print(df.to_string())
