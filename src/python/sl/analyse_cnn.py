import os
import numpy as np
import pandas as pd

rootdir = 'cnn_results'

splits = range(0, 20)

params_values = {}

for ceta in [0.001, 0.0001, 0.00001]:
    for batchsize in [50, 100, 500]:
        params = (ceta, batchsize)
        tag = f"batchsize={batchsize}_eta={ceta}"
        for split in splits:
            params_values.setdefault(params, {})
            ctag = f"{tag}_{split}"
            fn = os.path.join(rootdir, tag, f'cnn_{ctag}.csv')
            with open(fn, mode='r') as fh:
                last = fh.readlines()[-1]
                class_correct, class_total, test_loss = last.split(",")
                # parse the arrays
                class_correct = class_correct.split("=")[-1][1:-1].split(" ")
                class_correct = np.array([float(i) if i != '' else 0 for i in class_correct])
                class_total = class_total.split("=")[-1][1:-1].split(" ")
                class_total = np.array([float(i) if i != '' else 0 for i in class_total])
                test_err = 1 - sum(class_correct) / sum(class_total)
                print(last)
                print(test_err)
                params_values[params][split] = test_err

params_values = pd.DataFrame(params_values)

mean_vals = params_values.mean(axis=0).unstack()
std_vals = params_values.std(axis=0).unstack()

