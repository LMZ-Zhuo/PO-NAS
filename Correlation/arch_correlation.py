import json
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

file = "data/zc_nasbench201.json"
with open(file, "r") as f:
    data_stats = json.load(f)
task = 'cifar10'

valid_indices = {1, 2, 3, 4}

list_op0 = []
list_op1 = []
list_op2 = []
list_op3 = []
list_op4 = []
list_mix = []
param_scores = []

metrics = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacov', 'zen', 'nwot', 'params', 'flops']

for arch_str in data_stats[task]:
    arch_tuple = eval(arch_str)
    count_op0 = arch_tuple.count(0)
    count_op1 = arch_tuple.count(1)
    count_op2 = arch_tuple.count(2)
    count_op3 = arch_tuple.count(3)
    count_op4 = arch_tuple.count(4)

    arch_data = data_stats[task][arch_str]
    current_id = arch_data['id']
    param_score = arch_data['params']['score']
    param_scores.append((param_score, arch_str))

    if count_op0 >= 3:
        list_op0.append(arch_str)
    if count_op1 >= 3:
        list_op1.append(arch_str)
    if count_op2 >= 3:
        list_op2.append(arch_str)
    if count_op3 >= 3:
        list_op3.append(arch_str)
    if count_op4 >= 3:
        list_op4.append(arch_str)
    if (count_op0 == 0 and count_op1 <= 2 and count_op2 <= 2
            and count_op3 <= 2 and count_op4 <= 2):
        list_mix.append(arch_str)

param_scores_sorted = sorted(param_scores, key=lambda x: x[0])
n = len(param_scores_sorted)
low_threshold = n // 3
medium_threshold = 2 * n // 3

low_params = [arch for score, arch in param_scores_sorted[:low_threshold]]
medium_params = [arch for score, arch in param_scores_sorted[low_threshold:medium_threshold]]
high_params = [arch for score, arch in param_scores_sorted[medium_threshold:]]

class_lists = {
    'op0': list_op0, 'op1': list_op1, 'op2': list_op2, 'op3': list_op3, 'op4': list_op4,
    'mix': list_mix, 'low': low_params, 'med': medium_params, 'high': high_params
}

corr_matrix = np.zeros((len(class_lists), len(metrics)))

for cls_idx, (cls_name, arch) in enumerate(class_lists.items()):
    X = [data_stats[task][arch]['val_accuracy'] for arch in arch]
    for met_idx, metric in enumerate(metrics):
        Y = [data_stats[task][arch][metric]['score'] for arch in arch]
        rho, p_value = spearmanr(X, Y)
        corr_matrix[cls_idx, met_idx] = rho

df_corr = pd.DataFrame(corr_matrix,
                       index=class_lists.keys(),
                       columns=metrics)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)

def compute_global_correlation(df):
    all_archs = [arch_str for arch_str in data_stats[task]]
    accuracies = [data_stats[task][arch]['val_accuracy'] for arch in all_archs]
    global_row = {'class': 'all'}

    for metric in metrics:
        scores = [data_stats[task][arch][metric]['score'] for arch in all_archs]
        rho, _ = spearmanr(accuracies, scores)
        global_row[metric] = rho

    global_df = pd.DataFrame([global_row]).set_index('class')
    return pd.concat([df, global_df])

full_df = compute_global_correlation(df_corr)
print(full_df)
