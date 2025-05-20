import argparse
import json
import numpy as np
import os
from genotype_to_pretrain_data_tnb101_macro import build_dynamic_dataset_with_id
from pretrain_dataloader_tnb101_macro import get_nas_dataloader
from att_network_tnb101_macro import DynamicNASFramework
from pretrain_tnb101_macro import run_pretrain
from att_network_trainer_tnb101_macro import OnlineTrainer
import time
from tqdm import tqdm
import sys
from torch_geometric.data import Batch
import math
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Search on TransNAS-Bench-101")
    parser.add_argument('--benchmark', choices=['micro', 'macro'], default='macro')
    parser.add_argument('--task', choices=['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
                                        , default='class_scene')
    parser.add_argument('--diff_threshold', type=float, default=0.1, help='')
    parser.add_argument('--loss_threshold', type=float, default=0.1, help='')
    parser.add_argument('--step', type=float, default=100, help='')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--search_budget', default=100, type=int)
    parser.add_argument('--training_free_metrics', nargs='+',
                        default=['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacov'],
                        help='List of training-free metrics')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    file = "data/zc_transbench101_macro.json"

    xtask = args.task
    metric_names = args.training_free_metrics

    if xtask in ['segmentsemantic', 'normal', 'autoencoder']:
        if "synflow" in metric_names:
            metric_names.remove("synflow")

    with open(file, "r") as f:
        data_stats = json.load(f)

    num_arch = len(data_stats[xtask])

    val_accs_101 = []
    data_metrics = {}
    for metric in metric_names:
        data_metrics[metric] = []
    for data_key in data_stats[xtask]:
        data = data_stats[xtask][data_key]
        val_accs_101.append(data['val_accuracy'])

        for metric in metric_names:
            data_metrics[metric].append(data[metric]['score'])
    test_accs = val_accs_101
    test_order = np.flip(np.argsort(test_accs))
    test_rank = np.argsort(test_order)
    running_rounds = args.search_budget
    # search
    metric_str = '_'.join(metric_names)
    metric_str += f'_{xtask}'
    pretrain_database_path = f"data/pretrain_data/{metric_str}.pt"
    if not os.path.exists(pretrain_database_path):
        if not os.path.exists(f"data/pretrain_data"):
            os.makedirs(f"data/pretrain_data")
        print(f"generate pretrain dataset: {pretrain_database_path}")

        for metric_name in metric_names:
            print(metric_name, max(data_metrics[metric_name]), min(data_metrics[metric_name]))
            if max(data_metrics[metric_name]) - min(data_metrics[metric_name]) == 0:
                data_metrics[metric_name] = [float(i) for i in data_metrics[metric_name]]
            else:
                data_metrics[metric_name] = [(float(i) - min(data_metrics[metric_name])) /
                                             (max(data_metrics[metric_name]) - min(data_metrics[metric_name]))
                                             for i in data_metrics[metric_name]]

        build_dynamic_dataset_with_id(xtask, data_stats, metric_names, data_metrics, pretrain_database_path)
    else:
        print(f"pretrain dataset already exists: {pretrain_database_path}")

    dataset = torch.load(pretrain_database_path)

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model_save_path = f"data/pretrain_model/{metric_str}.pt"
    pretrain_model_path = f"data/pretrain_model/{metric_str}.pt/best_model.pth"

    if not os.path.exists(pretrain_model_save_path):
        if not os.path.exists(f"data/pretrain_model"):
            os.makedirs(f"data/pretrain_model")
        print(f"generate pretrain_model: {pretrain_model_save_path}")
        (train_loader, test_loader), metric_names = get_nas_dataloader(pretrain_database_path)
        model = DynamicNASFramework(num_metrics=len(metric_names)).to(device)
        run_pretrain(model, train_loader, test_loader, device, metric_names, pretrain_model_save_path, args.diff_threshold)

    else:
        print(f"pretrain_model already exists: {pretrain_model_save_path}")

    pretrain_model = DynamicNASFramework(num_metrics=len(metric_names), pretrain_mode=True)
    checkpoint = torch.load(pretrain_model_path)
    pretrain_model.arch_encoder.load_state_dict(checkpoint['arch_encoder'])
    pretrain_model.metric_head.load_state_dict(checkpoint['metric_head'])
    pretrain_model.to(device)
    pretrain_model.eval()

    loss_threshold = args.loss_threshold
    max_cycles = 5
    error_count_threshold = 0
    online_trainer = OnlineTrainer(pretrain_model_path, num_metrics=len(metric_names), max_cycles=max_cycles,
                                   loss_threshold=loss_threshold, diff_threshold=args.diff_threshold, total_budget=args.search_budget, device=f'cuda:{args.gpu}')

    cycle_epochs = args.search_budget
    current_cycle = 0
    step = args.step
    opt_archs = []
    val_accs = {}

    if current_cycle == 0:
        print(f"\n=== Init Cycle  ===")

        metrics_scores = []
        for i in range(len(dataset["arch_ids"])):
            metrics_scores.append((dataset["metrics"][i].mean().item(), dataset["arch_ids"][i].item()))
        sorted_scores = sorted(metrics_scores, key=lambda x: x[0], reverse=True)

        N = 3
        init_arch_ids = [item[1] for item in sorted_scores[:N]]
        init_arch_scores = [item[0] for item in sorted_scores[:N]]
        print(f"init_arch: {init_arch_ids},avg_metrics: {init_arch_scores}")

        for arch in init_arch_ids:
            val_acc = val_accs_101[arch]
            val_accs[arch] = val_acc
            opt_archs = opt_archs + [arch]

        acc_list = ', '.join([f'{arch_id}: {acc:.4f}' for arch_id, acc in sorted(val_accs.items())])
        print(f'current val: [ {acc_list} ]')
        best_id, best_acc = max(val_accs.items(), key=lambda x: x[1])
        print(f'best val: [ {best_id}: {best_acc:.4f} ]')

        init_train_data = []
        for arch in init_arch_ids:
            train_data = {
                'arch': dataset['arch'][arch],
                'metric_ids': torch.arange(len(metric_names)),
                'metrics': dataset['metrics'][arch],
                'true_score': torch.tensor([val_accs[arch]])
            }
            init_train_data.append(train_data)
        loss = online_trainer.select_and_train(step, init_train_data)

        print(f"Surrogate model init_stage loss: {loss}")

    while current_cycle < cycle_epochs:
        print(f"\n=== Cycle {current_cycle + 1}/{cycle_epochs} ===")
        time.sleep(0.5)
        scores = []
        batch_size = 128
        with torch.no_grad():

            total_archs = len(dataset["arch_ids"])

            indices = list(range(total_archs))
            batches = [indices[i:i + batch_size] for i in range(0, total_archs, batch_size)]

            with tqdm(total=len(batches), desc="Evaluate arch", unit="batch",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                      file=sys.stdout) as pbar:
                for batch_indices in batches:
                    arch_batch = Batch.from_data_list([
                        dataset['arch'][i].to(device) for i in batch_indices
                    ])
                    metric_ids_batch = torch.stack([
                        torch.arange(len(metric_names)).to(device) for _ in batch_indices
                    ])
                    metrics_batch = torch.stack([
                        dataset['metrics'][i].to(device) for i in batch_indices
                    ])

                    online_out = online_trainer.model(
                        arch=arch_batch,
                        metric_ids=metric_ids_batch,
                        metrics=metrics_batch
                    )

                    final_scores = online_out['score'].squeeze().cpu().tolist()
                    for score, idx in zip(final_scores, batch_indices):
                        scores.append((score, idx))

                    pbar.update(1)
                    pbar.set_postfix(
                        current_batch=f"{min(batch_indices)}-{max(batch_indices)}",
                        batch_size=len(batch_indices)
                    )

        cleaned_scores = [(x[0] if not math.isnan(x[0]) else -math.inf, x[1]) for x in scores]
        sorted_scores = sorted(cleaned_scores, key=lambda x: -x[0])

        for score, arch in sorted_scores:
            if arch not in val_accs:
                selected_arch = arch
                top_score = score
                break

        print(f"Selected Arch: {selected_arch},Score: {top_score:.4f}")

        if selected_arch not in val_accs:
            val_acc = val_accs_101[selected_arch]
            val_accs[selected_arch] = val_acc
            opt_archs = opt_archs + [selected_arch]
        print(f'current val: [ {selected_arch}: {val_acc:.4f} ]')
        best_id, best_acc = max(val_accs.items(), key=lambda x: x[1])
        print(f'best val: [ {best_id}: {best_acc:.4f} ]')

        train_data = [{
            'arch': dataset['arch'][selected_arch],
            'metric_ids': torch.arange(len(metric_names)),
            'metrics': dataset['metrics'][selected_arch],
            'true_score': torch.tensor([val_accs[selected_arch]])
        }]
        loss = online_trainer.select_and_train(step, train_data)
        print(f"Surrogate model final loss: {loss}")
        current_cycle += 1

    val_accs_opt_archs = [val_accs[arch] for arch in opt_archs]
    opt_acc = -10000
    opt_arch = 0
    highest_test_accs = []
    for i in range(len(val_accs_opt_archs)):
        acc = val_accs_opt_archs[i]
        arch = opt_archs[i]
        if acc > opt_acc:
            opt_acc = acc
            opt_arch = arch
        highest_test_accs.append(test_accs[opt_arch])
    print("Test accs of selected architecture for this round:" + str(highest_test_accs[-1]))
