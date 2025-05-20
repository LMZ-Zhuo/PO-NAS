import os
import sys
import time
import logging
import argparse
import pickle
import copy
import itertools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="scipy._lib.messagestream.MessageStream size changed")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from matrix_evolution import compute_cost, matrix_evolution
from tqdm import tqdm
from scipy.special import softmax
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from DARTS import utils
from DARTS.genotypes import *
from foresight.pruners.measures import fisher, grad_norm, grasp, snip, synflow, jacov
from foresight.pruners.utilities import *
from genotype_to_pretrain_data import genotype_to_gnn_data
from torch_geometric.data import Batch
import re
from genotypes import Genotype
from pretrain_dataloader import get_nas_dataloader
from att_network import DynamicNASFramework
from pretrain import run_pretrain
from att_network_trainer import OnlineTrainer
import hashlib


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--arch_path', type=str, default='data/sampled_archs.p', help='location of the data corpus')
parser.add_argument('--no_search', action='store_true', default=False, help='only apply sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for search')
parser.add_argument('--batch_size', type=int, default=576, help='batch size')
parser.add_argument('--metric_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--exc_num', type=int, default=20, help='num of excellent')
parser.add_argument('--init_epochs', type=int, default=10, help='init')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--scale', type=float, default=1e2, help="")
parser.add_argument('--n_sample', type=int, default=10000, help='pytorch manual seed')
parser.add_argument('--total_iters', type=int, default=25, help='BO')
parser.add_argument('--diff_threshold', type=float, default=0.1, help='diff_threshold')
parser.add_argument('--loss_threshold', type=float, default=0.1, help='loss_threshold')
parser.add_argument('--init_portion', type=float, default=0.25, help='pytorch manual seed')
parser.add_argument('--acq', type=str, default='ucb',help='choice of bo acquisition function, [ucb, ei, poi]')
args = parser.parse_args()
args.cutout = False
args.auxiliary = False

args.save = 'darts/search-{}-{}'.format(args.save, args.dataset)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, f'S{args.seed}-log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

iteration_counter = 0

if args.dataset == 'cifar10':
    NUM_CLASSES = 10
    from DARTS.model import NetworkCIFAR as Network
elif args.dataset == 'cifar100':
    NUM_CLASSES = 100
    from DARTS.model import NetworkCIFAR as Network
elif args.dataset == 'imagenet':
    NUM_CLASSES = 1000
    from DARTS.model import NetworkImageNet as Network
else:
    raise ValueError('Donot support dataset %s' % args.dataset)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # load the dataset
    if args.dataset == 'cifar10':
        train_queue, valid_queue = utils._get_cifar10(args)
    elif args.dataset == 'cifar100':
        train_queue, valid_queue = utils._get_cifar100(args)
    elif args.dataset == 'imagenet':
        train_queue, valid_queue = utils._get_imagenet(args)
    else:
        raise ValueError("Donot support dataset %s" % args.dataset)

    data_queues = [train_queue, valid_queue]

    metric_names = ['synflow', 'snip', 'jacov', 'fisher', 'grad_norm', 'grasp']

    # the domain for search
    pbounds = {}
    for metric in metric_names:
        pbounds[metric] = (-1, 1)

    sampled_genos, opt_genos, sampled_metrics = get_pool(data_queues, args)

    id_to_key = {}

    id = 0
    for key in sampled_metrics:
        id_to_key[id] = key
        id += 1

    base_name = os.path.basename(args.arch_path)
    file_name = os.path.splitext(base_name)[0]
    metric_str = '_'.join(metric_names)
    pretrain_database_path = f"data/pretrain_data/{file_name}/{metric_str}.pt"

    if not os.path.exists(pretrain_database_path):
        if not os.path.exists(f"data/pretrain_data/{file_name}"):
            os.makedirs(f"data/pretrain_data/{file_name}")
        print(f"generate pretrain dataset: {pretrain_database_path}")
        corrected_sampled_metrics = [
            {"genotype": geno_str, "metrics": metrics}
            for geno_str, metrics in sampled_metrics.items()
        ]
        build_dynamic_dataset_with_id(corrected_sampled_metrics, metric_names, pretrain_database_path)

    else:
        print(f"pretrain dataset already exists: {pretrain_database_path}")

    dataset = torch.load(pretrain_database_path)

    max_vals = dataset["max_vals"].squeeze()
    min_vals = dataset["min_vals"].squeeze()
    print(f"metrics: max_vals: {max_vals}, min_vals:{min_vals}")

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model_save_path = f"data/pretrain_model/{file_name}/{metric_str}.pt"
    pretrain_model_path = f"data/pretrain_model/{file_name}/{metric_str}.pt/best_model.pth"

    if not os.path.exists(pretrain_model_save_path):
        if not os.path.exists(f"data/pretrain_model/{file_name}"):
            os.makedirs(f"data/pretrain_model/{file_name}")
        print(f"generate pretrain_model: {pretrain_model_save_path}")
        (train_loader, test_loader), metric_names = get_nas_dataloader(pretrain_database_path)
        model = DynamicNASFramework(num_metrics=len(metric_names)).to(device)
        run_pretrain(model, train_loader, test_loader, device, metric_names, pretrain_model_save_path, args.diff_threshold)

    else:
        print(f"pretrain_model already exists: {pretrain_model_save_path}")

    pretrain_model = DynamicNASFramework(num_metrics=len(metric_names), pretrain_mode=True)
    checkpoint = torch.load(pretrain_model_path)
    pretrain_model.norm_encoder.load_state_dict(checkpoint['norm_encoder'])
    pretrain_model.reduce_encoder.load_state_dict(checkpoint['reduce_encoder'])
    pretrain_model.metric_head.load_state_dict(checkpoint['metric_head'])
    pretrain_model.to(device)
    pretrain_model.eval()

    loss_threshold = args.loss_threshold
    max_cycles = 5
    online_trainer = OnlineTrainer(pretrain_model_path, num_metrics=len(metric_names),max_cycles=max_cycles,
            loss_threshold=loss_threshold, diff_threshold=args.diff_threshold, total_budget=args.total_iters, device=f'cuda:{args.gpu}')

    cycle_epochs = args.total_iters
    current_cycle = 0
    step = 1000
    opt_archs = []
    val_accs = {}

    train_queue, _ = data_queues
    inputs, targets = next(iter(train_queue))
    inputs, targets = inputs[:args.metric_batch_size], targets[:args.metric_batch_size]

    inputs = inputs.cuda()
    targets = targets.cuda()

    start = time.time()

    # ============= init ==============

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
            logging.info("current opt_arch:" + str(arch))
            opt_archs.append(arch)
            opt_key = id_to_key[arch]
            opt_geno = sampled_genos[opt_key]
            opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
            val_acc = train(data_queues, opt_model)
            val_accs[arch] = val_acc

        acc_list = ', '.join([f'{arch_id}: {acc:.2f}%' for arch_id, acc in sorted(val_accs.items())])
        logging.info(f'current val: [ {acc_list} ]')
        best_id, best_acc = max(val_accs.items(), key=lambda x: x[1])
        logging.info(f'best val: [ {best_id}: {best_acc:.2f}% ]')

        init_train_data = []
        for arch in init_arch_ids:
            train_data = {
                'norm_cell': dataset['normal_cells'][arch],
                'reduce_cell': dataset['reduce_cells'][arch],
                'metric_ids': torch.arange(len(metric_names)),
                'metrics': dataset['metrics'][arch],
                'true_score': torch.tensor([val_accs[arch]])
            }
            init_train_data.append(train_data)
        loss = online_trainer.select_and_train(step, init_train_data)

        print(f"Surrogate model init_stage loss: {loss}")

    # ========= BO ===========
    while current_cycle < cycle_epochs:
        print(f"\n=== Cycle {current_cycle + 1}/{cycle_epochs} ===")
        time.sleep(0.5)

        scores = []
        batch_size = 128

        with torch.no_grad():
            total_archs = len(dataset["arch_ids"])
            indices = list(range(total_archs))
            batches = [indices[i:i + batch_size] for i in range(0, total_archs, batch_size)]
            with tqdm(total=len(batches), desc="Evaluate arch", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", file=sys.stdout) as pbar:
                for batch_indices in batches:
                    norm_cell_batch = Batch.from_data_list([
                        dataset['normal_cells'][i].to(device) for i in batch_indices
                    ])
                    reduce_cell_batch = Batch.from_data_list([
                        dataset['reduce_cells'][i].to(device) for i in batch_indices
                    ])
                    metric_ids_batch = torch.stack([
                        torch.arange(len(metric_names)).to(device) for _ in batch_indices
                    ])
                    metrics_batch = torch.stack([
                        dataset['metrics'][i].to(device) for i in batch_indices
                    ])

                    online_out = online_trainer.model(
                        norm_cell=norm_cell_batch,
                        reduce_cell=reduce_cell_batch,
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
        sorted_scores = sorted(scores, key=lambda x: -x[0])

    #   ========= evo ==========
        if current_cycle >= args.init_epochs:
            num_pop = 20
            excellent_archs = []
            excellent_scores = []
            for score, idx in sorted_scores:
                excellent_archs.append(idx)
                excellent_scores.append(score)
                if len(excellent_archs) == num_pop:
                    break
            logging.info('excellent arch: {}'.format(excellent_archs))
            logging.info('excellent scores: {}'.format(excellent_scores))

            cost_list = []
            score_cost_list = []
            excellent_pairs = list(itertools.combinations(excellent_archs, 2))
            for pair in excellent_pairs:
                opt_key_0, opt_key_1 = id_to_key[pair[0]], id_to_key[pair[1]]
                if pair[0] < args.n_sample and pair[1] < args.n_sample:
                    opt_geno_0, opt_geno_1 = sampled_genos[opt_key_0], sampled_genos[opt_key_1]
                elif pair[0] < args.n_sample <= pair[1]:
                    opt_geno_0, opt_geno_1 = sampled_genos[opt_key_0], opt_key_1
                elif pair[1] < args.n_sample <= pair[0]:
                    opt_geno_0, opt_geno_1 = opt_key_0, sampled_genos[opt_key_1]
                else:
                    opt_geno_0, opt_geno_1 = opt_key_0, opt_key_1
                cost = compute_cost(opt_geno_0, opt_geno_1)
                cost_list.append(cost)

            max_value = max(cost_list)
            min_value = min(cost_list)
            if max_value - min_value == 0:
                cost_list = [float(i) for i in cost_list]
            else:
                cost_list = [(float(i) - min_value) / (max_value - min_value) for i in cost_list]
            index_cost = 0
            if current_cycle <= args.total_iters - 5:
                N = 1 - (current_cycle - args.init_epochs) / (args.total_iters - args.init_epochs - 5)
            else:
                N = 0
            for pair in excellent_pairs:
                score = scores[pair[0]][0] + scores[pair[1]][0] + N * cost_list[index_cost]
                index_cost += 1
                score_cost_list.append(score)
            score_cost_list_order = np.flip(np.argsort(score_cost_list))

            evo_result_list = []
            for index in score_cost_list_order[:args.exc_num]:
                opt_key_0, opt_key_1 = id_to_key[excellent_pairs[index][0]], id_to_key[excellent_pairs[index][1]]
                if excellent_pairs[index][0] < args.n_sample and excellent_pairs[index][1] < args.n_sample:
                    opt_geno_0, opt_geno_1 = sampled_genos[opt_key_0], sampled_genos[opt_key_1]
                elif excellent_pairs[index][0] < args.n_sample <= excellent_pairs[index][1]:
                    opt_geno_0, opt_geno_1 = sampled_genos[opt_key_0], opt_key_1
                elif excellent_pairs[index][1] < args.n_sample <= excellent_pairs[index][0]:
                    opt_geno_0, opt_geno_1 = opt_key_0, sampled_genos[opt_key_1]
                else:
                    opt_geno_0, opt_geno_1 = opt_key_0, opt_key_1
                evo_result, _ = matrix_evolution(opt_geno_0, opt_geno_1)
                for arch in evo_result:
                    evo_result_list.append(arch)

            with torch.no_grad():
                pop_scores = []
                total_archs = len(evo_result_list)
                batch_size = 128
                indices = list(range(total_archs))
                batches = [indices[i:i + batch_size] for i in range(0, total_archs, batch_size)]

                with tqdm(total=len(batches), desc="Predict pop_pool score", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", file=sys.stdout) as pbar:
                    for batch_indices in batches:
                        norm_cell_batch = Batch.from_data_list([
                            genotype_to_gnn_data(parse_genotype(str(evo_result_list[i])), "normal").to(device) for i in batch_indices
                        ])
                        reduce_cell_batch = Batch.from_data_list([
                            genotype_to_gnn_data(parse_genotype(str(evo_result_list[i])), "reduce").to(device) for i in batch_indices
                        ])

                        pretrain_out = pretrain_model(
                            norm_cell=norm_cell_batch,
                            reduce_cell=norm_cell_batch
                        )
                        pred_metrics = pretrain_out['metric_pred'].squeeze()
                        metric_ids_batch = torch.stack([
                            torch.arange(len(metric_names)).to(device) for _ in batch_indices
                        ])
                        metrics_batch = pred_metrics

                        online_out = online_trainer.model(
                            norm_cell=norm_cell_batch,
                            reduce_cell=reduce_cell_batch,
                            metric_ids=metric_ids_batch,
                            metrics=metrics_batch
                        )

                        final_scores = online_out['score'].squeeze().cpu().tolist()
                        for score, idx in zip(final_scores, batch_indices):
                            pop_scores.append((score, idx))

                        pbar.update(1)
                        pbar.set_postfix(
                            current_batch=f"{min(batch_indices)}-{max(batch_indices)}",
                            batch_size=len(batch_indices)
                        )

            num_excellent = 50
            sorted_scores = sorted(pop_scores, key=lambda x: x[0], reverse=True)
            excellent_pop_archs_index = [item[1] for item in sorted_scores[:num_excellent]]
            excellent_pop_archs = [evo_result_list[i] for i in excellent_pop_archs_index]

            metric_lists = []
            for i in tqdm(excellent_pop_archs_index, desc="Compute metrics", file=sys.stdout):
                geno = evo_result_list[i]
                model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, geno).cuda()
                model.drop_path_prob = 0
                metric_list = compute_metrics(model, inputs, targets)
                metric_lists.append(metric_list)

            data_metrics_pop = {}
            for metric in metric_names:
                data_metrics_pop[metric] = []
                for i in range(len(metric_lists)):
                    data_metrics_pop[metric].append(metric_lists[i][metric])

            # normalization
            for i, metric in enumerate(metric_names):
                max_value = max_vals[i].item()
                min_value = min_vals[i].item()

                if max_value - min_value == 0:
                    data_metrics_pop[metric] = [float(i) for i in data_metrics_pop[metric]]
                else:
                    data_metrics_pop[metric] = [(float(i) - min_value) / (max_value - min_value) for i in
                                              data_metrics_pop[metric]]

            num_arch = len(id_to_key)
            for geno in evo_result_list:
                id_to_key[num_arch] = geno
                num_arch += 1

            dataset = add_new_data(dataset, excellent_pop_archs, data_metrics_pop, metric_names, (current_cycle - args.init_epochs), args.n_sample, num_excellent)

            with torch.no_grad():
                total_archs = num_excellent
                batch_size = 1
                indices = list(range(total_archs))
                batches = [indices[i:i + batch_size] for i in range(0, total_archs, batch_size)]
                with tqdm(total=len(batches), desc="Evaluate arch", unit="batch",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                          file=sys.stdout) as pbar:
                    i = 0
                    best_pop_score = -100
                    for batch_indices in batches:
                        norm_cell_batch = Batch.from_data_list([
                            dataset['normal_cells'][args.n_sample + i + (current_cycle - args.init_epochs) * num_excellent].to(device) for i in batch_indices
                        ])
                        reduce_cell_batch = Batch.from_data_list([
                            dataset['reduce_cells'][args.n_sample + i + (current_cycle - args.init_epochs) * num_excellent].to(device) for i in batch_indices
                        ])
                        metric_ids_batch = torch.stack([
                            torch.arange(len(metric_names)).to(device) for _ in batch_indices
                        ])
                        metrics_batch = torch.stack([
                            dataset['metrics'][args.n_sample + i + (current_cycle - args.init_epochs) * num_excellent].to(device) for i in batch_indices
                        ])

                        online_out = online_trainer.model(
                            norm_cell=norm_cell_batch,
                            reduce_cell=reduce_cell_batch,
                            metric_ids=metric_ids_batch,
                            metrics=metrics_batch
                        )

                        final_scores = online_out['score'].cpu().item()
                        if final_scores > best_pop_score:
                            best_pop_score = final_scores
                        scores.append((final_scores, args.n_sample + i + (current_cycle - args.init_epochs) * num_excellent))
                        i += 1

                        pbar.update(1)
                        pbar.set_postfix(
                            current_batch=f"{min(batch_indices)}-{max(batch_indices)}",
                            batch_size=len(batch_indices)
                        )
            logging.info('best_pop_score: {}'.format(best_pop_score))
            sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

        for score, arch in sorted_scores:
            if arch not in val_accs:
                selected_arch = arch
                top_score = score
                break
            else:
                print(f"Arch {arch} has been evaluated  |  Val: {val_accs[arch]:.4f}")

        print(f"Selected Arch: {selected_arch},Score: {top_score:.4f}")

        if selected_arch not in val_accs:
            logging.info("current opt_arch:" + str(selected_arch))
            opt_archs.append(selected_arch)
            if selected_arch >= args.n_sample:
                opt_geno = id_to_key[selected_arch]
            else:
                opt_key = id_to_key[selected_arch]
                opt_geno = sampled_genos[opt_key]
            opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
            val_acc = train(data_queues, opt_model)
            val_accs[selected_arch] = val_acc
            logging.info(f'current val: [ {selected_arch}: {val_acc} ]')

        best_id, best_acc = max(val_accs.items(), key=lambda x: x[1])
        logging.info(f'best val: [ {best_id}: {best_acc:.2f}% ]')

        train_data = [{
            'norm_cell': dataset['normal_cells'][selected_arch],
            'reduce_cell': dataset['reduce_cells'][selected_arch],
            'metric_ids': torch.arange(len(metric_names)),
            'metrics': dataset['metrics'][selected_arch],
            'true_score': torch.tensor([val_accs[selected_arch]])
        }]
        loss = online_trainer.select_and_train(step, train_data)
        print(f"Surrogate model final loss: {loss}")

        current_cycle += 1

    with torch.no_grad():
        total_archs = len(dataset["arch_ids"])
        scores = []
        batch_size = 128

        indices = list(range(total_archs))
        batches = [indices[i:i + batch_size] for i in range(0, total_archs, batch_size)]

        with tqdm(total=len(batches), desc="Evaluate arch", unit="batch",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                  file=sys.stdout) as pbar:
            for batch_indices in batches:
                norm_cell_batch = Batch.from_data_list([
                    dataset['normal_cells'][i].to(device) for i in batch_indices
                ])
                reduce_cell_batch = Batch.from_data_list([
                    dataset['reduce_cells'][i].to(device) for i in batch_indices
                ])
                metric_ids_batch = torch.stack([
                    torch.arange(len(metric_names)).to(device) for _ in batch_indices
                ])
                metrics_batch = torch.stack([
                    dataset['metrics'][i].to(device) for i in batch_indices
                ])

                online_out = online_trainer.model(
                    norm_cell=norm_cell_batch,
                    reduce_cell=reduce_cell_batch,
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
    N = 3
    sorted_scores = sorted(scores, key=lambda x: -x[0])

    excellent_archs = []
    excellent_scores = []
    for score, idx in sorted_scores:
        if idx not in opt_archs:
            excellent_archs.append(idx)
            excellent_scores.append(score)
            if len(excellent_archs) == N:
                break
    formatted_scores = [f"{s:.4f}" for s in excellent_scores]
    print(f"Selected Arch: {excellent_archs},Score: {formatted_scores}")

    for arch in excellent_archs:
        if arch not in opt_archs:
            logging.info("current opt_arch:" + str(arch))
            opt_archs.append(arch)
            opt_key = id_to_key[arch]
            if arch >= args.n_sample:
                opt_geno = opt_key
            else:
                opt_geno = sampled_genos[opt_key]
            opt_model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, opt_geno).cuda()
            val_acc = train(data_queues, opt_model)
            val_accs[arch] = val_acc

    opt_acc = min(val_accs.values()) - 1
    opt_arch = 0
    for i in range(len(opt_archs)):
        arch = opt_archs[i]
        val_acc = val_accs[arch]
        if val_acc > opt_acc:
            opt_acc = val_acc
            opt_arch = arch

    opt_geno = id_to_key[opt_arch]

    logging.info('Search cost = %.2f(h)' % ((time.time() - start) / 3600,))
    logging.info('best val: {}'.format(opt_acc))
    logging.info('Best index = %s' % (opt_arch,))
    logging.info('Genotype = %s' % (opt_geno,))


def get_pool(data_queues, args):
    size=[14 * 2, 7]
    train_queue, _ = data_queues

    if not os.path.exists(args.arch_path):
        start = time.time()

        logging.info('Start sampling architectures...')

        sampled_genos, opt_genos, sampled_metrics = {}, {}, {}

        new_weights = [np.random.random_sample(size) for _ in range(args.n_sample)]
        new_genos = [genotype(w.reshape(2, -1, size[-1])) for w in new_weights]
        new_keys = list(map(str, new_genos))

        sampled_genos = dict(zip(new_keys, new_genos))

        inputs, targets = next(iter(train_queue))
        inputs, targets = inputs[:args.metric_batch_size].cuda(), targets[:args.metric_batch_size].cuda()

        for i, (k, geno) in enumerate(sampled_genos.items()):
            if i % 1000 == 0:
                logging.info('Start computing the metrics for arch %06d' % (i,))

            model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, geno).cuda()
            model.drop_path_prob = 0
            metric_list = compute_metrics(model, inputs, targets)
            sampled_metrics.update({k: metric_list})

        with open(args.arch_path, 'wb') as f:
            pickle.dump([sampled_genos, opt_genos, sampled_metrics], f)

        logging.info('Sampling cost=%.2f(h)' % ((time.time()- start) / 3600, ))
    else:
        with open(args.arch_path, 'rb') as f:
            sampled_genos, opt_genos, sampled_metrics = pickle.load(f)

    return sampled_genos, opt_genos, sampled_metrics

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


def compute_metrics(net, inputs, targets):
    metric_list = {}
    metric_list['fisher'] = sum_arr(fisher.compute_fisher_per_weight(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy, "channel"))
    metric_list['grad_norm'] = sum_arr(grad_norm.get_grad_norm_arr(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy))
    metric_list['snip'] = sum_arr(snip.compute_snip_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    metric_list['synflow'] = sum_arr(synflow.compute_synflow_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param"))
    metric_list['jacov'] = jacov.compute_jacob_cov(copy.deepcopy(net).cuda(), inputs, targets)
    metric_list['grasp'] = sum_arr(grasp.compute_grasp_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    return metric_list

def train(data_queues, model):
    train_queue, valid_queue = data_queues

    if 'imagenet' in args.dataset:
        criterion = utils.CrossEntropyLabelSmooth(NUM_CLASSES, args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    model.train()

    for epoch in range(args.epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        logging.info('epoch %d lr %e drop_prob %e', epoch, scheduler.get_last_lr()[0], model.drop_path_prob)

        for step, (input, target) in enumerate(train_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += args.auxiliary_weight*loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        scheduler.step()

    valid_acc = infer(valid_queue, model, criterion)

    return valid_acc

def infer(valid_queue, model, criterion):
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %f', step, top1.avg)

    return top1.avg

def genotype(weights, steps=4, multiplier=4):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(
                W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal = _parse(softmax(weights[0], axis=-1))
    gene_reduce = _parse(softmax(weights[1], axis=-1))

    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def parse_genotype(geno_str):

    pattern = re.compile(
        r"normal=\[(.*?)\]\s*,\s*normal_concat=.*?,\s*reduce=\[(.*?)\]\s*,\s*reduce_concat=.*",
        re.DOTALL
    )
    match = pattern.search(geno_str.replace("\n", " "))

    if not match:
        raise ValueError(f"The Genotype string cannot be parsed: {geno_str[:50]}...")

    normal_ops_str = match.group(1)
    normal_ops = re.findall(r"\('([^']+)', (\d+)\)", normal_ops_str)
    normal_ops = [(op, int(src)) for op, src in normal_ops]

    reduce_ops_str = match.group(2)
    reduce_ops = re.findall(r"\('([^']+)', (\d+)\)", reduce_ops_str)
    reduce_ops = [(op, int(src)) for op, src in reduce_ops]

    return Genotype(
        normal=normal_ops,
        normal_concat=range(2, 6),
        reduce=reduce_ops,
        reduce_concat=range(2, 6)
    )

def build_dynamic_dataset_with_id(sampled_metrics, metric_names, save_path="data/gnn_data/gnn_dataset.pt"):

    normal_graphs = []
    reduce_graphs = []
    metrics_tensors = []
    arch_ids = []

    for idx, item in enumerate(sampled_metrics):
        arch_ids.append(idx)

        geno_str = item["genotype"]
        geno = parse_genotype(geno_str)

        normal_data = genotype_to_gnn_data(geno, "normal")
        reduce_data = genotype_to_gnn_data(geno, "reduce")

        metric_values = [
            item["metrics"][name]
            for name in metric_names
            if name in item["metrics"]
        ]
        assert len(metric_values) == len(metric_names), f"Metrics lost: {geno_str}"

        normal_graphs.append(normal_data)
        reduce_graphs.append(reduce_data)
        metrics_tensors.append(torch.tensor(metric_values, dtype=torch.float32))

    metrics_tensor = torch.stack(metrics_tensors)

    min_vals, _ = metrics_tensor.min(dim=0, keepdim=True)
    max_vals, _ = metrics_tensor.max(dim=0, keepdim=True)

    ranges = max_vals - min_vals

    metrics_normalized = (metrics_tensor - min_vals) / ranges

    dataset = {
        "arch_ids": torch.tensor(arch_ids, dtype=torch.int64),
        "normal_cells": normal_graphs,
        "reduce_cells": reduce_graphs,
        "metrics": metrics_normalized,
        "metric_names": metric_names,
        "min_vals": min_vals,
        "max_vals": max_vals
    }

    torch.save(dataset, save_path)
    print(f"Dataset generated | Number of architectures: {len(arch_ids)} | ID: [{min(arch_ids)}, {max(arch_ids)}]")\



def add_new_data(dataset, new_samples, data_metrics_pop, metric_names, iter, n_sample, num_excellent):

    normal_graphs = []
    reduce_graphs = []
    metrics_tensors = []
    arch_ids = []
    i = 0

    for arch in new_samples:

        arch_ids.append(n_sample + iter * num_excellent + i)

        geno_str = str(arch)
        geno = parse_genotype(geno_str)

        normal_data = genotype_to_gnn_data(geno, "normal")
        reduce_data = genotype_to_gnn_data(geno, "reduce")

        metric_values = [
            data_metrics_pop[name][i]
            for name in metric_names
        ]
        assert len(metric_values) == len(metric_names), f"Metrics lost: {geno_str}"

        normal_graphs.append(normal_data)
        reduce_graphs.append(reduce_data)
        metrics_tensors.append(torch.tensor(metric_values, dtype=torch.float32))

        i += 1

    metrics_tensor = torch.stack(metrics_tensors)

    dataset["normal_cells"].extend(normal_graphs)
    dataset["reduce_cells"].extend(reduce_graphs)

    dataset["arch_ids"] = torch.cat([
        dataset["arch_ids"],
        torch.tensor(arch_ids, dtype=torch.int64)
    ], dim=0)

    dataset["metrics"] = torch.cat([
        dataset["metrics"],
        metrics_tensor
    ], dim=0)
    return dataset

def get_model_hash(model):
    return hashlib.sha256(
        b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    ).hexdigest()

if __name__ == '__main__':
    main()