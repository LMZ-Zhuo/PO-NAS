import torch
from torch_geometric.data import Data

operation_to_id = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
}

def parse_arch_to_graph(arch):
    arch = list(map(int, arch.strip("()").split(", ")))
    num_nodes = 4
    edges = []
    edge_ops = []

    current_node = 1
    prev_node = 0
    for i, part in enumerate(arch):
        if i == 1 or i == 3:
            current_node += 1
            prev_node = 0
        edges.append((prev_node, current_node))
        edge_ops.append(part)
        prev_node += 1

    node_features = []
    for node in range(num_nodes):
        prev_nodes = [1 if i < node else 0 for i in range(4)]
        node_idx = [node]
        op_counts = [0] * 4
        for edge, op in zip(edges, edge_ops):
            if edge[1] == node:
                op_id = op
                op_counts[op_id] += 1
        node_features.append(prev_nodes + node_idx + op_counts)
    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    edge_attr = []
    for op in edge_ops:
        op_id = op
        op_one_hot = [0] * 4
        op_one_hot[op_id] = 1
        edge_attr.append(op_one_hot)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

def build_dynamic_dataset_with_id(task, data_stats, metric_names, data_metrics, save_path="data/pretrain_data/pretrain_dataset.pt"):

    arch_graphs = []
    metrics_tensors = []
    arch_ids = []

    for arch, item in data_stats[task].items():

        arch_ids.append(item['id'])

        arch_data = parse_arch_to_graph(arch)

        metric_values = [
            data_metrics[name][int(item['id'])]
            for name in metric_names
            if name in item
        ]

        arch_graphs.append(arch_data)
        metrics_tensors.append(torch.tensor(metric_values, dtype=torch.float32))

    metrics_tensor = torch.stack(metrics_tensors)

    dataset = {
        "arch_ids": torch.tensor(arch_ids, dtype=torch.int64),
        "arch": arch_graphs,
        "metrics": metrics_tensor,
        "metric_names": metric_names,
    }

    torch.save(dataset, save_path)
    print(f"Dataset generated | Number of architectures: {len(arch_ids)} | ID: [{min(arch_ids)}, {max(arch_ids)}]")

