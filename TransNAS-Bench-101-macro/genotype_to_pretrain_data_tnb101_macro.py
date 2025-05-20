import torch
from torch_geometric.data import Data


def parse_arch_to_graph(arch):

    arch = list(map(int, arch.strip("()").split(", ")))
    num_nodes = 7
    edges = []
    edge_ops = []

    current_node = 1
    prev_node = 0
    true_nodes = len(arch)
    for i, part in enumerate(arch):
        edges.append((prev_node, current_node))
        edge_ops.append(part)
        prev_node += 1
        current_node += 1
    for i in range(num_nodes - true_nodes - 1):
        edges.append((prev_node, current_node))
        prev_node += 1
        current_node += 1

    node_features = []
    for node in range(num_nodes):
        if node <= true_nodes:
            prev_nodes = [1]
        else:
            prev_nodes = [0]
        node_idx = [node]
        op_counts = [0] * 4
        for edge, op in zip(edges, edge_ops):
            if edge[1] == node:
                op_id = op - 1
                op_counts[op_id] += 1
        node_features.append(prev_nodes + node_idx + op_counts)
    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    edge_attr = []
    for op in edge_ops:
        op_id = op
        op_one_hot = [0] * 3
        if op_id == 2:
            op_one_hot[0] = 1
            op_one_hot[1] = 1
        elif op_id == 3:
            op_one_hot[0] = 1
            op_one_hot[2] = 1
        elif op_id == 4:
            op_one_hot[0] = 1
            op_one_hot[1] = 1
            op_one_hot[2] = 1
        else:
            op_one_hot[0] = 1
        edge_attr.append(op_one_hot)
    for i in range(num_nodes - true_nodes - 1):
        op_one_hot = [0] * 3
        edge_attr.append(op_one_hot)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

def build_dynamic_dataset_with_id(task, data_stats, metric_names, data_metrics, save_path="data/gnn_data/gnn_dataset.pt"):

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
