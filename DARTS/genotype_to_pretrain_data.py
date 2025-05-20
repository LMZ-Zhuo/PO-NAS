import torch
from torch_geometric.data import Data


def genotype_to_gnn_data(genotype, cell_type):
    operation_to_id = {
        'max_pool_3x3': 0,
        'avg_pool_3x3': 1,
        'skip_connect': 2,
        'sep_conv_3x3': 3,
        'sep_conv_5x5': 4,
        'dil_conv_3x3': 5,
        'dil_conv_5x5': 6
    }
    num_ops = len(operation_to_id)
    total_nodes = 6
    node_features = torch.zeros((total_nodes, 6 + 1 + num_ops))

    edges = []
    edge_attrs = []

    ops = genotype.normal if cell_type == "normal" else genotype.reduce

    for step in range(4):
        target_node = 2 + step

        candidate_mask = [1 if i < target_node else 0 for i in range(6)]
        node_features[target_node, :6] = torch.tensor(candidate_mask)

        node_features[target_node, 6] = step

        op1, src1 = ops[step * 2]
        op2, src2 = ops[step * 2 + 1]

        edges.append((src1, target_node))
        edges.append((src2, target_node))

        edge_attr1 = torch.zeros(num_ops)
        edge_attr1[operation_to_id[op1]] = 1
        edge_attrs.append(edge_attr1)

        edge_attr2 = torch.zeros(num_ops)
        edge_attr2[operation_to_id[op2]] = 1
        edge_attrs.append(edge_attr2)

        node_features[target_node, 7:] += edge_attr1
        node_features[target_node, 7:] += edge_attr2

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=total_nodes
    )

