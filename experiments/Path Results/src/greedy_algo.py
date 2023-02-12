import torch


def hungarian(data):
    edge_index_1 = data["edge_index_1"]
    edge_index_2 = data["edge_index_2"]
    n1 = data["n1"]
    n2 = data["n2"]
    adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.shape[1]), (n1, n1)).to_dense()
    adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.shape[1]), (n2, n2)).to_dense()
    f_1 = torch.mm(adj_1, data["features_1"])
    f_2 = torch.mm(adj_2, data["features_2"])

    hb = 2.0 * (n1 + n2)
    A = [[hb for i in range(n2)] for j in range(n1)]
    for u in range(n1):
        for v in range(n2):
            cost = f_1[u].sum() + f_2[v].sum() - 2.0 * torch.min(f_1[u], f_2[v]).sum()
            A[u][v] -= cost

    return torch.tensor(A)
