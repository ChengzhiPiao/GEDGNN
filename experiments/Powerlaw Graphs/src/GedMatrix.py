import torch
import torch.nn

class GedMatrixModule(torch.nn.Module):
    """
    GED matrix module.
    d is the size of input feature;
    k is the size of hidden layer.

    Input: n1 * d, n2 * d
    step 1 matmul: (n1 * d) matmul (k * d * d) matmul (n2 * d).t() -> k * n1 * n2
    step 2 mlp(k, 2k, k, 1): k * n1 * n2 -> (n1n2) * k -> (n1n2) * 2k -> (n1n2) * k -> (n1n2) * 1 -> n1 * n2
    Output: n1 * n2
    """
    def __init__(self, d, k):
        """
        :param args: Arguments object.
        """
        super(GedMatrixModule, self).__init__()

        self.d = d
        self.k = k
        self.init_weight_matrix()
        self.init_mlp()

    def init_weight_matrix(self):
        """
        Define and initilize a weight matrix of size (k, d, d).
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.k, self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def init_mlp(self):
        """
        Define a mlp: k -> 2*k -> k -> 1
        """
        k = self.k
        layers = []

        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        # layers.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similar matrix.
        :param embedding_1: GCN(graph1) of size (n1, d)
        :param embedding_2: GCN(graph2) of size (n2, d)
        :return result: a similar matrix of size (n1, n2)
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2

        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        matrix = matrix.reshape(self.k, -1).t()
        matrix = self.mlp(matrix)

        return matrix.reshape(n1, n2)


class SimpleMatrixModule(torch.nn.Module):
    """
    Simple matrix module.
    d is the size of input feature;
    k is the size of hidden layer.

    Input: n1 * d, n2 * d
    step 1 matmul: (n1 * d) matmul (k * d * d) matmul (n2 * d).t() -> k * n1 * n2
    step 2 mlp(k, 2k, k, 1): k * n1 * n2 -> (n1n2) * k -> (n1n2) * 2k -> (n1n2) * k -> (n1n2) * 1 -> n1 * n2
    Output: n1 * n2
    """
    def __init__(self, k):
        """
        :param args: Arguments object.
        """
        super(SimpleMatrixModule, self).__init__()

        self.k = k
        self.init_mlp()

    def init_mlp(self):
        """
        Define a mlp: k -> 2*k -> k -> 1
        """
        k = self.k
        layers = []

        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        # layers.append(torch.nn.Sigmoid())

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similar matrix.
        :param embedding_1: GCN(graph1) of size (n1, d)
        :param embedding_2: GCN(graph2) of size (n2, d)
        :return result: a similar matrix of size (n1, n2)
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.k == d2

        tmp_1 = embedding_1.unsqueeze(1).repeat(1, n2, 1)  # n1*d -> n1 1 d -> n1 n2 d
        tmp_2 = embedding_2.unsqueeze(0).repeat(n1, 1, 1)  # n2*d -> 1 n2 d -> n1 n2 d
        matrix = (tmp_1.reshape([n1 * n2, -1]) * tmp_2.reshape([n1 * n2, -1])).reshape([n1 * n2, -1])

        matrix = self.mlp(matrix)

        return matrix.reshape(n1, n2)

def fixed_mapping_loss(mapping, gt_mapping):
    mapping_loss = torch.nn.BCEWithLogitsLoss()
    n1, n2 = mapping.shape

    epoch_percent = 1.0 / n2
    if epoch_percent >= 1.0:
        return mapping_loss(mapping, gt_mapping)

    num_1 = gt_mapping.sum().item()
    num_0 = n1 * n2 - num_1
    if num_1 >= num_0: # There is no need to use mask. Directly return the complete loss.
        return mapping_loss(mapping, gt_mapping)

    p_base = num_1 / num_0
    p = 1.0 - (p_base + epoch_percent * (1-p_base))

    #p = 1.0 - (epoch_num + 1.0) / 10
    mask = (torch.rand([n1, n2]) + gt_mapping) > p
    # gt_mask = gt_mapping > 0.5
    return mapping_loss(mapping[mask], gt_mapping[mask])
    # return 0.5 * mapping_loss(mapping[mask], gt_mapping[mask]) + 0.5 * mapping_loss(mapping[gt_mask], gt_mapping[gt_mask])

def fixed_mapping_loss1(mapping, gt_mapping):
    mapping_loss = torch.nn.BCEWithLogitsLoss()
    n1, n2 = mapping.shape
    p = 1.0 - 1.0 / n2
    mask = (torch.rand([n1, n2]) + gt_mapping) > p
    return mapping_loss(mapping[mask], gt_mapping[mask])

def fixed_mapping_loss2(mapping, gt_mapping):
    n1, n2 = mapping.shape
    mapping = torch.sigmoid(mapping)

    row_avg = (gt_mapping * mapping).sum(dim=1) / gt_mapping.sum(dim=1) # weighted avg by row
    gt_weight =  row_avg.sum() / n1

    base_weight = mapping.sum() / (n1 * n2)

    return base_weight - gt_weight

def fixed_mapping_loss3(mapping, gt_mapping):
    n1, n2 = mapping.shape
    #mapping = torch.sigmoid(mapping)

    m = torch.nn.Softmax(dim=1)
    #map_matrix = m(gt_mapping)

    return torch.nn.functional.mse_loss(m(mapping), m(gt_mapping))