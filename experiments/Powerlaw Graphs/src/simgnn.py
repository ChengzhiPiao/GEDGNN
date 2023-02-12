"""SimGNN class and runner."""

import sys
import time

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.nn.conv import GCNConv, GINConv
from layers import AttentionModule, TenorNetworkModule, sinkhorn, MatchingModule
from GedMatrix import GedMatrixModule, SimpleMatrixModule, fixed_mapping_loss
from utils import load_all_graphs, load_labels, load_ged_full, load_ged_full_simple, load_features
import matplotlib.pyplot as plt
# from kbest_matching import KBestMSolver
from kbest_matching_with_lb import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau

class GPN(torch.nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GPN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.feature_count = self.args.tensor_neurons
        self.args.gnn_operator = 'gin'

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
            self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
            self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.number_labels, self.args.filters_1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_1, self.args.filters_1),
                torch.nn.BatchNorm1d(self.args.filters_1))

            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_1, self.args.filters_2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_2, self.args.filters_2),
                torch.nn.BatchNorm1d(self.args.filters_2))

            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filters_2, self.args.filters_3),
                torch.nn.ReLU(),
                torch.nn.Linear(self.args.filters_3, self.args.filters_3),
                torch.nn.BatchNorm1d(self.args.filters_3))

            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.matching_1 = MatchingModule(self.args)
        self.matching_2 = MatchingModule(self.args)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        #using_dropout = self.training
        using_dropout = False
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, p=self.args.dropout, training=using_dropout)
        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        tmp_feature_1 = abstract_features_1
        tmp_feature_2 = abstract_features_2

        abstract_features_1 = torch.sub(tmp_feature_1, self.matching_2(tmp_feature_2))
        abstract_features_2 = torch.sub(tmp_feature_2, self.matching_1(tmp_feature_1))

        abstract_features_1 = torch.abs(abstract_features_1)
        abstract_features_2 = torch.abs(abstract_features_2)

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score.view(-1)

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def set_map_matrix(self):
        self.mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        if self.args.has_map_matrix:
            self.mapMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        self.costMatrix = GedMatrixModule(self.args.filters_3, self.args.hidden_dim)
        # self.costMatrix = SimpleMatrixModule(self.args.filters_3)

        # bias
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.fully_connected_second = torch.nn.Linear(self.args.bottle_neck_neurons,
                                                      self.args.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.args.bottle_neck_neurons_2,
                                                     self.args.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons_3, 1)
        # self.bias_model = torch.nn.Linear(2, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def forward(self, data, is_testing=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        A_1 = data["A_1"]
        A_2 = data["A_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        gt_mapping = data["mapping"]

        n1 = features_1.shape[0]
        n2 = features_2.shape[0]
        # assert n1 == mapping.sum().item()

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)

        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        soft_matrix = m(map_matrix) * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if is_testing:
            mapping_ged = -torch.log(score) * data["avg_v"]
            return map_matrix, score, mapping_ged
        else:
            return map_matrix, score

        """ Use gt_mapping to predict ged.
        if predict_value:
            m = torch.nn.Softmax(dim=1)
            map_matrix = m(gt_mapping)
            soft_matrix = map_matrix * cost_matrix

            bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
            # bias_value = self.bias_model(torch.tensor([n1 * 1.0, n2 * 1.0]))   #simple bias
            score = torch.sigmoid(soft_matrix.sum() + bias_value)
            return score

            row_sum = mapping.sum(dim=1, keepdim=True)
            cost_matrix =  (mapping / row_sum) * cost_matrix
            matrix_score = cost_matrix.sum()
        """


class GPNTrainer(object):
    """
    GPN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.training_time = 0.0
        self.training_loss = 0.0
        self.testing_graph_set = []
        self.testing_time = []
        self.error = []
        self.m_error = []
        self.ged_MAE = []
        self.ged_MSE = []
        self.match_MAE = []
        self.match_MSE = []
        self.testing_results = []
        self.results = []

        # self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.load_data()
        self.transfer_data_to_torch()
        self.init_graph_pairs()
        # del self.graphs

        self.setup_model()

    def setup_model(self):
        self.model = GPN(self.args, self.number_of_labels).to(self.device)

    def load_data(self):
        """
        Load graphs, ged and labels if needed.
        self.ged: dict-dict, ged['graph_id_1']['graph_id_2'] stores the ged value.
        """
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0

        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)

        '''
        #load anchor based features
        feature_dim, features = load_features(self.args.abs_path, dataset_name, 'anchor')
        self.number_of_labels += feature_dim
        self.features_2 = features
        '''

        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        # print(self.global_labels)
        self.ged_dict = load_ged_full(self.args.abs_path, dataset_name)
        print("Load ged dict.")
        # print(self.ged['2050']['30'])
        t2 = time.time()
        self.load_data_time = t2 - t1

    def transfer_data_to_torch(self):
        """
        Transfer loaded data to torch.
        """
        t1 = time.time()

        self.edge_index = []
        self.A = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long().to(self.device)
            self.edge_index.append(edge)
            A = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (g['n'], g['n'])).to_dense().to(self.device)
            self.A.append(A)

        features, features_2 = None, None
        if hasattr(self, 'features'):
            features = [torch.tensor(x).float().to(self.device) for x in self.features]
        if hasattr(self, 'features_2'):
            features_2 = [torch.tensor(x).float().to(self.device) for x in self.features_2]
        if features is None:
            self.features = features_2
        elif features_2 is None:
            self.features = features
        else:
            self.features = [torch.cat((x1, x2), 1) for x1, x2 in zip(features, features_2)]
        print("Feature shape of 1st graph:", self.features[0].shape)
        del features
        del features_2

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[0.0 for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            # mapping[i][i] = torch.tensor([x for x in range(self.gn[i])]).long().to(self.device)
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float, device=self.device)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                real_ged, gt_mappings = self.ged_dict[id_pair]
                ged[i][j] = ged[j][i] = real_ged
                mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                # mapping_matrix = torch.zeros([n1, n2]).to(self.device)
                for gt_mapping in gt_mappings:
                    for x, y in enumerate(gt_mapping):
                        mapping_list[x][y] = 1
                mapping_matrix = torch.tensor(mapping_list).float().to(self.device)
                # mapping_matrix = F.normalize(mapping_matrix)
                mapping[i][j] = mapping[j][i] = mapping_matrix
                # mapping[i][j] = mapping[j][i] = torch.tensor(gt_mapping).long().to(self.device)
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1

    def init_graph_pairs(self):
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing2_graphs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        if self.args.demo:
            train_num = 30
            val_num = 40
            test_num = 50
            self.args.epochs = 1

        if self.args.demo_testing == "final":
            random.seed(1)
            tmp = 100
            li = list(range(train_num))
            for i in range(val_num, test_num):
                random.shuffle(li)
                self.testing_graphs.append((i, li[:tmp]))
            li = list(range(val_num, test_num))
            for i in range(val_num, test_num):
                random.shuffle(li)
                self.testing2_graphs.append((i, li[:tmp]))
        elif self.args.demo_testing == "simple":
            random.seed(1)
            li = list(range(val_num))
            for i in range(train_num, val_num):
                self.val_graphs.append((i, random.choice(li)))
            for i in range(val_num, test_num):
                self.testing_graphs.append((i, random.choice(li)))
            li = list(range(val_num, test_num))
            for i in range(val_num, test_num):
                self.testing2_graphs.append((i, random.choice(li)))
        elif self.args.demo_testing == "normal":
            random.seed(1)
            tmp = 20
            li = list(range(val_num))
            for i in range(train_num, val_num):
                random.shuffle(li)
                for j in li[:tmp]:
                    self.val_graphs.append((i, j))

            for i in range(val_num, test_num):
                random.shuffle(li)
                for j in li[:tmp]:
                    self.testing_graphs.append((i, j))

            li = list(range(val_num, test_num))
            for i in range(val_num, test_num):
                random.shuffle(li)
                for j in li[:tmp]:
                    self.testing2_graphs.append((i, j))
        else:
            for i in range(train_num):
                for j in range(val_num):
                    self.training_graphs.append((i, j))

            for i in range(train_num, val_num):
                for j in range(val_num):
                    self.val_graphs.append((i, j))

            for i in range(val_num, test_num):
                for j in range(val_num):
                    self.testing_graphs.append((i, j))

            for i in range(val_num, test_num):
                for j in range(val_num, test_num):
                    self.testing2_graphs.append((i, j))

        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} val graph pairs.".format(len(self.val_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        print("Generate {} testing2 graph pairs.".format(len(self.testing2_graphs)))

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph + self.args.batch_size])
        return batches

    def pack_graph_pair_with_mapping(self, graph_pair):
        """
        Prepare the graph pair data for simgnn model.
        :param graph_pair: (id_1, id_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (id_1, id_2) = graph_pair
        gid_pair = (self.gid[id_1], self.gid[id_2])
        if gid_pair not in self.ged_dict:
            (id_2, id_1) = graph_pair

        real_ged = self.ged[id_1][id_2]

        new_data["id_1"] = id_1
        new_data["id_2"] = id_2

        new_data["edge_index_1"] = self.edge_index[id_1]
        new_data["edge_index_2"] = self.edge_index[id_2]
        new_data["features_1"] = self.features[id_1]
        new_data["features_2"] = self.features[id_2]

        # avg_v = (self.gn[id_1] + self.gn[id_2]) / 2.0
        # new_data["avg_v"] = torch.tensor([avg_v]).float().to(self.device)
        # new_data["avg_v"] = avg_v
        new_data["ged"] = real_ged
        # new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
        higher_bound = max(self.gn[id_1], self.gn[id_2]) + max(self.gm[id_1], self.gm[id_2])
        new_data["hb"] = higher_bound
        new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)

        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        for graph_pair in batch:
            data = self.pack_graph_pair_with_mapping(graph_pair)
            target = data["target"]
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(target, prediction)
            # self.values.append((target - prediction).item())

        losses.backward()
        self.optimizer.step()
        return losses.item()

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        self.values = []
        with tqdm(total=self.args.epochs * len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",
                  file=sys.stdout) as pbar:
            for epoch in range(self.args.epochs):
                batches = self.create_batches()
                self.loss_sum = 0
                main_index = 0
                for index, batch in enumerate(batches):
                    batch_total_loss = self.process_batch(batch)  # without average
                    self.loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = self.loss_sum / main_index  # the average loss of current epoch
                    pbar.update(len(batch))
                    pbar.set_description(
                        "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.args.model_epoch + 1, round(1000 * loss, 3),
                                                                       index,
                                                                       round(1000 * batch_total_loss / len(batch), 3)))
                tqdm.write("Epoch {}: loss={}".format(self.args.model_epoch + 1, round(1000 * loss, 3)))
                self.training_loss = round(1000 * loss, 3)
        t2 = time.time()
        self.training_time = t2 - t1
        if len(self.values) > 0:
            self.prediction_analysis(self.values, "training_score")

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def final_score(self, testing_graph_set='test'):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        self.testing_graph_set.append(testing_graph_set)
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        elif testing_graph_set == 'train':
            testing_graphs = self.training_graphs
        else:
            assert False

        self.model.eval()
        # self.model.train()

        num = 0 # total testing number
        time_usage = []
        mse = []    # score mse
        mae = []    # ged mae
        num_acc = 0 # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0 # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        debug = 0

        for i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair_with_mapping((i, j))
                target, gt_ged, hb = data["target"].item(), data["ged"], data["hb"]
                prediction = self.model(data).item()

                num += 1
                mse.append((prediction - target)**2)
                pre_ged = prediction * hb
                round_pre_ged = round(pre_ged)
                pre.append(pre_ged)
                gt.append(gt_ged)

                """
                print(i, j, hb, gt_ged, pre_ged)
                debug += 1
                if debug == 10:
                    exit(0)
                """

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)
            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        #print(mae[:10])

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)
        self.results.append((testing_graph_set, num, time_usage, mse, mae, acc, fea, rho, tau, pk10, pk20))

    def print_final_results(self):
        method_name = 'Noah-GPN'
        for r in self.results:
            print(method_name, self.args.dataset, *r, sep='\t')

        with open(self.args.abs_path + self.args.result_path + 'Final_results.txt', 'a') as f:
            for r in self.results:
                print(method_name, self.args.dataset, *r, sep='\t', file=f)

    def score(self, testing_graph_set='test', demo=False):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        self.testing_graph_set.append(testing_graph_set)
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        elif testing_graph_set == 'train':
            testing_graphs = self.training_graphs
        else:
            assert False

        t1 = time.time()
        self.model.eval()
        scores = []
        scores2 = []
        ged_MAE = []
        ged_MSE = []

        values = []

        if demo:
            # randomly select 1% testing graphs for a demo testing
            random.shuffle(testing_graphs)
            demo_num = len(testing_graphs) // 10
            testing_graphs = testing_graphs[:demo_num]

        debug = 0

        # for graph_pair in tqdm(testing_graphs, file=sys.stdout):
        for i, j_list in tqdm(testing_graphs, file=sys.stdout):
            for j in j_list:
                graph_pair = (i, j)
                data = self.pack_graph_pair_with_mapping(graph_pair)
                target, gt_ged, hb = data["target"], data["ged"], data["hb"]
                # gt_ged = torch.tensor([gt_ged])
                target = target.item()
                prediction = self.model(data).item()
                pre_ged = round(prediction * hb)
                scores.append((prediction - target)**2)

                """
                print(i, j, hb, gt_ged, pre_ged)
                debug+=1
                if debug == 10:
                    exit(0)
                """

                ged_MAE.append(abs(pre_ged - gt_ged))
                ged_MSE.append((pre_ged - gt_ged)**2)
                #scores.append(F.mse_loss(prediction, target).item())
                #ged_MAE.append(torch.nn.functional.l1_loss(pre_ged, gt_ged).item())
                #ged_MSE.append(torch.nn.functional.mse_loss(pre_ged, gt_ged).item())
                # values.append((mapping_ged - gt_ged).item())

        print(ged_MAE[:10])
        self.error.append(round(1000 * np.mean(scores), 0))
        self.ged_MAE.append(round(np.mean(ged_MAE), 3))
        self.ged_MSE.append(round(np.mean(ged_MSE), 3))
        t2 = time.time()
        self.testing_time.append(round(t2 - t1, 1))

        if len(values) > 0:
            self.prediction_analysis(values, testing_graph_set)

        # self.plot_error(match_MAE, dataset=testing_graph_set)
        # self.plot_error2d(match_MAE, groundtruth, dataset=testing_graph_set)

    def print_simple_result(self):
        print(self.args.model_epoch + self.args.epochs, end='\t')
        print(1000 * self.args.learning_rate, end='\t')
        print(round(self.training_time, 1), end='\t')
        print(self.training_loss, end='\t')
        for i in range(len(self.error)):
            print(self.testing_graph_set[i], self.error[i], self.ged_MAE[i], self.ged_MSE[i], sep='\t', end='\t')
        print('')
        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print(self.args.model_epoch + self.args.epochs, end='\t', file=f)
            # print(1000 * self.args.learning_rate, end='\t', file=f)
            print(round(self.training_time, 1), end='\t', file=f)
            print(self.training_loss, end='\t', file=f)
            for i in range(len(self.error)):
                print(self.testing_graph_set[i], self.error[i], self.ged_MAE[i], self.ged_MSE[i], sep='\t', end='\t',
                      file=f)
            print('', file=f)


    def prediction_analysis(self, values, info_str=''):
        """
        Analyze the performance of value prediction.
        :param values: an array of (pre_ged - gt_ged); Note that there is no abs function.
        """
        if not self.args.prediction_analysis:
            return
        neg_num = 0
        pos_num = 0
        pos_error = 0.
        neg_error = 0.
        for v in values:
            if v >= 0:
                pos_num += 1
                pos_error += v
            else:
                neg_num += 1
                neg_error += v

        tot_num = neg_num + pos_num
        tot_error = pos_error - neg_error

        pos_error = round(pos_error / pos_num, 3) if pos_num > 0 else None
        neg_error = round(neg_error / neg_num, 3) if neg_num > 0 else None
        tot_error = round(tot_error / tot_num, 3) if tot_num > 0 else None

        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print("prediction_analysis", info_str, sep='\t', file=f)
            print("num", pos_num, neg_num, tot_num, sep='\t', file=f)
            print("err", pos_error, neg_error, tot_error, sep='\t', file=f)
            print("--------------------", file=f)

    def demo_testing(self, testing_graph_set='test'):
        print("\n\nDemo testing on {} set.\n".format(testing_graph_set))
        self.testing_graph_set.append(testing_graph_set)
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        elif testing_graph_set == 'train':
            testing_graphs = self.training_graphs
        else:
            assert False

        self.model.eval()

        # demo_num = 10
        demo_num = len(testing_graphs)
        # random.shuffle(testing_graphs)
        testing_graphs = testing_graphs[:demo_num]
        total_num = 0
        num_10 = 0
        num_100 = 0
        num_1000 = 0
        score_10 = [[], [], []]
        score_100 = [[], [], []]
        score_1000 = [[], [], []]

        values0 = []
        values1 = []
        values2 = []
        values3 = []

        m = torch.nn.Softmax(dim=1)
        for graph_pair in tqdm(testing_graphs, file=sys.stdout):
            data = self.pack_graph_pair_with_mapping(graph_pair)
            avg_v = data["avg_v"]  # (n1+n2)/2.0, a scalar, not a tensor
            gt_ged, target = data["ged"], data["target"]  # gt ged value and score
            soft_matrix, _, prediction = self.model(data, is_testing=True)
            pre_ged, gt_ged, gt_score = prediction.item(), gt_ged.item(), target.item()

            values0.append(pre_ged - gt_ged)

            soft_matrix = (torch.sigmoid(soft_matrix) * 1e9 + 1).round()
            # soft_matrix = (m(soft_matrix) * 1e9 + 1).int()
            # soft_matrix = ((soft_matrix - soft_matrix.min()) * 1e9 + 1).round()

            n1, n2 = soft_matrix.shape
            #print(data["edge_index_1"].shape)
            g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=n1)
            g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=n2)
            g1.ndata['f'] = data["features_1"]
            g2.ndata['f'] = data["features_2"]

            # if n1 < 10 or n2 < 10:
            #   continue

            total_num += 1
            test_k = self.args.postk

            solver = KBestMSolver(soft_matrix, g1, g2, pre_ged)
            for k in range(test_k):
                '''
                matching, weightsum, sp_ged = solver.get_matching(k + 1)
                if weightsum is None:
                    print(k, solver.min_ged, gt_ged)
                    break
                mapping = torch.zeros([n1, n2])
                for i, j in enumerate(matching):
                    mapping[i][j] = 1.0
                mapping_ged = self.model.ged_from_mapping(mapping, data["A_1"], data["A_2"], data["features_1"],
                                                          data["features_2"])
                min_res = min(min_res, mapping_ged.item())
                '''
                solver.get_matching(k + 1)
                min_res = solver.min_ged
                # a gt_mapping is found
                if abs(min_res - gt_ged) < 1e-12:
                    # fix pre_ged using lower bound
                    fixed_pre_ged = max(solver.lb_value, pre_ged)
                    # fix pre_ged using upper bound
                    if min_res < fixed_pre_ged:
                        fixed_pre_ged = min_res

                    fixed_pre_s = exp(-fixed_pre_ged / avg_v)
                    pre_score = abs(fixed_pre_ged - gt_ged)
                    pre_score2 = (fixed_pre_s - gt_score) ** 2
                    map_score = 0.0
                    if k < 10:
                        score_10[0].append(pre_score2)
                        score_10[1].append(pre_score)
                        score_10[2].append(map_score)
                        num_10 += 1
                        values1.append(fixed_pre_ged - gt_ged)
                    if k < 100:
                        score_100[0].append(pre_score2)
                        score_100[1].append(pre_score)
                        score_100[2].append(map_score)
                        num_100 += 1
                        values2.append(fixed_pre_ged - gt_ged)
                    if k < 1000:
                        score_1000[0].append(pre_score2)
                        score_1000[1].append(pre_score)
                        score_1000[2].append(map_score)
                        num_1000 += 1
                        values3.append(fixed_pre_ged - gt_ged)
                    break
                if k in [9, 99, 999]:
                    # fix pre_ged using lower bound
                    fixed_pre_ged = max(solver.lb_value, pre_ged)
                    # fix pre_ged using upper bound
                    if min_res < fixed_pre_ged:
                        fixed_pre_ged = min_res

                    fixed_pre_s = exp(-fixed_pre_ged / avg_v)
                    pre_score = abs(fixed_pre_ged - gt_ged)
                    pre_score2 = (fixed_pre_s - gt_score) ** 2
                    map_score = abs(min_res - gt_ged)
                    if k + 1 == 10:
                        score_10[0].append(pre_score2)
                        score_10[1].append(pre_score)
                        score_10[2].append(map_score)
                        values1.append(fixed_pre_ged - gt_ged)
                    elif k + 1 == 100:
                        score_100[0].append(pre_score2)
                        score_100[1].append(pre_score)
                        score_100[2].append(map_score)
                        values2.append(fixed_pre_ged - gt_ged)
                    elif k + 1 == 1000:
                        score_1000[0].append(pre_score2)
                        score_1000[1].append(pre_score)
                        score_1000[2].append(map_score)
                        values3.append(fixed_pre_ged - gt_ged)

        if test_k >= 10:
            print("10:", len(score_10[0]), round(np.mean(score_10[1]), 3), round(np.mean(score_10[2]), 3), sep='\t')
            print("{} / {} = {}".format(num_10, total_num, round(num_10 / total_num, 3)))
        if test_k >= 100:
            print("100:", len(score_100[0]), round(np.mean(score_100[1]), 3), round(np.mean(score_100[2]), 3), sep='\t')
            print("{} / {} = {}".format(num_100, total_num, round(num_100 / total_num, 3)))
        if test_k >= 1000:
            print("1000:", len(score_1000[0]), round(np.mean(score_1000[1]), 3), round(np.mean(score_1000[2]), 3), sep='\t')
            print("{} / {} = {}".format(num_1000, total_num, round(num_1000 / total_num, 3)))

        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print('', file=f)
            print(self.args.model_epoch, testing_graph_set, demo_num, sep='\t', file=f)
            if test_k >= 10:
                print("10", round(np.mean(score_10[0]) * 1000, 3), round(np.mean(score_10[1]), 3), round(np.mean(score_10[2]), 3), round(num_10 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_10, total_num, round(num_10 / total_num, 3)), file=f)
            if test_k >= 100:
                print("100", round(np.mean(score_100[0]) * 1000, 3), round(np.mean(score_100[1]), 3), round(np.mean(score_100[2]), 3), round(num_100 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_100, total_num, round(num_100 / total_num, 3)), file=f)
            if test_k >= 1000:
                print("1000", round(np.mean(score_1000[0]) * 1000, 3), round(np.mean(score_1000[1]), 3), round(np.mean(score_1000[2]), 3), round(num_1000 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_1000, total_num, round(num_1000 / total_num, 3)), file=f)
            # print('', file=f)

        self.prediction_analysis(values0, "base")
        if test_k >= 10:
            self.prediction_analysis(values1, "10")
        if test_k >= 100:
            self.prediction_analysis(values2, "100")
        if test_k >= 1000:
            self.prediction_analysis(values3, "1000")

    def plot_error(self, errors, dataset=''):
        name = self.args.dataset
        if dataset:
            name = name + '(' + dataset + ')'
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution on {}".format(name))

        bins = list(range(int(max(errors)) + 2))
        plt.hist(errors, bins=bins, density=True)
        plt.savefig(self.args.abs_path + self.args.result_path + name + '_error.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    def plot_error2d(self, errors, groundtruth, dataset=''):
        name = self.args.dataset
        if dataset:
            name = name + '(' + dataset + ')'
        plt.xlabel("Error")
        plt.ylabel("GroundTruth")
        plt.title("Error-GroundTruth Distribution on {}".format(name))

        # print(len(errors), len(groundtruth))
        errors = [round(x) for x in errors]
        groundtruth = [round(x) for x in groundtruth]
        plt.hist2d(errors, groundtruth, density=True)
        plt.colorbar()
        plt.savefig(self.args.abs_path + self.args.result_path + '' + name + '_error2d.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    def plot_results(self):
        results = torch.tensor(self.testing_results).t()
        name = self.args.dataset
        epoch = str(self.args.model_epoch + 1)
        n = results.shape[1]
        x = torch.linspace(1, n, n)
        plt.figure(figsize=(10, 4))
        plt.plot(x, results[0], color="red", linewidth=1, label='ground truth')
        plt.plot(x, results[1], color="black", linewidth=1, label='simgnn')
        plt.plot(x, results[2], color="blue", linewidth=1, label='matching')
        plt.xlabel("test_pair")
        plt.ylabel("ged")
        plt.title("{} Epoch-{} Results".format(name, epoch))
        plt.legend()
        # plt.ylim(-0.0,1.0)
        plt.savefig(self.args.abs_path + self.args.result_path + name + '_' + epoch + '.png', dpi=120,
                    bbox_inches='tight')
        # plt.show()

    '''
    def print_simple_result(self):
        print(self.args.model_epoch + self.args.epochs, end='\t')
        print(1000 * self.args.learning_rate, end='\t')
        print(round(self.training_time, 1), end='\t')
        print(self.training_loss, end='\t')
        for i in range(len(self.error)):
            print(self.testing_graph_set[i], self.error[i], self.m_error[i], self.ged_MAE[i], self.ged_MSE[i], self.match_MAE[i], self.match_MSE[i],
                  sep='\t', end='\t')
        print('')
        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print(self.args.model_epoch + self.args.epochs, end='\t', file=f)
            #print(1000 * self.args.learning_rate, end='\t', file=f)
            print(round(self.training_time, 1), end='\t', file=f)
            print(self.training_loss, end='\t', file=f)
            for i in range(len(self.error)):
                print(self.testing_graph_set[i], self.error[i], self.m_error[i], self.ged_MAE[i], self.ged_MSE[i], self.match_MAE[i], self.match_MSE[i],
                      sep='\t', end='\t', file=f)
            print('', file=f)
    '''

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print(self.args.model_epoch + self.args.epochs, end='\t', file=f)
            print(round(self.training_time, 1), end='\t', file=f)
            print(self.training_loss, end='\t', file=f)
            for i in range(len(self.error)):
                print(self.testing_graph_set[i], self.testing_time[i], self.error[i],
                      self.ged_MAE[i], self.ged_MSE[i], self.match_MAE[i], self.match_MSE[i], sep='\t', end='\t',
                      file=f)
            print('', file=f)

    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch))

    def load(self, epoch):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch)))
