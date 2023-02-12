import sys
import time
from typing import List

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from kbest_matching_with_lb import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau

from models import GPN, SimGNN, GedGNN, TaGSim
from GedMatrix import fixed_mapping_loss


class Trainer(object):
    """
    A general model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []

        # self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        # self.gen_delta_graphs()
        self.init_graph_pairs()

        self.setup_model()

    def setup_model(self):
        if self.args.model_name == 'GPN':
            self.model = GPN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "SimGNN":
            self.args.filters_1 = 64
            self.args.filters_2 = 32
            self.args.filters_3 = 16
            self.args.histogram = True
            self.args.target_mode = 'exp'
            self.model = SimGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "GedGNN":
            if self.args.dataset in ["AIDS", "Linux"]:
                self.args.loss_weight = 10.0
            else:
                self.args.loss_weight = 1.0
            # self.args.target_mode = 'exp'
            self.args.gtmap = True
            self.model = GedGNN(self.args, self.number_of_labels).to(self.device)
        elif self.args.model_name == "TaGSim":
            self.args.target_mode = 'exp'
            self.model = TaGSim(self.args, self.number_of_labels).to(self.device)
        else:
            assert False

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        if self.args.model_name in ["GPN", "SimGNN"]:
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target = data["target"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(target, prediction)
                # self.values.append((target - prediction).item())
        elif self.args.model_name == "GedGNN":
            weight = self.args.loss_weight
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target, gt_mapping = data["target"], data["mapping"]
                prediction, _, mapping = self.model(data)
                losses = losses + fixed_mapping_loss(mapping, gt_mapping) + weight * F.mse_loss(target, prediction)
                if self.args.finetune:
                    if self.args.target_mode == "linear":
                        losses = losses + F.relu(target - prediction)
                    else: # "exp"
                        losses = losses + F.relu(prediction - target)
        elif self.args.model_name == "TaGSim":
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                ta_ged = data["ta_ged"]
                prediction, _ = self.model(data)
                losses = losses + torch.nn.functional.mse_loss(ta_ged, prediction)
        else:
            assert False

        losses.backward()
        self.optimizer.step()
        return losses.item()

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
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        # print(self.global_labels)

        ged_dict = dict()
        # We could load ged info from several files.
        # load_ged(ged_dict, self.args.abs_path, dataset_name, 'xxx.json')
        load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
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
        # self.A = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long().to(self.device)
            self.edge_index.append(edge)
            # A = torch.sparse_coo_tensor(edge, torch.ones(edge.shape[1]), (g['n'], g['n'])).to_dense().to(self.device)
            # self.A.append(A)

        self.features = [torch.tensor(x).float().to(self.device) for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float, device=self.device)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float().to(self.device)
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1

    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense().to(device)

        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]  # the last del_num edges in edge are removed
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]  # add reverse edges
        new_edge = new_edge + [[x, x] for x in range(n)]  # add self-loops

        new_edge = torch.tensor(new_edge).t().long().to(device)

        feature2 = torch.zeros(f.shape).to(device)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data

    def gen_delta_graphs(self):
        k = self.args.num_delta_graphs
        n = len(self.graphs)
        for i, g in enumerate(self.graphs):
            # Do not generate delta graphs for small graphs.
            if g['n'] <= 10:
                continue
            # gen k delta graphs
            f = self.features[i]
            self.delta_graphs[i] = [delta_graph(g, f, self.device) for j in range(k)]

    def check_pair(self, i, j):
        if i == j:
            return (0, i, j)
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return (0, i, j)
        elif (id2, id1) in self.ged_dict:
            return (0, j, i)
        else:
            return None

    def init_graph_pairs(self):
        random.seed(1)

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

        assert self.args.graph_pair_mode == "combine"
        dg = self.delta_graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        self.training_graphs.append(tmp)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    self.training_graphs.append((1, i, j))

        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)

        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.val_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.val_graphs.append((1, i, list(range(k))))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing_graphs.append((1, i, list(range(k))))

        li = []
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                li.append(i)

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing2_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing2_graphs.append((1, i, list(range(k))))

        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} testing2 graph pairs.".format(len(self.testing2_graphs), self.args.num_testing_graphs))

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

    def pack_graph_pair(self, graph_pair):
        """
        Prepare the graph pair data for GedGNN model.
        :param graph_pair: (pair_type, id_1, id_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (pair_type, id_1, id_2) = graph_pair
        if pair_type == 0:  # normal case
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)

            real_ged = self.ged[id_1][id_2][0]
            ta_ged = self.ged[id_1][id_2][1:]

            new_data["id_1"] = id_1
            new_data["id_2"] = id_2

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = self.edge_index[id_2]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = self.features[id_2]

            if self.args.gtmap:
                new_data["mapping"] = self.mapping[id_1][id_2]
        elif pair_type == 1:  # delta graphs
            new_data["id"] = id_1
            dg: dict = self.delta_graphs[id_1][id_2]

            real_ged = dg["ta_ged"][0]
            ta_ged = dg["ta_ged"][1:]

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = dg["edge_index"]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = dg["features"]

            if self.args.gtmap:
                new_data["mapping"] = dg["mapping"]
        else:
            assert False

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2]) if pair_type == 0 else (dg["n"], dg["m"])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = real_ged
        # new_data["ta_ged"] = ta_ged
        if self.args.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            new_data["avg_v"] = avg_v
            new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
            new_data["ta_ged"] = torch.exp(torch.tensor(ta_ged).float() / -avg_v).to(self.device)
        elif self.args.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            new_data["hb"] = higher_bound
            new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)
            new_data["ta_ged"] = (torch.tensor(ta_ged).float() / higher_bound).to(self.device)
        else:
            assert False

        return new_data

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
                loss_sum = 0
                main_index = 0
                for index, batch in enumerate(batches):
                    batch_total_loss = self.process_batch(batch)  # without average
                    loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = loss_sum / main_index  # the average loss of current epoch
                    pbar.update(len(batch))
                    pbar.set_description(
                        "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),
                                                                       index,
                                                                       round(1000 * batch_total_loss / len(batch), 3)))
                tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
                training_loss = round(1000 * loss, 3)
        t2 = time.time()
        training_time = t2 - t1
        if len(self.values) > 0:
            self.prediction_analysis(self.values, "training_score")

        self.results.append(
            ('model_name', 'dataset', 'graph_set', "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            print("## Training", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

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

    def score(self, testing_graph_set='test', test_k=0):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()
        # self.model.train()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                model_out = self.model(data) if test_k == 0 else self.test_matching(data, test_k)
                prediction, pre_ged = model_out[0], model_out[1]
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:  # TaGSim
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

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

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mse', 'mae', 'acc',
                             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            if test_k == 0:
                print("## Testing", file=f)
            else:
                print("## Post-processing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def batch_score(self, testing_graph_set='test', test_k=100):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()
        # self.model.train()

        batch_results = []
        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            res = []
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                gt_ged = data["ged"]
                time_list, pre_ged_list = self.test_matching(data, test_k, batch_mode=True)
                res.append((gt_ged, pre_ged_list, time_list))
            batch_results.append(res)

        batch_num = len(batch_results[0][0][1]) # len(pre_ged_list)
        for i in range(batch_num):
            time_usage = []
            num = 0  # total testing number
            mse = []  # score mse
            mae = []  # ged mae
            num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
            num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
            rho = []
            tau = []
            pk10 = []
            pk20 = []

            for res in batch_results:
                pre = []
                gt = []
                for gt_ged, pre_ged_list, time_list in res:
                    time_usage.append(time_list[i])
                    pre_ged = pre_ged_list[i]
                    round_pre_ged = round(pre_ged)

                    num += 1
                    mse.append(-0.001)
                    pre.append(pre_ged)
                    gt.append(gt_ged)

                    mae.append(abs(round_pre_ged - gt_ged))
                    if round_pre_ged == gt_ged:
                        num_acc += 1
                        num_fea += 1
                    elif round_pre_ged > gt_ged:
                        num_fea += 1
                rho.append(spearmanr(pre, gt)[0])
                tau.append(kendalltau(pre, gt)[0])
                pk10.append(self.cal_pk(10, pre, gt))
                pk20.append(self.cal_pk(20, pre, gt))

            time_usage = round(np.mean(time_usage), 3)
            mse = round(np.mean(mse) * 1000, 3)
            mae = round(np.mean(mae), 3)
            acc = round(num_acc / num, 3)
            fea = round(num_fea / num, 3)
            rho = round(np.mean(rho), 3)
            tau = round(np.mean(tau), 3)
            pk10 = round(np.mean(pk10), 3)
            pk20 = round(np.mean(pk20), 3)
            self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                                 fea, rho, tau, pk10, pk20))

            print(*self.results[-1], sep='\t')
            with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
                print(*self.results[-1], sep='\t', file=f)

    def print_results(self):
        for r in self.results:
            print(*r, sep='\t')

        with open(self.args.abs_path + self.args.result_path + 'results.txt', 'a') as f:
            for r in self.results:
                print(*r, sep='\t', file=f)

    def test_matching(self, data, test_k, batch_mode=False):
        prediction, pre_ged, soft_matrix = self.model(data)
        m = torch.nn.Softmax(dim=1)
        soft_matrix = (m(soft_matrix) * 1e9 + 1).round()
        n1, n2 = soft_matrix.shape
        # print(data["edge_index_1"].shape)
        g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=n1)
        g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=n2)
        g1.ndata['f'] = data["features_1"]
        g2.ndata['f'] = data["features_2"]

        if batch_mode:
            t1 = time.time()
            solver = KBestMSolver(soft_matrix, g1, g2)
            res = []
            time_usage = []
            for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                if i > test_k:
                    break
                solver.get_matching(i)
                min_res = solver.min_ged
                t2 = time.time()
                time_usage.append(t2 - t1)
                res.append(min_res)
                time_usage.append(t2 - t1)
                res.append(min(pre_ged, min_res))
            return time_usage, res
        else:
            solver = KBestMSolver(soft_matrix, g1, g2)
            solver.get_matching(test_k)
            min_res = solver.min_ged
            return None, min_res

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
            data = self.pack_graph_pair(graph_pair)
            avg_v = data["avg_v"]  # (n1+n2)/2.0, a scalar, not a tensor
            gt_ged, target = data["ged"], data["target"]  # gt ged value and score
            soft_matrix, _, prediction = self.model(data, is_testing=True)
            pre_ged, gt_ged, gt_score = prediction.item(), gt_ged.item(), target.item()

            values0.append(pre_ged - gt_ged)

            soft_matrix = (torch.sigmoid(soft_matrix) * 1e9 + 1).round()
            # soft_matrix = (m(soft_matrix) * 1e9 + 1).int()
            # soft_matrix = ((soft_matrix - soft_matrix.min()) * 1e9 + 1).round()

            n1, n2 = soft_matrix.shape
            # print(data["edge_index_1"].shape)
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
            print("1000:", len(score_1000[0]), round(np.mean(score_1000[1]), 3), round(np.mean(score_1000[2]), 3),
                  sep='\t')
            print("{} / {} = {}".format(num_1000, total_num, round(num_1000 / total_num, 3)))

        with open(self.args.abs_path + self.args.result_path + self.args.dataset + '.txt', 'a') as f:
            print('', file=f)
            print(self.cur_epoch, testing_graph_set, demo_num, sep='\t', file=f)
            if test_k >= 10:
                print("10", round(np.mean(score_10[0]) * 1000, 3), round(np.mean(score_10[1]), 3),
                      round(np.mean(score_10[2]), 3), round(num_10 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_10, total_num, round(num_10 / total_num, 3)), file=f)
            if test_k >= 100:
                print("100", round(np.mean(score_100[0]) * 1000, 3), round(np.mean(score_100[1]), 3),
                      round(np.mean(score_100[2]), 3), round(num_100 / total_num, 3), sep='\t', file=f)
                # print("{} / {} = {}".format(num_100, total_num, round(num_100 / total_num, 3)), file=f)
            if test_k >= 1000:
                print("1000", round(np.mean(score_1000[0]) * 1000, 3), round(np.mean(score_1000[1]), 3),
                      round(np.mean(score_1000[2]), 3), round(num_1000 / total_num, 3), sep='\t', file=f)
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
        epoch = str(self.cur_epoch + 1)
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

    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch))

    def load(self, epoch):
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path + self.args.dataset + '_' + str(epoch)))
