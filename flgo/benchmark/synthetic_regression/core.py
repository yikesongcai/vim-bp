from flgo.benchmark.base import *
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator
from flgo.benchmark.toolkits.partition import BasicPartitioner
import numpy as np
TaskCalculator = GeneralCalculator

class TaskPipe(XYHorizontalTaskPipe):
    def load_data(self, running_time_option) -> dict:
        test_data = self.feddata['server']['data']
        test_data = self.TaskDataset(torch.tensor(test_data['x']), torch.tensor(test_data['y']))
        local_datas = [self.TaskDataset(torch.tensor(self.feddata[cname]['data']['x']),
                                        torch.tensor(self.feddata[cname]['data']['y'])) for cname in
                       self.feddata['client_names']]
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        for key in self.feddata['server'].keys():
            if key == 'data':
                continue
            task_data['server'][key] = self.feddata['server'][key]
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = local_datas[cid]
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['local_test'] and cdata_val is not None:
                cdata_val, cdata_test = self.split_dataset(cdata_val, running_time_option['local_test_ratio'])
            else:
                cdata_test = None
            task_data[cname] = {'train': cdata_train, 'val': cdata_val, 'test': cdata_test, 'op':self.feddata[cname]['data']['w']}
            for key in self.feddata[cname]:
                if key == 'data':
                    continue
                task_data[cname][key] = self.feddata[cname][key]
        return task_data

class TaskGenerator(BasicTaskGenerator):
    def __init__(self, alpha=0.0, beta=0.0, num_clients=30, imbalance=0.0, mean_datavol=400, dimension=60, num_classes=10, *args, **kwargs):
        super().__init__('synthetic_regression', '')
        self.num_clients = num_clients
        self.alpha = alpha
        self.beta = beta
        self.imbalance = imbalance
        self.mean_datavol = mean_datavol
        self.dimension = dimension
        self.num_classes = num_classes

    def load_data(self):
        W_global = np.random.normal(0, 1, (self.dimension, self.num_classes))
        b_global = np.random.normal(0, 1, self.num_classes)
        v_global = np.zeros(self.dimension)
        # create Sigma = Diag([i^(-1.2) for i in range(60)])
        diagonal = np.zeros(self.dimension)
        for j in range(self.dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        Sigma = np.diag(diagonal)
        # V
        V = np.zeros((self.num_clients, self.dimension))
        samples_per_client = BasicPartitioner().data_imbalance_generator(self.num_clients, int(self.mean_datavol * self.num_clients),self.imbalance)
        # Concept Skew
        if self.alpha>=0:
            Us = np.random.normal(0, self.alpha, self.num_clients)
            W = [np.random.normal(Us[k], 1, (self.dimension, self.num_classes)) for k in range(self.num_clients)]
            b = [np.random.normal(Us[k], 1, self.num_classes) for k in range(self.num_clients)]
        else:
            W = [W_global for _ in range(self.num_clients)]
            b = [b_global for _ in range(self.num_clients)]
        # Feature Skew
        if self.beta>=0:
            B = np.random.normal(0, self.beta, self.num_clients)
            for cid in range(self.num_clients): V[cid] = np.random.normal(B[cid], 1, self.dimension)
        else:
            for cid in range(self.num_clients): V[cid] = v_global
        X_split = [[] for _ in range(self.num_clients)]
        y_split = [[] for _ in range(self.num_clients)]
        optimal_local = [np.concatenate((wk, bk.reshape(1, bk.shape[0])), axis=0).tolist() for wk, bk in zip(W, b)]
        test_data = {'x': [], 'y':[]}
        local_datas = []
        for cid in range(self.num_clients):
            # X_ki~N(v_k, Sigma)
            X_k = np.random.multivariate_normal(V[cid], Sigma, samples_per_client[cid])
            Y_k = np.zeros(samples_per_client[cid], dtype=int)
            for i in range(samples_per_client[cid]):
                # Y_ki = argmax(softmax(W_k x_ki + b_k))
                tmp = np.dot(X_k[i], W[cid]) + b[cid]
                Y_k[i] = np.argmax(np.exp(tmp)/np.sum(np.exp(tmp)))
            X_split[cid] = X_k.tolist()
            y_split[cid] = Y_k.tolist()
            # split dataset to train and testing part
            k = int(samples_per_client[cid]*0.85)
            x_train = X_split[cid][:k]
            y_train = y_split[cid][:k]
            x_test = X_split[cid][k:]
            y_test = y_split[cid][k:]
            test_data['x'].extend(x_test)
            test_data['y'].extend(y_test)
            local_datas.append({'x': x_train, 'y':y_train, 'w':np.concatenate((W[cid], np.expand_dims(b[cid], 0)), axis=0).tolist()})
        self.test_data = test_data
        self.local_datas = local_datas
        self.optimal_local = optimal_local
        return X_split, y_split, optimal_local

    def get_task_name(self):
        partitioner_name = 'iid' if self.alpha<0 and self.beta<0 else 'a{:.1f}b{:.1f}'.format(self.alpha, self.beta)
        if self.imbalance>0: partitioner_name += '_imb{:.1f}'.format(self.imbalance)
        return '_'.join(['B-'+self.benchmark, 'P-'+partitioner_name, 'N-'+str(self.num_clients)])
