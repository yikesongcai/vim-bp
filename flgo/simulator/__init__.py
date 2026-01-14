r"""
This module is to simulate arbitrary system heterogeneity that may occur in practice.
We conclude four types of system heterogeneity from existing works.
System Heterogeneity Description:
    1. **Availability**: the devices will be either available or unavailable at each moment, where only the
                    available devices can be selected to participate in training.

    2. **Responsiveness**: the responsiveness describes the length of the period from the server broadcasting the
                    gloabl model to the server receiving the locally trained model from a particular client.

    3. **Completeness**: since the server cannot fully control the behavior of devices,it's possible for devices to
                    upload imcomplete model updates (i.e. only training for a few steps).

    4. **Connectivity**: the clients who promise to complete training may suffer accidients so that the server may lose
                    connections with these client who will never return the currently trained local model.

We build up a client state machine to simulate the four types of system heterogeneity, and provide high-level
APIs to allow customized system heterogeneity simulation.

**Example**: How to customize the system heterogeneity:
```python
>>> class MySimulator(flgo.simulator.base.BasicSimulator):
...     def update_client_availability(self):
...         # update the variable 'prob_available' and 'prob_unavailable' for all the clients
...         self.set_variable(self.all_clients, 'prob_available', [0.9 for _ in self.all_clients])
...         self.set_variable(self.all_clients, 'prob_unavailable', [0.1 for _ in self.all_clients])
...
...     def update_client_connectivity(self, client_ids):
...         # update the variable 'prob_drop' for clients in client_ids
...         self.set_variable(client_ids, 'prob_drop', [0.1 for _ in client_ids])
...
...     def update_client_responsiveness(self, client_ids, *args, **kwargs):
...         # update the variable 'latency' for clients in client_ids
...         self.set_variable(client_ids, 'latency', [np.random.randint(5,100) for _ in client_ids])
...
...     def update_client_completeness(self, client_ids, *args, **kwargs):
...         # update the variable 'working_amount' for clients in client_ids
...         self.set_variable(client_ids, 'working_amount',  [max(int(self.clients[cid].num_steps*np.random.rand()), 1) for cid in client_ids])
>>> r = flgo.init(task, algorithm=fedavg, Simulator=MySimulator)
>>> # The runner r will be runned under the customized system heterogeneity, where the clients' states will be flushed by
>>> # MySimulator.update_client_xxx at each moment of the virtual clock or particular events happen (i.e. a client was selected)
```

We also provide some preset Simulator like flgo.simulator.DefaultSimulator and flgo.simulator.
"""
import os

from flgo.simulator.default_simulator import Simulator as DefaultSimulator
from flgo.simulator.phone_simulator import Simulator as PhoneSimulator
from flgo.simulator.base import BasicSimulator
import numpy as np
import random
import matplotlib.pyplot as plt

class ResponsivenessExampleSimulator(BasicSimulator):
    def initialize(self):
        self.client_time_response = {cid: np.random.randint(5, 1000) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(self.client_time_response.values()))

    def update_client_responsiveness(self, client_ids):
        latency = [self.client_time_response[cid] for cid in client_ids]
        self.set_variable(client_ids, 'latency', latency)

class CompletenessExampleSimulator(BasicSimulator):
    def update_client_completeness(self, client_ids):
        if not hasattr(self, '_my_working_amount'):
            rs = self.random_module.normal(1.0, 1.0, len(self.clients))
            rs = rs.clip(0.01, 2)
            self._my_working_amount = {cid:max(int(r*self.clients[cid].num_steps),1) for  cid,r in zip(self.clients, rs)}
        working_amount = [self._my_working_amount[cid] for cid in client_ids]
        self.set_variable(client_ids, 'working_amount', working_amount)

class AvailabilityExampleSimulator(BasicSimulator):
    def update_client_availability(self):
        if self.gv.clock.current_time==0:
            self.set_variable(self.all_clients, 'prob_available', [1.0 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [0.0 for _ in self.clients])
            return
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

class ConnectivityExampleSimulator(BasicSimulator):
    def initialize(self):
        drop_probs = self.random_module.uniform(0.,0.05, len(self.clients)).tolist()
        self.client_drop_prob = {cid: dp for cid,dp in zip(self.clients, drop_probs)}

    def update_client_connectivity(self, client_ids):
        self.set_variable(client_ids, 'prob_drop', [self.client_drop_prob[cid] for cid in client_ids])

class ExampleSimulator(BasicSimulator):
    def initialize(self):
        drop_probs = self.random_module.uniform(0.,0.05, len(self.clients)).tolist()
        self.client_drop_prob = {cid: dp for cid,dp in zip(self.clients, drop_probs)}
        self.client_time_response = {cid: np.random.randint(5, 1000) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(self.client_time_response.values()))
        self.server.tolerance_for_latency = 999999

    def update_client_connectivity(self, client_ids):
        self.set_variable(client_ids, 'prob_drop', [self.client_drop_prob[cid] for cid in client_ids])

    def update_client_responsiveness(self, client_ids):
        latency = [self.client_time_response[cid] for cid in client_ids]
        self.set_variable(client_ids, 'latency', latency)

    def update_client_availability(self):
        if self.gv.clock.current_time==0:
            self.set_variable(self.all_clients, 'prob_available', [1.0 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [0.0 for _ in self.clients])
            return
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

    def update_client_completeness(self, client_ids):
        if not hasattr(self, '_my_working_amount'):
            rs = self.random_module.normal(1.0, 1.0, len(self.clients))
            rs = rs.clip(0.01, 2)
            self._my_working_amount = {cid:max(int(r*self.clients[cid].num_steps),1) for  cid,r in zip(self.clients, rs)}
        working_amount = [self._my_working_amount[cid] for cid in client_ids]
        self.set_variable(client_ids, 'working_amount', working_amount)

def visualize_availability(data, sort=True, title='', show=True, save=False):
    """
    Visualize availability matrix

    Args:
        data (numpy.ndarray): a 2d-array where each row corresponds to round-wise availability of all the clients and each column refers to the availability of one client across rounds.
    """
    if not isinstance(data, np.ndarray):data = np.array(data)
    data = data.T
    if sort:
        row_sums = np.sum(data, axis=1)
        sorted_indices = np.argsort(row_sums)
        data = data[sorted_indices]
    fig, ax1 = plt.subplots()
    ax1.imshow(data, cmap='Greens')
    bars = np.sum(data, axis=1)
    ax1.barh(list(range(len(bars))), bars, alpha=0.25)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Client ID')
    data2 = np.sum(data, axis=0)
    ax1.plot(list(range(len(data2))), data2, 'r', label='num')
    tit = 'Client Availability'
    if title!='': tit = tit + '-' + title
    ax1.invert_yaxis()
    plt.title(tit)
    plt.grid(False)
    plt.tight_layout()
    if save: plt.savefig(f'{tit}.png', dpi=300, bbox_inches='tight')
    if show: plt.show()
    return tit

def visualize_capacity(data, sort=True, title='', show=True, save=False):
    if not isinstance(data, np.ndarray):data = np.array(data)
    data = data.T
    if sort:
        row_sums = np.sum(data, axis=1)
        sorted_indices = np.argsort(row_sums)
        data = data[sorted_indices]
    mean_cap = np.mean(data, axis=1)
    min_cap = np.min(data, axis=1)
    max_cap = np.max(data, axis=1)
    plt.figure()
    plt.bar(list(range(len(mean_cap))), mean_cap, color='gray')
    plt.plot(list(range(len(mean_cap))), mean_cap, linewidth=2, color='black')
    plt.scatter(list(range(len(min_cap))), min_cap, s=6, marker='o', color='green')
    plt.scatter(list(range(len(min_cap))), max_cap, s=6, marker='o', color='red')
    plt.xlabel("Client ID")
    plt.ylabel("Capacity Ratio")
    tit = "Device Capacity"
    if title!='':tit=tit + '-' + title
    plt.title(tit)
    plt.tight_layout()
    if save: plt.savefig(f'{tit}.png', dpi=300, bbox_inches='tight')
    if show: plt.show()
    return tit

def visualize_latency(data, sort=True, title='', show=True, save=False):
    if not isinstance(data, np.ndarray):data = np.array(data)
    if sort:
        h = data.T
        rows = np.mean(h, axis=1)
        sorted_indices = np.argsort(rows)
        h = h[sorted_indices]
        data = h.T
    means = np.mean(data, axis=0)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.barh(np.arange(len(means)), means, color='red', alpha=0.2)
    ax2.set_ylabel('Clients', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # ax2.scatter(full_max, len(means), color='red', s=100, marker='*', label='Full Par.')
    rounds = data.shape[0]
    colors = ['red', 'pink', 'yellow', 'green', 'purple']
    ps = [1.0, 0.5,0.2, 0.1, 0.01, ]
    all_d = []
    all_n = []
    for p,c in zip(ps, colors):
        ds = []
        for k in range(20):
            d = np.array([np.random.choice(data[i], size=max(int(data.shape[1] * p), 1), replace=False).tolist() for i in range(rounds)]).max(axis=1).mean()
            ds.append(d)
        d = np.array(ds).mean()
        all_d.append(d)
        all_n.append(len(means)*p)
        # ax2.scatter(d, len(means)*p, color=c, s=100, marker='*', label=f'{int(p*100)}% Par.')
    ax2.plot(all_d, all_n, '--', color='black')
    for d,n,c,p in zip(all_d, all_n, colors,ps):
        ax2.scatter(d, n, color=c, s=100, marker='*', label=f'{int(p*100)}% Par.')
    hist_values, bins, patches = ax1.hist(means, bins='auto',edgecolor='none', color='skyblue')
    ax1.set_xlabel('Latency (virtual time unit)')
    ax1.set_ylabel('Num of Client', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])  # 计算每个bin的中心
    if bin_centers.size>2:
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)  # 更高分辨率的 x 数据
        spline = make_interp_spline(bin_centers, hist_values, k=2)  # 使用三次样条插值
        y_smooth = spline(x_smooth)
        ax1.plot(x_smooth, y_smooth, color='blue', linestyle='-', linewidth=2)
        ax1.scatter(bin_centers, hist_values, color='blue', marker='o')
    else:
        ax1.scatter(bin_centers, hist_values, color='blue', marker='o')
    tit = "Responsiveness"
    if title!='':tit=tit + '-' + title
    plt.title(tit)
    plt.legend(loc='upper left')
    plt.tight_layout()
    if save: plt.savefig(f'{tit}.png', dpi=300, bbox_inches='tight')
    if show: plt.show()
    return tit

def visualize_completeness(data, sort=True, title="", show=True, save=False):
    if not isinstance(data, np.ndarray):data = np.array(data).T
    if sort:
        rows = np.mean(data, axis=1)
        sorted_indices = np.argsort(rows)
        data = data[sorted_indices]
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap='Reds', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Completeness Degree',  fraction=0.046)
    tit = "Training Completeness"
    if title!='':tit=tit + '-' + title
    plt.title(tit)
    plt.gca().invert_yaxis()
    plt.ylabel('Client ID')
    plt.xlabel('Rounds')
    plt.tight_layout()
    # plt.axis('equal')
    # plt.ylim([0, data.shape[0]])
    ax.set_aspect('auto', adjustable='datalim')
    ax.set_ylim([0, data.shape[0]])
    if save: plt.savefig(f'{tit}.png', dpi=300, bbox_inches='tight')
    if show: plt.show()
    return tit

def visualize_stability(data, sort=True, title="", show=True, save=False):
    if not isinstance(data, np.ndarray): data = np.array(data).T
    if sort:
        rows = np.sum(data[:,0,:], axis=1)
        sorted_indices = np.argsort(rows)
        data = data[sorted_indices]
    dropped = data[:,0,:]
    probs = data[:,1,:]
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('P(Dropping)', color='blue')
    ax1.set_xlabel('Client ID')
    ax1.boxplot(probs.T)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Actual Dropping Times', color='orange')
    dropped = dropped.sum(axis=1)
    ax2.bar(list(range(len(dropped))), dropped, color='orange', alpha=0.2)
    tit = "Connection Stability"
    if title!='':tit=tit + '-' + title
    plt.title(tit)
    plt.xticks([])
    plt.tight_layout()
    if save: plt.savefig(f'{tit}.png', dpi=300, bbox_inches='tight')
    if show: plt.show()
    return tit

def visualize_simulator(runner, sort=False, together=True, save=True, select=[], title=""):
    """
    Visualize a simulator.

    Args:
        runner (Any): the runner generated by flgo.init
        sort (bool): reasonably sorting clients in each system heterogeneity figure
        together (bool): plot all the figures together in one figure
        save (bool): save the figure to disk
        select (list): select which system heterogeneity to be visualized. Default is selecting all. [0,1,2,3,4] corresponds to [availability, responsiveness, completeness, connectivity,capacity] respectively.
    """
    runner.proportion = 1.0
    runner.sample_option = 'full'
    runner.eval_interval = -1
    completeness = []
    latencies = []
    avails = []
    drops = []
    caps = []
    if select==[]: select=list(range(5))
    def pack(self, client_id, mtype=0):
        return {}

    def iterate(self):
        caps.append([c._capacity for c in self.clients])
        avails.append([int(c.is_idle()) for c in self.clients])
        self.selected_clients = list(range(self.num_clients))
        res = self.communicate(self.selected_clients)
        dropped = [1. if cid in self._dropped_selected_clients else 0. for cid in self.selected_clients]
        pdrop = self.gv.simulator.get_variable(self.gv.simulator.idx2id(self.selected_clients), 'prob_drop')
        drops.append([dropped, pdrop])
        completeness.append([1.0*c._working_amount/c.num_steps for c in self.clients])
        latencies.append([c._latency for c in self.clients])
        return

    def reply(self, svr_pkg):
        return {}

    runner.__class__.pack = pack
    runner.__class__.iterate = iterate
    for c in runner.clients:
        c.__class__.reply = reply
        c.actions = {0: c.reply}
    runner.run()
    n = f"{runner.gv.simulator.__class__.__name__}-R{runner.num_rounds}"
    if together:
        ts = []
        if 0 in select: ts.append(visualize_availability(avails, sort, show=False, save=True))
        if 1 in select: ts.append(visualize_latency(latencies, sort, show=False, save=True))
        if 2 in select: ts.append(visualize_completeness(completeness, sort, show=False, save=True))
        if 3 in select: ts.append(visualize_stability(drops, sort, show=False, save=True))
        if 4 in select: ts.append(visualize_capacity(caps, sort, show=False, save=True))
        plt.close('all')
        image_files = [f"{t}.png" for t in ts]
        import matplotlib.image as mpimg
        fig, axs = plt.subplots(1, len(select), figsize=(4*len(select), 4))
        for ax, image_file in zip(axs, image_files):
            img = mpimg.imread(image_file)  # 读取图像
            ax.imshow(img)  # 显示图像
            ax.axis('off')  # 关闭坐标轴
        plt.suptitle(title if title!='' else n)
        plt.tight_layout()
        if save: plt.savefig(f'{n}.png', dpi=300, bbox_inches='tight')
        plt.show()
        for img in image_files: os.remove(img)
    else:
        n = title if title!='' else n
        if 0 in select: visualize_availability(avails, sort, n, save=save)
        if 1 in select: visualize_latency(latencies, sort, n, save=save)
        if 2 in select: visualize_completeness(completeness, sort, n, save=save)
        if 3 in select: visualize_stability(drops, sort, n, save=save)
        if 4 in select: visualize_capacity(caps, sort, n, save=save)







