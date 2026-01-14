r"""
Synthetic dataset is from the article 'Federated Optimization in Heterogeneous Networks' (http://arxiv.org/abs/1812.06127) of Li et. al.
The para of this benchmark is
Args:::
    alpha (float): Synthetic(alpha, beta), default value is 0
    beta (float): Synthetic(alpha, beta), default value is 0
    num_clients (int): the number of all the clients, default value is 30
    imbalance (float): the degree of data-size-imbalance ($imbalance \in [0,1]$), default value is 0.0
    mean_datavol (int): the mean data size of all the clients, default value is 400
    dimension (int): the dimension of the feature, default value is 60
    num_classes (int): the number of classes in the label set, default value is 10

Example::
```python
    >>> import flgo
    >>> import flgo.benchmark.synthetic_regression as synthetic
    >>> config = {'benchmark':{'name':synthetic, 'para':{'alpha':1., 'beta':1., 'num_clients':30}}}
    >>> flgo.gen_task(config, './my_synthetic11')
```
"""
from flgo.benchmark.synthetic_regression.model import lr

default_model = lr

def visualize(generator, partitioner, task_path:str):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.cm as cm
    local_models = generator.optimal_local
    local_models = np.array(local_models)
    local_models = local_models.reshape(local_models.shape[0], -1)
    local_datas = generator.local_datas
    n_clients = len(local_datas)
    local_data_sizes = np.array([len(d['y']) for d in local_datas])
    local_data_sizes = local_data_sizes/local_data_sizes.max()
    cmap = cm.Blues
    colors = cmap(local_data_sizes)
    perplexity = max(n_clients/5., 1.) if n_clients<=30 else 30.
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    data_2d = tsne.fit_transform(local_models)
    plt.figure()
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, edgecolors='k', s=50)
    plt.title(f't-SNE of (w*|b*) for Syn({generator.alpha},{generator.beta}, Imb.={generator.imbalance})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig(os.path.join(task_path, 'res.png'))
    plt.show()