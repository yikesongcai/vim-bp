import numpy as np
import torch.utils.data as tud
import flgo.decorator as fd

class LabelFlipper(fd.BasicDecorator):
    """
    Flip labels in datasets.

    Args:
        ratio_flip (float): the ratio of clients that will flip their labels
        label_map (dict): the label flipping map, e.g. {1:0} refers to flipping label 1 to label 0
    """
    class YFlipDataset(tud.Dataset):
        def __init__(self, data, lb_map):
            self.data = data
            self.lb_map = lb_map

        def __getitem__(self, index):
            x,y = self.data[index]
            new_y = self.lb_map[y] if y in self.lb_map else y
            return x,new_y

        def __len__(self):
            return len(self.data)

    def __init__(self, ratio_flip:float, label_map:dict={}):
        self.ratio_flip = ratio_flip
        self.label_map = label_map
        self.flipped_clients = []

    def __call__(self, runner, *args, **kwargs):
        num_flip = int(len(runner.clients) * self.ratio_flip)
        if self.ratio_flip > 0.: num_flip = max(1, num_flip)
        self.flipped_clients = np.random.choice(list(range(len(runner.clients))), num_flip, replace=False).tolist()
        for cid in self.flipped_clients:
            c = runner.clients[cid]
            flipped_train_data = self.flip_data(c.train_data)
            c.set_data(flipped_train_data, 'train')
        self.register_runner(runner)

    def flip_data(self, train_data):
        return self.YFlipDataset(train_data, self.label_map)

    def __str__(self):
        return f"LabelFlipper{self.ratio_flip}_F{len(self.label_map)}"



