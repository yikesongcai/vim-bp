import torch
import torch.utils.data as tud
import flgo.decorator as fd
class GaussianNoiser(fd.BasicDecorator):
    """
    This noiser is a non-official implementation of Gaussian noise in 'Federated Learning on Non-IID Data Silos: An Experimental Study.'.
    The official implementation is at https://github.com/Xtra-Computing/NIID-Bench.

    Args:
        sigma (float): Standard deviation of the Gaussian noise.
        dynamic (bool): whether to use dynamic noise or not.
    """
    class NoiseDataset(tud.Dataset):
        def __init__(self, data, sigma, dynamic=True):
            self.data = data
            self.sigma = sigma
            self.dynamic = dynamic
            if not self.dynamic:
                x = data[0][0]
                self.noise = [torch.randn_like(x)*self.sigma for _ in range(len(data))]
            else:
                self.noise = None

        def __getitem__(self, index):
            x,y = self.data[index]
            if self.dynamic:
                new_x = x + torch.randn_like(x)*self.sigma
            else:
                new_x = x + self.noise[index]
            return new_x,y

        def __len__(self):
            return len(self.data)

    def __init__(self, sigma:float, dynamic=True):
        self.sigma = sigma
        self.dynamic = dynamic

    def __call__(self, runner, *args, **kwargs):
        sigmas = [self.sigma*(i+1.)/len(runner.clients) for i in range(len(runner.clients))]
        for c,sc in zip(runner.clients, sigmas):
            c.set_data(self.NoiseDataset(c.train_data, sc, self.dynamic), 'train')
            if c.val_data is not None and len(c.val_data) > 0:
                c.set_data(self.NoiseDataset(c.val_data, sc, self.dynamic), 'val')
            if c.test_data is not None and len(c.test_data) > 0:
                c.set_data(self.NoiseDataset(c.test_data, sc, self.dynamic), 'test')
        self.register_runner(runner)

    def __str__(self):
        return f"Gaussian-0-{self.sigma}*iN"

class HardGaussianNoiser(fd.BasicDecorator):
    """
    This noiser adds noise to features with different means and standard deviations across clients.

    Args:
        std_mu (float): Standard deviation of the clients' local means of Gaussian noise.
        std_sigma (float): Standard deviation of the Gaussian noise.
        dynamic (bool): whether to use dynamic noise or not.
    """
    class NoiseDataset(tud.Dataset):
        def __init__(self, data, mu, sigma, dynamic=True):
            self.data = data
            self.sigma = sigma
            self.mu = mu
            self.dynamic = dynamic
            if not self.dynamic:
                x = data[0][0]
                self.noise = [mu+torch.randn_like(x)*self.sigma for _ in range(len(data))]
            else:
                self.noise = None

        def __getitem__(self, index):
            x, y = self.data[index]
            if self.dynamic:
                new_x = x + self.mu + torch.randn_like(x) * self.sigma
            else:
                new_x = x + self.noise[index]
            return new_x, y

        def __len__(self):
            return len(self.data)

    def __init__(self, std_mu:float, std_sigma: float, dynamic=True):
        self.std_mu = std_mu
        self.std_sigma = std_sigma
        self.dynamic = dynamic

    def __call__(self, runner, *args, **kwargs):
        example = runner.clients[0].train_data[0][0]
        for c in runner.clients:
            cmu = torch.randn_like(example)*self.std_mu
            csigma = torch.randn(1)*self.std_sigma
            c.set_data(self.NoiseDataset(c.train_data, cmu, csigma, self.dynamic), 'train')
            if c.val_data is not None and len(c.val_data) > 0:
                c.set_data(self.NoiseDataset(c.val_data, cmu, csigma, self.dynamic), 'val')
            if c.test_data is not None and len(c.test_data) > 0:
                c.set_data(self.NoiseDataset(c.test_data, cmu, csigma, self.dynamic), 'test')
        self.register_runner(runner)

    def __str__(self):
        return f"Gaussian_Nmu-0-{self.std_mu}_Nsigma-0-{self.std_sigma}"
