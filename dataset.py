from torch.utils.data import Dataset
import numpy as np
import torch


def random_cluster(xc, yc, n_points):
    return np.array([[xc + np.random.randn(), yc + np.random.randn()] for _ in range(n_points)])


class ToyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.n_points = 1000
        self.in_cluster_centers = [[0, 0], [10, 0], [0, 10]]
        self.data = []
        self.labels = []
        for k, (xc, yc) in enumerate(self.in_cluster_centers):
            cluster = self.generate_cluster(xc, yc, self.n_points)
            self.data.extend(cluster / 11)
            self.labels.extend([k for _ in range(self.n_points)])

        self.data_min = np.array(self.data).min(0)
        self.data_max = np.array(self.data).max(0)

        self.data_out = []
        self.labels_out = []
        self.out_cluster_center = [10, 10]
        cluster = self.generate_cluster(self.out_cluster_center[0],
                                        self.out_cluster_center[1],
                                        self.n_points)
        self.data_out.extend(cluster / 11)
        self.labels_out.extend([3 for _ in range(self.n_points)])

    @staticmethod
    def generate_cluster(xc, yc, n_points):
        return random_cluster(xc=xc, yc=yc, n_points=n_points)

    def __len__(self):
        return self.n_points * 3

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))
