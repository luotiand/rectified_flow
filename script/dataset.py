from torch.utils.data import Dataset, DataLoader
import torch
class MyDataset(Dataset):
    def __init__(self, equation, dt=0.01, N=10000):
        """
        初始化数据集
        :param equation: 用于生成数据的 Poisson 方程实例
        :param dt: 时间步长
        :param N: 样本数
        """
        self.eq = equation
        self.dt = dt
        self.N = N
        self.a, self.x = self.eq.sample(dt=self.dt, N=self.N)  # 生成样本数据

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本索引
        :return: (a, x) 对
        """
        a_sample = self.a[idx]
        x_sample = self.x[idx]
        return torch.from_numpy(a_sample), torch.from_numpy(x_sample)
class MyDataset2(Dataset):
    def __init__(self, a,x):
        """
        初始化数据集
        :param equation: 用于生成数据的 Poisson 方程实例
        :param dt: 时间步长
        :param N: 样本数
        """
        self.a, self.x = a,x  # 生成样本数据

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本索引
        :return: (a, x) 对
        """
        a_sample = self.a[idx]
        x_sample = self.x[idx]
        return a_sample, x_sample