import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
    


class MyDataset_ns(Dataset):
    def __init__(self, 
                 file_path: str,
                 input_steps: int = 10,
                 pred_steps: int = 10,
                 train: bool = True,
                 train_ratio: float = 0.9,
                 seed: int = 42):
        """
        Navier-Stokes 时间序列预测数据集
        参数说明:
            file_path: HDF5文件路径
            input_steps: 输入时间步数
            pred_steps: 预测时间步数
            train: 是否为训练集
            train_ratio: 训练集比例
            seed: 随机种子
        """
        # 读取原始数据
        with h5py.File(file_path, 'r') as f:
            u = np.array(f['u'][()], dtype=np.float32)  # (50, 64, 64, 5000)
        
        # 调整维度顺序 (样本数, 时间步, 空间X, 空间Y)
        u_processed = u.transpose(3, 0, 1, 2)  # (5000, 50, 64, 64)
        
        # 分割输入输出时间窗口
        self.x = u_processed[:, :input_steps, :, :]          # (5000, 10, 64, 64)
        self.y = u_processed[:, input_steps:input_steps+pred_steps, :, :]  # (5000, 10, 64, 64)
        
        # 数据集分割
        np.random.seed(seed)
        indices = np.random.permutation(self.x.shape[0])
        split_idx = int(len(indices) * train_ratio)
        
        self.indices = indices[:split_idx] if train else indices[split_idx:]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_sample = torch.from_numpy(self.x[real_idx])  # (10, 64, 64)
        y_sample = torch.from_numpy(self.y[real_idx])  # (10, 64, 64)
        
        # 添加通道维度 (适配CNN输入)
        return x_sample, y_sample # (10, 1, 64, 64)

# 使用示例 -------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化数据集
    train_dataset = MyDataset_ns(
        file_path='/home/luotian.ding/myproject/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=True
    )
    
    test_dataset = MyDataset_ns(
        file_path='/home/luotian.ding/myproject/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=False
    )
    
    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 验证数据形状
    sample_x, sample_y = next(iter(train_loader))
    print("输入数据维度:", sample_x.shape)   # 预期: [32, 10, 1, 64, 64]
    print("输出数据维度:", sample_y.shape)   # 预期: [32, 10, 1, 64, 64]