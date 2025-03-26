import torch.nn as nn
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.double)
class MLP1d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 200) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(0.5),   
            nn.Tanh(),
            nn.Linear(h_dim, 4*h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)
        return self.net(x)
class MLP2d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(int(2 * dim**2), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, int(dim**2))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        n = x.shape[0]
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)

        x = x.view(n, -1)
        y = self.net(x)
        z = y.view(n, int(self.dim), int(self.dim)) 
        
        return z

class MLP2d_add(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(int(3 * dim**2), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, int(dim**2))
        )

    def forward(self,a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        n = x.shape[0]
        a_mean = a.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        a_std = a.std(dim=(1,2), keepdim=True) + 1e-6
        a = (a - a_mean) / a_std
        x_mean = x.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        x_std = x.std(dim=(1,2), keepdim=True) + 1e-6
        x = (x - x_mean) / x_std
        t = t-0.5
        t = t.expand_as(x)
        x = torch.cat((x, a), dim=-1)
        x = torch.cat((x, t), dim=-1)
        x = x.view(n, -1)
        y = self.net(x)
        z = y.view(n, int(self.dim), int(self.dim)) 
        z = z*a_std+a_mean
        return z

# class MLP2d_ns(nn.Module):
#     def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
#         super().__init__()
#         self.dim = dim
#         self.net = nn.Sequential(
#             nn.Linear(int(3 * dim**2), h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, 4*h_dim),
#             nn.ReLU(),
#             nn.Linear(4*h_dim, 4*h_dim),
#             nn.ReLU(),
#             nn.Linear(4*h_dim, int(dim**2))
#         )
        

#     def forward(self,a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
#         n = x.shape[0]
#         m = x.shape[1]
#         a = a.to(dtype=torch.float64)
#         a_mean = a.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
#         a_std = a.std(dim=(1,2), keepdim=True) + 1e-6
#         a = (a - a_mean) / a_std
#         x_mean = x.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
#         x_std = x.std(dim=(1,2), keepdim=True) + 1e-6
#         x = (x - x_mean) / x_std
#         t = t-0.5
#         t = t.expand_as(x)
#         x = torch.cat((x, a), dim=-1)
#         x = torch.cat((x, t), dim=-1)
#         x = x.view(n, m,-1)
#         y = self.net(x)
#         z = y.view(n,m, int(self.dim), int(self.dim)) 
#         z = z*a_std+a_mean
#         return z

class MLP2d_ns(nn.Module):
    def __init__(self, 
                 dim: int = 64, 
                 h_dim: int = 1024,
                 time_steps: int = 10,
                 k_list: list = [1, 2]):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps
        self.k_list = k_list
        
        # 时间核参数化
        self.time_kernel = nn.Sequential(
            nn.Linear(6, dim**2),  # 6个特征 → 空间维度
            nn.ReLU(),
            nn.Linear(dim**2, dim**2)
        )
        
        # 主网络 (输入维度调整为3*dim²)
        self.main_net = nn.Sequential(
            nn.Linear(3*dim**2, h_dim),  # 新增x特征
            nn.ReLU(),
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, dim**2)
        )

    def generate_time_features(self, t: torch.Tensor) -> torch.Tensor:
        """生成6通道时间特征"""
        # 原始t形状: [bs, 1, 1, 1]
        bs = t.shape[0]
        
        # 扩展至 [bs, time_steps, 1]
        t_expanded = t.squeeze(-1).squeeze(-1)  # [bs, 1]
        t_expanded = t_expanded.unsqueeze(1).expand(-1, self.time_steps, -1)  # [bs, T, 1]
        
        # 生成6个核特征
        features = []
        for k in self.k_list:
            # clip核
            features.append((t_expanded / k).clamp(0, 1))
            # exp核
            features.append(1 - torch.exp(-t_expanded / k))
            # sin核
            features.append(torch.sin(t_expanded * torch.pi / k))
        
        # 合并特征 [bs, T, 6]
        return torch.cat(features, dim=-1)

    def forward(self, a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        # 输入形状验证
        assert a.dim() == 4, "输入a应为4维张量"
        bs, T, H, W = a.shape
        
        # 标准化x
        x_mean = x.mean(dim=(2,3), keepdim=True)
        x_std = x.std(dim=(2,3), keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std
        a_mean = a.mean(dim=(2,3), keepdim=True)
        a_std = a.std(dim=(2,3), keepdim=True) + 1e-6
        a_norm = (a - a_mean) / a_std
        # 生成时间特征 [bs, T, 6]
        time_feat = self.generate_time_features(t)
        
        # 时间特征映射到空间维度 [bs, T, H*W]
        time_feat = self.time_kernel(time_feat)  # [bs, T, H*W]
        time_feat = time_feat.view(bs, T, H, W)  # [bs, T, H, W]
        
        # 特征融合 (新增x)
        combined = torch.cat([
            a_norm.view(bs, T, -1),    # 原始a特征 [bs, T, H*W]
            x_norm.view(bs, T, -1), # 标准化x特征 [bs, T, H*W]
            time_feat.view(bs, T, -1) # 时间特征 [bs, T, H*W]
        ], dim=-1)  # [bs, T, 3*H*W]
        
        # 主网络处理
        output = self.main_net(combined)  # [bs, T, H*W]
        
        # 反标准化恢复x尺度
        return output.view(bs, T, H, W) * x_std + x_mean

class CNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, width=4):
        super().__init__()
        self.width = width
        self.device = "cuda"
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.Conv2d(width, width * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.Sigmoid(),
        )
        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.out = nn.Conv2d(out_channels,out_channels,kernel_size =1,stride = 1, padding=0)
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Expand time input and stack with spatial input
        t = t.expand_as(x).to(self.device)  # Ensure `t` is on the same device as `x`
        x_1 = torch.stack([x, t], dim=1).to(self.device)  # Ensure `x` is on the same device as `t`

        # Down-sampling + residual connection
        x_2 = self.down(x_1)
        x_3 = self.up(x_2)
        x_4 = self.out(x_3)
        z = x_4.squeeze(1)+x
        return z

# class CNN_add(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, width=16):
#         super().__init__()
#         self.width = width
#         self.device = "cuda"
#         # Down-sampling layers
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, width, kernel_size=8, stride=2, padding=3),
#             nn.Sigmoid(),
#             nn.Conv2d(width, width * 4, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid(),
#             nn.Conv2d(width*4, width * 8, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#         )
#         # Up-sampling layers
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(width * 8, width*4, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#             nn.ConvTranspose2d(width * 4, width, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid(),
#             nn.ConvTranspose2d(width, out_channels, kernel_size=8, stride=2, padding=3),
#         )
#         self.out = nn.Conv2d(out_channels,out_channels,kernel_size =1,stride = 1, padding=0)
#     def forward(self,a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
#         # Expand time input and stack with spatial input
#         t = t.expand_as(x).to(self.device)  # Ensure `t` is on the same device as `x`
#         x_1 = torch.stack([a, x, t], dim=1).to(self.device)  # Ensure `x` is on the same device as `t`
#         # Down-sampling + residual connection
#         x_2 = self.down(x_1)
#         x_3 = self.up(x_2)
#         x_4 = self.out(x_3)
#         z = x_4.squeeze(1)+x
#         return z

class CNN_add(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=16):
        super().__init__()
        self.width = width
        self.device = "cuda"
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.out = nn.Conv2d(out_channels,out_channels,kernel_size =1,stride = 1, padding=0)
    def forward(self,a: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        a_mean = a.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        a_std = a.std(dim=(1,2), keepdim=True) + 1e-6
        a = (a - a_mean) / a_std
        x_mean = x.mean(dim=(1,2), keepdim=True)  # [bs,1,1]
        x_std = x.std(dim=(1,2), keepdim=True) + 1e-6
        x = (x - x_mean) / x_std
        t = t-0.5
        # Expand time input and stack with spatial input
        t = t.expand_as(x).to(self.device)  # Ensure `t` is on the same device as `x`
        x_1 = torch.stack([a, x, t], dim=1).to(self.device)  # Ensure `x` is on the same device as `t`
        # Down-sampling + residual connection
        x_2 = self.down(x_1)
        x_3 = self.up(x_2)
        z = x_3.squeeze(1)+x
        return z

class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param modes: 傅立叶变换中保留的频率模式数量
        """
        super(FourierLayer2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 实部和虚部的权重矩阵 (用于低频模式的线性操作)
        self.weights_real = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes))
        self.weights_imag = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes))

    def forward(self, x):
        # 对输入 x 进行傅立叶变换，形状为 (batch_size, channels, height, width)
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # 保留傅立叶变换的低频部分（前 self.modes 个频率模式）
        x_ft = x_ft[:, :, :self.modes, :self.modes]

        # 分别对实部和虚部应用权重矩阵
        x_ft_real = torch.einsum("bixy,ioxy->boxy", x_ft.real, self.weights_real) - torch.einsum("bixy,ioxy->boxy", x_ft.imag, self.weights_imag)
        x_ft_imag = torch.einsum("bixy,ioxy->boxy", x_ft.real, self.weights_imag) + torch.einsum("bixy,ioxy->boxy", x_ft.imag, self.weights_real)

        # 合并实部和虚部，得到更新后的傅立叶系数
        x_ft_updated = torch.complex(x_ft_real, x_ft_imag)

        # 执行逆傅立叶变换，将数据转换回空间域
        x = torch.fft.irfft2(x_ft_updated, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x


class FNO(nn.Module):
    def __init__(self, in_channels = 2 , out_channels = 1, dim = 100, cutoff_ratio = 0.1, width = 64, layers = 4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param dim: 输入数据的维度（例如，64表示64x64的输入）
        :param cutoff_ratio: 保留低频模式的比例
        :param width: 中间层的宽度
        :param layers: 傅立叶层的数量
        """
        super(FNO, self).__init__()
        self.width = width

        # 根据输入维度计算 modes
        modes = int(dim * cutoff_ratio)

        # 初始卷积层，将输入映射到一个高维空间
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        # 创建多个傅立叶层
        self.fourier_layers = nn.ModuleList([FourierLayer2D(width, width, modes) for _ in range(layers)])

        # 最后的全连接层，将高维数据映射到输出空间
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):

#         """
        n = x.shape[0]
         # Expand t to match x's dimensions and concatenate along the channel axis
        t = t.expand_as(x)
        x = torch.stack([x, t], dim=1)  # Stack along channel dimension to create 2 input channels
        # 执行初始的卷积操作
        x = self.fc0(x)

        # 通过多个傅立叶层
        for layer in self.fourier_layers:
            x = layer(x)

        # 执行最后的卷积操作，将高维数据映射回到输出空间
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1)  # 移除通道维度，变为 (b, d, d)

        return x