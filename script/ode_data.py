#!-*- coding:utf-8 -*-

import numpy as np
class Eq1(object):
    def __init__(self, a0: float=-1.0, a1: float=1.0, x0: float=1.0, T: float=1.0) -> None:
        self.a0 = a0
        self.a1 = a1
        self.x0 = x0
        self.T = T

    def sample(self, dt: float=0.1, N: int=100):
        nt = int(self.T / dt)
        a = np.random.uniform(low=self.a0, high=self.a1, size=(N,)).reshape(-1, 1).repeat(nt, axis=1)
        t = np.linspace(0.0, self.T, num=nt, endpoint=False)
        x = self.x0 * np.exp(a * t.reshape(1, -1))
        return (a, x)

class WaveEquation(object):
    def __init__(self, a0: float = 0.5, a1: float = 2.0, x0: float = 0.0, T: float = 1.0, A: float = 1.0) -> None:
        """
        类Wave方程的初始化
        a0, a1: 控制波速c的范围
        x0: 初始位置
        T: 时间范围
        A: 振幅，设为常数
        """
        self.a0 = a0  # 波速的下界
        self.a1 = a1  # 波速的上界
        self.x0 = x0  # 初始位置
        self.T = T    # 时间长度
        self.A = A    # 振幅常数

    def sample(self, dt: float = 0.01, N: int = 100):
        """
        生成N组随机波速参数a，并计算Wave方程的解析解x
        """
        nt = int(self.T / dt)  # 时间步数
        t = np.linspace(0.0, self.T, num=nt, endpoint=False)  # 时间离散

        # 随机生成N组波速参数a
        a = np.random.uniform(low=self.a0, high=self.a1, size=(N, 1)).repeat(nt, axis=1)  # 形状为(N, nt)

        # 计算解 x = A * cos(a * t - x0)
        x = self.A * np.cos(a * t.reshape(1, -1) - self.x0)  # 解的形状与a相同 (N, nt)

        return (a, x)


class PoissonEquation(object):
    def __init__(self, k: float=1.0, a0: float=0.0, a1: float=np.pi, T: float=1.0) -> None:
        """
        Poisson 方程的初始化

        :param k: 正弦项的系数
        :param a0: x 和 y 的最小值
        :param a1: x 和 y 的最大值
        :param T: 时间 T （仅用于一致性）
        """
        self.k = k
        self.a0 = a0
        self.a1 = a1
        self.T = T

    def sample(self, dt: float=0.1, N: int=100):
        """
        生成 N 组泊松方程的源项 f 和对应的解析解 u，每个样本在二维网格上。

        :param dt: 网格步长
        :param N: 采样数量
        :return: N 组源项 f 和解析解 u
        """
        nt = int(self.T / dt)

        # 生成 x 和 y 坐标的网格
        x = np.linspace(self.a0, self.a1, nt)
        y = np.linspace(self.a0, self.a1, nt)
        X, Y = np.meshgrid(x, y)

        f_samples = []
        u_samples = []

        for _ in range(N):
            # 随机生成不同的 k 值（或者你可以固定 k）
            k_sample = np.random.uniform(low=self.k, high=self.k * 2)
            
            # 源项 f(x, y) = -k_sample^2 * sin(k_sample * x) * sin(k_sample * y)
            f = -k_sample**2 * np.sin(k_sample * X) * np.sin(k_sample * Y)
            
            # 解析解 u(x, y) = sin(k_sample * x) * sin(k_sample * y)
            u = np.sin(k_sample * X) * np.sin(k_sample * Y)

            f_samples.append(f)
            u_samples.append(u)

        # 将采样结果转为 numpy 数组
        f_samples = np.array(f_samples)
        u_samples = np.array(u_samples)

        return f_samples, u_samples

# class PoissonEquation(object):
#     def __init__(self, nx: int = 100, ny: int = 100, Lx: float = 1.0, Ly: float = 1.0) -> None:
#         """
#         Poisson 方程的初始化
#         nx, ny: 网格点数量
#         Lx, Ly: 定义域的长度
#         """
#         self.nx = nx  # x 方向的网格点数量
#         self.ny = ny  # y 方向的网格点数量
#         self.Lx = Lx  # x 方向的长度
#         self.Ly = Ly  # y 方向的长度
#         self.dx = Lx / (nx - 1)  # x 方向的网格步长
#         self.dy = Ly / (ny - 1)  # y 方向的网格步长

#     def sample(self, dt: float = 0.1, N: int = 100, tol: float = 1e-5, max_iter: int = 100):
#         """
#         使用有限差分法求解 N 个 Poisson 方程
#         随机生成 N 组源项 f(x, y)
#         tol: 收敛容忍度
#         max_iter: 最大迭代次数
#         """
#         # 初始化解 u 和随机生成的源项 a
#         u = np.zeros((N, self.nx, self.ny))  # N 组解的初始化
#         a = np.random.uniform(low=-1.0, high=1.0, size=(N, self.nx, self.ny))  # 生成 N 组随机源项 a

#         # 对每组 a 进行迭代求解 Poisson 方程
#         for iteration in range(max_iter):
#             u_old = u.copy()

#             # 迭代 N 次以更新每个 u[i] 的解
#             for k in range(N):  # 对于每个 a[k]
#                 for i in range(1, self.nx - 1):
#                     for j in range(1, self.ny - 1):
#                         u[k, i, j] = (self.dy**2 * (u_old[k, i+1, j] + u_old[k, i-1, j]) +
#                                     self.dx**2 * (u_old[k, i, j+1] + u_old[k, i, j-1]) -
#                                     self.dx**2 * self.dy**2 * a[k, i, j]) / (2 * (self.dx**2 + self.dy**2))

#             # 判断是否所有 u 都收敛
#             if np.max(np.abs(u - u_old)) < tol:
#                 print(f"Converged after {iteration} iterations.")
#                 break

#         return (a, u)  # 返回 N 组源项 a 和解 u


class HeatEquation(object):
    def __init__(self, x0: float=0.0, T: float=1.0) -> None:
        self.x0 = x0
        self.T = T

    def sample(self, dt: float=0.1, N: int=100):
        nt = int(self.T / dt)
        a = np.random.uniform(low=-1.0, high=1.0, size=(N, 2))  # 随机生成两个多项式系数
        t = np.linspace(0.0, self.T, num=nt, endpoint=False)
        # 示例：热传导方程的形式为 u_t = a*u_xx
        # 这里简化为 u(t) = x0 + a0*t + a1*t^2 (u的近似解)
        x = self.x0 + a[:, 0] * t.reshape(1, -1) + a[:, 1] * (t.reshape(1, -1) ** 2)
        return (a, x)
