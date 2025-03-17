import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess  # 引入 subprocess 模块
from script.plot import plot_2d_results, plot_results, hinton
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
import matplotlib.pyplot as plt
from script.dataset import MyDataset
from torch.utils.data import DataLoader
from scorenet.scorenet import MLP1d, MLP2d, FNO, CNN
import argparse
torch.set_default_dtype(torch.double)
torch.backends.cudnn.benchmark = True

def squared_absolute_error_loss(output, target):
    return torch.mean((torch.abs(output - target)) ** 2)

def get_free_gpu():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ]
    )
    memory_free = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return memory_free.index(max(memory_free))

def main(config):
    # 从配置字典中提取参数
    save_path = config['save_path']
    para_path = config['para_path']
    eq_T = config['eq_T']
    N = config['N']
    niter = config['niter']
    lr = config['lr']
    batch_size = config['batch_size']
    T = config['T']
    eq_dt = config['eq_dt']
    rf_dt = config['rf_dt']
    h_dim = config['h_dim']
    train = config['train']
    scorenet_model_class = config['scorenet_model_class']
    device = config['device']
    model_name = config['model_name']
    rf = config['rf']
    eq = config['eq']

    dataset = MyDataset(eq, dt=eq_dt, N=N)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 动态加载模型类
    scorenet_model = globals()[scorenet_model_class](in_channels=2, out_channels=1, width=4)
    print(f"Model file will be saved as: {model_name}")

    # 检查多 GPU 环境
    free_gpu_index = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"GPU Name: {torch.cuda.get_device_name(free_gpu_index)}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")


    # 初始化数据
    a, x = eq.sample(dt=eq_dt, N=10)
    a = torch.from_numpy(a).to(device)
    x = torch.from_numpy(x).to(device)

    # 初始化 score_net 模型，无论训练与否都需加载
    score_net = scorenet_model.to(device)
    if torch.cuda.device_count() > 1:
        score_net = nn.DataParallel(score_net)

    if train:
        if device.type == 'cuda':
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
            print(f"GPU Max Memory Usage: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")

        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        # scaler = torch.cuda.amp.GradScaler()  # 使用混合精度训练

        # 训练
        training_loss = [None for _ in range(niter)]
        start_time = time.time()  # 记录开始时间
        iter_start_time = start_time  # 每次迭代开始时间

        for it in range(niter):
            for a_, x_ in dataloader:
                a_ = a_.to(device)
                x_ = x_.to(device)

                t = torch.rand(batch_size, 1).to(device)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                opt.zero_grad()
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                score = score_net(xt_, t)
                loss = criterion(exact_score, score)
                loss.backward()
                opt.step()
                del a_, x_, t, xt_, exact_score, score  # 删除未使用的变量以释放显存
                torch.cuda.empty_cache()

            training_loss[it] = loss.item()

            if ((it + 1) % (niter * 1 // 5) == 0):
                opt.param_groups[0]["lr"] /= 8

            if (it + 1) % 5 == 0:
                iter_end_time = time.time()
                elapsed_time = iter_end_time - iter_start_time
                iter_start_time = iter_end_time
                estimated_remaining_time = (niter - it - 1) * (elapsed_time / 5)
                print(f"Iteration {it + 1}/{niter}, Loss: {loss.item():.4f}")
                print(f"Elapsed time for last 5 iterations: {elapsed_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")

        torch.save(score_net.state_dict(), f"{para_path}{model_name}")

        # visualize loss curve
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.plot(training_loss)
        ax.set_yscale("log")
        ax.set_title("Rectified Flow Training", fontdict={"size": 20})
        ax.set_xlabel("Iteration", fontdict={"size": 16})
        ax.set_ylabel("Training Loss", fontdict={"size": 16})
        ax.grid(True)
        fig.savefig(f"{save_path}{scorenet_model_class.lower()}_training_loss.png", dpi=300)
        plt.cla()
        plt.clf()
        plt.close()
        # 加载模型用于测试/评估
        score_net.load_state_dict(torch.load(f"{para_path}{model_name}"))
    score_net.eval()  # <-- 添加这行切换到评估模式
    with torch.no_grad():
        # 评估 Operator Learning 和逆问题
        xt = [a]
        for t in np.arange(start=0.0, stop=T, step=rf_dt):
            t = torch.ones_like(x) * t
            score = score_net(xt[-1], t)
            xt_ = rf.forward_process(xt=xt[-1], score=score, dt=rf_dt)
            xt.append(xt_)

        yt = [x]
        for t in np.arange(start=0.0, stop=T, step=rf_dt):
            t = torch.ones_like(x) * t
            score = score_net(yt[-1], T - t)
            yt_ = rf.reverse_process(xt=yt[-1], score=score, dt=rf_dt)
            yt.append(yt_)

    # 绘制结果
    plot_2d_results(
        data1=xt[-1],
        data2=x,
        labels=['xt (2D)', 'x (exact 2D)'],
        title='Operator Learning: xt vs x (2D)',
        filename=f'{save_path}{scorenet_model_class.lower()}_operator_learning_2d.png'
    )

    plot_2d_results(
        data1=yt[-1],
        data2=a,
        labels=['yt (2D)', 'a (exact 2D)'],
        title='Inverse Problem: yt vs a (2D)',
        filename=f'{save_path}{scorenet_model_class.lower()}_inverse_problem_2d.png'
    )

    print(f"xt[-1] shape: {xt[-1].shape}, x shape: {x.shape}")
    print(f"yt[-1] shape: {yt[-1].shape}, a shape: {a.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    # 动态加载配置文件
    config_path = args.config
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)

    main(config)
