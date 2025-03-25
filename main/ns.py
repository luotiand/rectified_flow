import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import subprocess  # 引入 subprocess 模块
from script.plot import plot_2d_results, plot_results, hinton,plot_3d_compare_with_diff
from script.ode_data import Eq1, WaveEquation, PoissonEquation, HeatEquation
from rectified.rectified_flow import RectFlow
import time
import matplotlib.pyplot as plt
from script.dataset import MyDataset_ns
from torch.utils.data import DataLoader
from scorenet.scorenet import MLP1d, MLP2d, FNO, CNN,MLP2d_ns,CNN_add
import argparse
import logging
torch.set_default_dtype(torch.double)
torch.backends.cudnn.benchmark = True

def setup_logger(save_path):
    """配置日志记录器"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'training.log')),
            logging.StreamHandler()
        ]
    )


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
    setup_logger(save_path)
    logging.info("Initializing training process...")
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

    train_dataset = MyDataset_ns(
        file_path='/remote-sync/luotian.ding/my_project/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=True
    )
    
    test_dataset = MyDataset_ns(
        file_path='/remote-sync/luotian.ding/my_project/rectified_flow/data/ns/ns_V1e-3_N5000_T50.mat',
        input_steps=10,
        pred_steps=10,
        train=False
    )
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    # 动态加载模型类
    scorenet_model = globals()[scorenet_model_class](dim = 64, h_dim = 1024)
    logging.info(f"Model file will be saved as: {model_name}")

    # 检查多 GPU 环境
    free_gpu_index = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu_index}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logging.info(f"Detected {torch.cuda.device_count()} GPU(s)")
        logging.info(f"GPU Name: {torch.cuda.get_device_name(free_gpu_index)}")
        logging.info(f"CUDA Memory Usage: Allocated = {torch.cuda.memory_allocated(device)/1024**2:.2f} MB, "
                     f"Cached = {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")


    # 初始化数据
    sample_a, sample_x = test_dataset.get_full_data()
    a = sample_a.to(device).to(dtype=torch.float64)
    x = sample_x.to(device).to(dtype=torch.float64)

    # 初始化 score_net 模型，无论训练与否都需加载
    score_net = scorenet_model.to(device)
    if torch.cuda.device_count() > 1:
        score_net = nn.DataParallel(score_net)
        logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    if train:
        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        # 训练
        training_loss = [None for _ in range(niter)]
        start_time = time.time()  # 记录开始时间
        iter_start_time = start_time  # 每次迭代开始时间

        for it in range(niter):
            for a_, x_ in dataloader:
                a_ = a_.to(device).to(dtype=torch.float64)
                x_ = x_.to(device).to(dtype=torch.float64)

                t = torch.rand(batch_size, 1).to(device).to(dtype=torch.float64)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                opt.zero_grad()
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                score = score_net(a_, xt_, t)
                loss = criterion(exact_score, score)

                loss.backward()
                opt.step()
                del a_, x_, t, xt_, exact_score, score  # 删除未使用的变量以释放显存
                torch.cuda.empty_cache()
            training_loss[it] = loss.item()

            if (it+1) % (niter//8) == 0:
                new_lr = opt.param_groups[0]['lr'] / 8
                logging.info(f"Reducing learning rate from {opt.param_groups[0]['lr']} to {new_lr}")
                opt.param_groups[0]['lr'] = new_lr

            if (it + 1) % 5 == 0:
                iter_end_time = time.time()
                elapsed_time = iter_end_time - iter_start_time
                iter_start_time = iter_end_time
                estimated_remaining_time = (niter - it - 1) * (elapsed_time / 5)
                logging.info(f"Iteration {it + 1}/{niter}, Loss: {loss.item():.8f}")
                logging.info(f"Elapsed time for last 5 iterations: {elapsed_time:.2f} seconds")
                logging.info(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")

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
            score = score_net(a, xt[-1], t)
            xt_ = rf.forward_process(xt=xt[-1], score=score, dt=rf_dt)
            xt.append(xt_)

        yt = [x]
        for t in np.arange(start=0.0, stop=T, step=rf_dt):
            t = torch.ones_like(x) * t
            score = score_net(x, yt[-1], T - t)
            yt_ = rf.reverse_process(xt=yt[-1], score=score, dt=rf_dt)
            yt.append(yt_)
    abs_error = np.abs((xt[-1] - x).cpu().detach().numpy())    
    mae = np.mean(abs_error)
    # 计算相对误差（避免除以零）
    epsilon = 1e-6  # 小的正数，防止除以零
    # 只在非零值上计算相对误差
    mask = np.abs(xt[-1].cpu().detach().numpy()) > epsilon
    rel_error = np.abs(((xt[-1] - x).cpu().detach().numpy())[mask] / (np.abs((x.cpu().detach().numpy())[mask]) + epsilon)) * 100
    mre = np.mean(rel_error)
    
    # 计算其他误差指标
    rmse = np.sqrt(np.mean(np.square((xt[-1] - x).cpu().detach().numpy())))
    
    # 打印结果
    logging.info("\n" + "="*50)
    logging.info("误差统计 (取所有样本平均)")
    logging.info("="*50)
    logging.info(f"平均绝对误差 (MAE): {mae:.6f}")
    logging.info(f"平均相对误差 (MRE): {mre:.6f}%")
    logging.info(f"均xt[-1]方根误差 (RMSE): {rmse:.6f}")
    logging.info("="*50 + "\n")
    plot_2d_results(
        data1=torch.mean(xt[-1],dim = 1),
        data2=torch.mean(x,dim = 1),
        labels=['xt (2D)', 'x (exact 2D)'],
        title='Operator Learning: xt vs x (2D)',
        filename=f'{save_path}{scorenet_model_class.lower()}_operator_learning_2d.png'
    )

    plot_3d_compare_with_diff(
        data1=xt[-1].cpu().detach().numpy(), 
        data2=x.cpu().detach().numpy(),
        titles='Operator Learning: xt vs x (3D)',  # 自定义三个子图标题
        cmap_main='plasma',  # 主数据颜色映射
        cmap_diff='coolwarm',  # 差异场颜色映射
        elev=40,  # 俯视角度
        azim=-90, # 水平旋转角度
        filename=f'{save_path}{scorenet_model_class.lower()}_operator_learning_3d.png'       
    )

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