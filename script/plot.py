import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_compare_with_diff(data1, data2, titles, cmap_main='viridis', cmap_diff='seismic', 
                             elev=25, azim=-135, filename='operator_learning_3d.png' ,figsize=(24, 8)):
    """
    并排显示 data1、data2 和差异场的 3D 对比
    
    参数：
        data1: 数据1 (N_test, T, H, W)
        data2: 数据2 (N_test, T, H, W)
        titles: 标题列表 [data1_title, data2_title, diff_title]
        cmap_main: 主数据颜色映射
        cmap_diff: 差异颜色映射
        elev: 俯仰角
        azim: 方位角
        figsize: 画布尺寸
    """
    # 数据预处理
    abs_error = np.abs(data1 - data2)
    mae = np.mean(abs_error)
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean(np.square(data1 - data2)))
    
    # 计算更合理的相对误差
    # 方法1：相对于数据范围的归一化误差
    data_range = np.max(data1) - np.min(data1)
    nrmse = rmse / data_range * 100  # 归一化RMSE（百分比）
    
    # 方法2：使用更稳健的相对误差计算（考虑数据的典型尺度）
    data_scale = np.mean(np.abs(data1))  # 数据绝对值的平均数作为尺度
    scaled_error = np.mean(abs_error) / data_scale * 100 if data_scale > 0 else 0
    
    # 打印结果
    print("\n" + "="*50)
    print("误差统计 (取所有样本平均)")
    print("="*50)
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"归一化RMSE (NRMSE): {nrmse:.2f}%")
    print(f"相对于数据尺度的误差: {scaled_error:.2f}%")
    
    # 添加额外的描述性统计
    print("\n数据统计:")
    print(f"参考数据范围: [{np.min(data1):.4f}, {np.max(data1):.4f}], 均值: {np.mean(data1):.4f}")
    print(f"预测数据范围: [{np.min(data2):.4f}, {np.max(data2):.4f}], 均值: {np.mean(data2):.4f}")
    print("="*50 + "\n")
    data1_avg = data1.mean(axis=0)  # (T, H, W)
    data2_avg = data2.mean(axis=0)
    diff = data1_avg - data2_avg
    
    # 创建画布和子图
    fig = plt.figure(figsize=figsize)
    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d')
    ]
    
    # 绘制各子图
    _plot_3d_field(axes[0], data1_avg, title=titles[0], cmap=cmap_main, elev=elev, azim=azim)
    _plot_3d_field(axes[1], data2_avg, title=titles[1], cmap=cmap_main, elev=elev, azim=azim)
    _plot_3d_diff(axes[2], diff, title=titles[2], cmap=cmap_diff, elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
def _plot_3d_field(ax, data, title, cmap, elev, azim):
    """绘制单个 3D 场"""
    T, H, W = data.shape
    
    # 修复：生成比数据多一个点的网格
    x, y, z = np.meshgrid(
        np.arange(W+1), 
        np.arange(H+1), 
        np.arange(T+1), 
        indexing='ij'
    )

    # 生成等值面 - 确保数据与网格匹配
    threshold = np.median(data)
    
    # 将布尔数组调整为与网格相同大小
    voxel_array = np.zeros((W+1, H+1, T+1), dtype=bool)
    voxel_array[:-1, :-1, :-1] = data.transpose(2, 1, 0) > threshold
    
    # 调整颜色数组大小
    colors = np.zeros((W+1, H+1, T+1, 4))
    normalized_data = data.transpose(2, 1, 0) / data.max()
    colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(normalized_data)
    
    # 绘制体素
    ax.voxels(voxel_array, facecolors=colors, edgecolor='none', alpha=0.3)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Width', labelpad=10)
    ax.set_ylabel('Height', labelpad=10)
    ax.set_zlabel('Time Step', labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    
    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(data)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)

def _plot_3d_diff(ax, diff, title, cmap, elev, azim):
    """绘制差异场"""
    T, H, W = diff.shape
    
    # 修复：生成比数据多一个点的网格
    x, y, z = np.meshgrid(
        np.arange(W+1), 
        np.arange(H+1), 
        np.arange(T+1), 
        indexing='ij'
    )
    
    # 归一化差异值
    vmax = np.abs(diff).max()
    norm_diff = diff / vmax
    
    # 创建比数据大一维的布尔掩码
    pos_mask = np.zeros((W+1, H+1, T+1), dtype=bool)
    neg_mask = np.zeros((W+1, H+1, T+1), dtype=bool)
    
    # 填入数据
    pos_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) > 0
    neg_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) < 0
    
    # 创建颜色数组
    pos_colors = np.zeros((W+1, H+1, T+1, 4))
    neg_colors = np.zeros((W+1, H+1, T+1, 4))
    
    # 填入颜色数据
    pos_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0)/2)
    neg_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0)/2)
    
    # 绘制体素
    ax.voxels(pos_mask, facecolors=pos_colors, edgecolor='none', alpha=0.5)
    ax.voxels(neg_mask, facecolors=neg_colors, edgecolor='none', alpha=0.5)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Width', labelpad=10)
    ax.set_ylabel('Height', labelpad=10)
    ax.set_zlabel('Time Step', labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
# 使用示例 -------------------------------------------------
# 生成示例数据
data1 = np.random.randn(100, 10, 64, 64).clip(-1,1)  # (N_test=100, T=10, H=64, W=64)
data2 = 0.8 * data1 + np.random.randn(100, 10, 64, 64) * 0.2

# 可视化对比
plot_3d_compare_with_diff(
    data1, data2,
    titles=["High-Resolution Model", "Low-Resolution Model", "Model Difference"],
    cmap_main='viridis',
    cmap_diff='seismic',
    elev=30,
    azim=-120
)

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    plt.figure(figsize=(6, 6))
    ax = ax if ax is not None else plt.gca()

    if max_weight is None:
        max_weight = 2**np.ceil(np.log(np.max(np.abs(matrix)))/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('auto', 'box')
    ax.set_xticks([])
    ax.set_yticks([])

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.title("Hinton Diagram of Model Weights")
    plt.savefig('hint', dpi=300)
    plt.show()

# Plot the Hinton diagram

def plot_2d_results(data1, data2, labels, title, filename):
    """
    绘制二维数据的比较图和差值图并保存。

    :param data1: 第一个二维数据集 (如 xt)
    :param data2: 第二个二维数据集 (如 x)
    :param labels: 数据集标签
    :param title: 图表标题
    :param filename: 保存的文件名
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制第一个数据集
    im1 = axes[0].imshow(data1.mean(axis=0).cpu().detach().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title(labels[0])
    fig.colorbar(im1, ax=axes[0])

    # 绘制第二个数据集
    im2 = axes[1].imshow(data2.mean(axis=0).cpu().detach().numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title(labels[1])
    fig.colorbar(im2, ax=axes[1])

    # 计算并绘制差值
    diff = (data1 - data2).mean(axis=0).cpu().detach().numpy()
    im3 = axes[2].imshow(diff, cmap='seismic', aspect='auto')  # 使用 'seismic' 颜色图来突出差异
    axes[2].set_title(f"Difference ({labels[0]} - {labels[1]})")
    fig.colorbar(im3, ax=axes[2])

    # 总标题
    plt.suptitle(title)
    # 保存图像
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_results(data1, data2, labels, title, xlabel, ylabel, filename):
    """
    绘制数据的比较图并保存。

    :param data1: 第一个数据集
    :param data2: 第二个数据集
    :param labels: 数据集标签
    :param title: 图表标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    :param filename: 保存的文件名
    """
    plt.figure()
    plt.plot(data1.mean(dim=1).cpu().detach().numpy(), label=labels[0].format(len(data1)))
    plt.plot(data2.mean(dim=1).cpu().detach().numpy(), label=labels[1].format(len(data2)))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()