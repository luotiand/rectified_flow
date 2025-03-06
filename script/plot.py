import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

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