import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    # 手写DataLoader
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs: :param ys: :param batch_size: :param pad_with_last_sample: pad with the last sample to make number
        of samples divisible to batch_size.如果最后一个batch_size，不够分，则填充
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            '''
            解释为什么最后还需要取余：
            目的：避免被补充结果是batch_size的整数倍
            举例：
            batch_size = 4, len(xs) = 8
            batch_size - (len(xs) % batch_size) = 4 - 0 = 4
            意味着batch_size刚好能分完，也要补充一个batch_size
            (batch_size - (len(xs) % batch_size)) % batch_size = (4 - 0) % 4 = 4 % 4 = 0
            '''
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            '''
            axis = 0 -> 按行处理
            axis = 1 -> 按列处理
            '''
            x_padding = np.repeat(xs[-1:], num_padding,
                                  axis=0)  # 作用：选择最后一个元素（列表的形式），重复num_padding次。关键点：xs[-1:]返回值仍然是一个列表，一个包含最后一个值的列表
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)  # 已经使用了整除，所以可以不用int()强制转换
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(
            self.size)  # permutation：排列。np.random.permutation(n) 会生成一个包含从 0 到 n-1 的数字的随机排列数组。
        xs, ys = self.xs[permutation], self.ys[permutation]  # a = [1,2,3] -> a[2,0,1] 随机排列结果为[3,1,2]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0  # 索引，用于跟踪当前批次的索引位置。初始值为 0，表示开始从数据集的第一个批次开始。

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))  # min：防止结尾索引超出范围
                x_i = self.xs[start_ind: end_ind, ...]  # ...：是省略号表示符，用于保留原始数据的其他维度。
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i  # 用于生成器函数。它会将当前批次的输入数据 x_i 和标签 y_i 返回给调用者，并暂停函数的执行。下次调用生成器时，函数会从暂停的地方继续执行。使用next()输出下一个值
                self.current_ind += 1

        return _wrapper()  # _wrapper 函数的生成器对象。不是函数，而是对象。有点闭包的感觉。


# 以下全是不同方式的归一化
class StandardScaler():
    """
    Standard the input
    归一化：转化为正态分布
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std  # 方差

    def transform(self, data):  # 标准化
        return (data - self.mean) / self.std

    def inverse_transform(self, data):  # 逆标准化。
        return (data * self.std) + self.mean


def sym_adj(adj):
    """
    无向图
    Symmetrically normalize adjacency matrix.
    对称归一化邻接矩阵，计算方式与GCN的推导相似，D^-0.5 * A * D^-0.5，其中 A 是邻接矩阵，D 是节点度数的对角矩阵。
    作用：将邻接矩阵进行归一化处理，使得每一行（或每一列）的和都等于1，同时保持矩阵的对称性
    目的：解决在使用邻接矩阵进行图神经网络训练时出现的两个问题——度数偏差和梯度消失
    对称归一化是图神经网络（GNN）中的一种常见操作，用于确保图卷积在不同节点度数下表现稳定
    """
    adj = sp.coo_matrix(adj)  # 将输入的邻接矩阵转换为COO格式的稀疏矩阵。COO格式（坐标格式）是一种用于存储稀疏矩阵的格式，适合对稀疏矩阵进行高效的数学运算。
    rowsum = np.array(adj.sum(axis=1)).flatten()  # axis = 1，按行计算
    d_inv_sqrt = np.power(rowsum, -0.5)  # numpy.power 是 NumPy 提供的用于执行幂运算的函数。它可以对数组中的每个元素进行幂计算
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 避免出现无穷大
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 创建一个稀疏对角矩阵。稀疏阵是为了计算方便
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(
        np.float32).todense()  # .transpose()维度交换。对于高纬矩阵，可以选择交换的维度；对于二维矩阵，相当于转置


def asym_adj(adj):
    """
    有向图
    Asymmetrically normalize adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    归一化拉普拉斯矩阵
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    # 计算缩放的拉普拉斯矩阵
    if undirected:  # 无向图
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    # adj邻接矩阵其实是一个列表
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":  # 默认参数
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]  # 考虑图的正向和反向传播。邻接矩阵转置后表示入度。
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}  # 字典
    for category in ['train', 'val', 'test']:  # 独立同分布
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


# mask掩码是为了忽略无效数据
def masked_mse(preds, labels, null_val=np.nan):
    # 均方误差
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)  # # 创建mask矩阵，标记不是 NaN 的元素，返回的是boolean矩阵。~表示取非
    else:
        mask = (labels != null_val)  # null_val：缺失值。如果null_val不是NaN，那么labels != null_val，会返回有效值的boolean矩阵
    mask = mask.float()  # 转为浮点型张量。mask本来是布尔型，转为浮点型后就全是0和1
    mask /= torch.mean((mask))  # 归一化，使有效数据的mask均值为1
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask),
                       mask)  # 将 mask 中的 NaN 值替换为 0。torch.where(condition,x,y)：condition = True，选择x对应位置的值；condition
    # = False，选择y对应位置的值
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)  # 返回一个标量，即一个值


def masked_rmse(preds, labels, null_val=np.nan):
    # 根均方误差 -> 均方误差开根号
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    # 平均绝对误差
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    # 平均绝对百分比误差
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
