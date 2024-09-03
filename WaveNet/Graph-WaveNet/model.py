import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    """
    实现了一个图卷积的变体。它的输入包括一个特征张量 x 和一个权重矩阵 A，输出是一个新的特征张量，其形状和维度发生变化（从 ncvl 变成 ncwl）。
    这个过程可以看作是将节点特征沿图的边进行传播和聚合。
    图卷积的核心思想：将结点特征与邻居结点特征进行聚合。
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        x.shape = ncvl，A.shape = vw，张量爱因斯坦求和就是让x的某一维度与A进行矩阵乘法，实现维度变化
        n 是批次大小（batch size），
        c 是通道数（channels），
        v和w 是节点数量（例如图卷积中的节点），
        l 是特征维度（例如时间序列长度或其他维度）
        等价于
        # 重塑 x 使其适合与 A 相乘
        x = x.permute(0, 1, 3, 2)  # 现在 x 的形状为 (n, c, l, v)

        # 将 x 的最后一维与 A 相乘
        x = torch.matmul(x, A)  # 结果形状为 (n, c, l, w)

        # 重新排列维度使其与目标形状一致
        x = x.permute(0, 1, 3, 2)  # 结果形状为 (n, c, w, l)
        """
        x = torch.einsum('ncvl,vw->ncwl', (x, A))  # 这就相当于是在聚合了
        return x.contiguous()  # 保证返回的张量在内存中是连续的


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in  # 涉及多阶图的拼接，+1 的主要目的是保留原始特征，即没有经过卷积操作的特征
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order  # 阶数，表示使用的邻居数量

    def forward(self, x, support):
        out = [x]  # 保留了输入特征
        for a in support:
            x1 = self.nconv(x, a)  # 这相当于聚合每个节点及其直接邻居的特征
            out.append(x1)
            for k in range(2, self.order + 1):  # 阶数，考虑邻居的邻居
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  #
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    # aptinit：默认值是None。如果在train.py中，选择了args.randomadj随机初始化自适应邻接矩阵，则aptinit = None；如果没有选择，则aptinit = support[0]，即第一个邻接矩阵
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj  # 是否使用自适应邻接矩阵(boolean)

        # nn.ModuleList：PyTorch 中的一个容器类，可以被视为一个列表，用于存储一组子模块（例如层）
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()  # BatchNormal
        self.gconv = nn.ModuleList()

        # 就是结构图中第一个Linear，其作用与Linear相同，就是改变输入维度的
        '''
         卷积的理解：
         每个卷积核的shape为(height,width,in_channels)，每一个卷积核都会接收所有的输入通道数据，并产生一个输出通道的数据。
         但是卷积核的bachsize可以改变，也就是为out_channels，从而实现输出维度的改变
         卷积只在height, width, depth上进行操作。不会在channels和batch上进行操作
        '''
        self.start_conv = nn.Conv2d(in_channels=in_dim,  # 输入数据的通道数
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1  # 感受野

        self.supports_len = 0  # 邻接矩阵的个数
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:  # 初始化 nodevec1 和 nodevec2，这些是可学习的节点嵌入向量，用于构建自适应邻接矩阵。
            if aptinit is None:  # aptinit 自适应邻接矩阵的初始化值
                if supports is None:
                    self.supports = []
                # 对应论文里的公式(5)
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(
                    device)  # 将张量注册为可学习参数，成为模型的一部分，调用model.parameters()会返回它；参与梯度计算
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1  # nodevec1 和 nodevec2 之后会通过公式（5）生成一个自适应邻接矩阵，所以邻接矩阵数量先加一。
            else:
                if supports is None:
                    self.supports = []
                """
                奇异值分解（Singular Value Decomposition), SVD：是一种将一个矩阵分解成三个矩阵的技术
                A=UEV(T)
                A：待分解矩阵
                U与V(T，表示转置)，是正交矩阵，表示正交的特征向量。线性代数中，正交阵 X 正交阵的转置 = 单位阵
                E：是对角阵，包含奇异值
                当 aptinit（自适应邻接矩阵的初始化值）被提供时，我们希望从这个初始矩阵中提取一些信息来更好地初始化模型中的可学习节点嵌入向量（nodevec1 和 nodevec2）。
                """
                m, p, n = torch.svd(aptinit)
                # m[:, :10] 和 n[:, :10]：取 m 和 n 中前 10 个奇异向量。这些向量对应于最大的奇异值，代表了原始矩阵 aptinit 的主要特征。
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))  # 全称是 "matrix multiplication"（矩阵乘法）。
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # 循环构建网络。
        for b in range(blocks):  # 块 > 层。块是更高级的结构单元，层是基本组成单元。一个块内可能包含多个层。
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                """
                1x1 卷积不会改变输入的宽度或高度，只是用来调整通道数。
                """
                # dilated convolutions
                # tanh
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))  # (1,kernel_size)指在时间维度（宽）上进行滑动，捕获时间序列的局部特征

                # sigmoid
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,  # 由一维修改为了二维
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,  # 一维变二维
                                                 out_channels=skip_channels,
                                                 kernel_size=1))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # 最后的两个线性层
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)  # input:(64,2,207,13)
        # 如果输入的形状小于感受野，那么就padding一下。对齐的思想
        if in_len < self.receptive_field:  # self.receptive_field == 13
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        # 结构图中的第一个Linear，目的是改变维度
        x = self.start_conv(x)  # (64,2,207,13) -> (64,32,207,13)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # nodevec1和nodevec2是两个可学习参数，会随着模型更新，也就意味着邻接矩阵更新
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):  # 一个板块（block）内有两个层（TCN和GCN）

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # 门控TCN部分，提取时间特征
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            """
            为什么使用对应元素相乘？
            gate 的输出在 [0, 1] 范围内（因为 sigmoid 激活函数），用于控制哪些信息通过。
            filter 提供了实际的特征信息，而 gate 控制这些特征信息的通过程度。
            两个张量按元素相乘意味着，filter 中的每个值都会根据 gate 的值进行调制（也就是有选择性地放大或抑制）。
            """
            x = filter * gate  # 对应元素相乘，不是矩阵乘法

            # parametrized skip connection

            s = x  # x = filter * gate 更新后的x
            s = self.skip_convs[i](s)  # 其实仅仅只是调整了s的通道数

            try:
                # print("Try")
                skip = skip[:, :, :, -s.size(3):]  # 负数表示从后往前进行切片，仅选却倒数s.size(3)个元素
            except Exception as e:
                # print("\nError occurred:{}".format(e))
                # skip = 0
                # print("Except")
                skip = torch.zeros_like(s)
            # finally:
            #     print("Finally")

            skip = s + skip  # 保留每一个TCN的输出结果
            # GCN部分，提取空间特征
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:  # 是否使用自适应邻接矩阵
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)  # 如果不使用GCN，则通过1X1卷积，改变输出维度
            x = x + residual[:, :, :, -x.size(3):]  # 体现残差
            x = self.bn[i](x)  # 批归一化

        # x的结果其实都累积在了skip中。
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
