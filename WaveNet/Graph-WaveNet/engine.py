import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae  # 去除无效值后的均方绝对误差
        self.scaler = scaler
        self.clip = 5  # 梯度裁剪的阈值

    def train(self, input, real_val):
        """
        model.train()方法告诉模型在训练模式下运行。
        训练模式下，模型会启用 dropout 和 batch normalization 等训练时特有的操作。
        model.eval() 则是将模型切换为评估模式，此时模型会停用dropout，并使用训练时的统计数据进行 batch normalization。
        """
        self.model.train()
        self.optimizer.zero_grad()
        """
        torch.nn.functional.pad(input, pad, mode='constant', value=0)
        input:
        输入张量。它是需要进行填充的多维张量。
        
        pad:
        一个指定填充大小的元组 (pad_left, pad_right, pad_top, pad_bottom, ...)。
        pad 元组的顺序是从最后一个维度开始向前数，每两个为一个维度。例如：
        对于一个 2D 张量（如图像），pad 是 (pad_left, pad_right, pad_top, pad_bottom)。
        对于一个 3D 张量（如视频或彩色图像），pad 是 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)。
        
        mode:
        字符串，指定填充的模式。默认是 'constant'，即用常数进行填充。可选的模式有：
        'constant'：用常数进行填充。
        'reflect'：使用反射边界条件进行填充。
        'replicate'：使用复制边界条件进行填充。
        'circular'：使用循环（圆形）填充。
        
        value:
        用于填充的常数值，仅当 mode='constant' 时才需要指定。默认值为 0。
        """
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)  # torch.unsqueeze(input, dim) 在input中的dim维度插入一个新维度。该操作不改变数据内容，只修改张量的维度
        predict = self.scaler.inverse_transform(output)  # 训练前，对数据进行了标准化，忽略了原始数据的尺度等问题；训练后，要逆变化回来。
        loss = self.loss(predict, real, 0.0)
        loss.backward()  # 这里开始计算梯度，每个参数的梯度都保存在其.grad属性中。如a参数的梯度在a.grad中。
        if self.clip is not None:
            """
                torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)：梯度裁剪，防止梯度爆炸。norm_type范数类型默认是2范数
                计算梯度时，如果梯度超过了max_norm允许的最大梯度范数，则会按比例缩放（裁剪）这个值，使其不超过最大允许范数                
            """
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
