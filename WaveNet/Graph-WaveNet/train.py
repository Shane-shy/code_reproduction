import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

from tqdm import tqdm
from contextlib import redirect_stdout

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='number of GPU')  # 多卡改为单卡
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')  # 是否随机初始化自适应邻接矩阵
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
# parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='./garage/metr/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype) # 其实就邻接矩阵用到了
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)  # batch_size = 64
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]  # adj_mx是一个包含邻接矩阵的列表，该列表可能只包含一个邻接矩阵，也可能包含两个邻接矩阵

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("-" * 50)
    print("start training...")
    his_loss = []  # 历史损失值
    val_time = []  # 评估时间
    train_time = []  # 训练时间
    for i in tqdm(range(1, args.epochs + 1), desc="training"):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):  # .get_iterator()返回一个迭代器
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            # if iter % args.print_every == 0:
            #     log = '\nIter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #     print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):  # (x,y)是同一个batch中的
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        torch.save(engine.model.state_dict(), args.save + "epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training finished")
    print("-" * 50)

    # Evaluating
    print("Evaluating start")
    bestid = np.argmin(his_loss)
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    engine.model.load_state_dict(
        torch.load(args.save + "epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():  # 不计算梯度
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in tqdm(range(12), desc="evaluating"): # 数据集中，传感器记录的时间窗口是5min，这里的12表示记录12 * 5 = 60 min
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    # torch.save(engine.model.state_dict(),
    #            args.save + "_exp" + str(args.expsid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
    # 修改命名
    torch.save(engine.model.state_dict(),
               args.save + "best_" + str(round(his_loss[bestid], 2)) + ".pth")
    print("Evaluating finished")


if __name__ == "__main__":
    with open('log_metr.txt', 'w', buffering=1) as f:
        """
        buffering = 1：行缓冲。即遇到换行符（print默认结尾是换行符），则立即将内存缓存区内的数据写入日志。
        或者
        在每一个print内添加flush = True参数，用于刷新缓存区。
        
        buffering = 1，行缓冲，可以确保日志数据几乎立即写入到了文件中，防止丢失，但可能增加I/O操作降性能。是buffering = 0，不使用缓存，立即写入文件的折中方法。
        """
        with redirect_stdout(f):
            t1 = time.time()
            main()
            t2 = time.time()
            print("Total time spent: {:.4f}".format(t2 - t1))
