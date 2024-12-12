import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
from torch import autograd


class Net(nn.Module):
    def __init__(self, inport_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inport_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            # 多项式f(x)用线性变化，用x即可
        )

    def forward(self, x):
        return self.net(x)


device = torch.device('cuda')
net = Net(2, 100, 1)
net.to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error 均方误差求
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器


def initial_fun(x):
    a = torch.cos((torch.pi) * x) * (x ** 2)
    a = autograd.Variable(a.float(), requires_grad=True).to(device)
    return a

def find_min_subset_with_indices(nums_tensor,tl):
    # 确保输入是一个torch tensor，如果不是，转换为tensor
    # 获取数值的总和
    nums_tensor=nums_tensor.t()
    nums_tensor1=nums_tensor[2,:]
    total_sum = nums_tensor1.sum().item()
    # 计算10%的总和
    target = total_sum * tl
    # 获取数值和索引的元组
    indexed_nums1 = [(num, idx) for idx, num in enumerate(nums_tensor1)]
    # 按数值降序排序
    indexed_nums_sorted = sorted(indexed_nums1, key=lambda x: x[0], reverse=True)
    current_sum = 0  # 当前和
    selected_indices = []  # 存放选中的标号
    # 从排序后的列表中选取元素直到总和超过目标
    for num, idx in indexed_nums_sorted:
        current_sum += num.item()  # 累加当前的数值
        selected_indices.append(nums_tensor[ :2,idx])
        if current_sum > target:  # 当和超过目标时停止
            break
    return torch.stack(selected_indices)

# x,t的输入都需要为(n,1)
def pde(x, t):
    combin = torch.cat((x, t), dim=1)
    lenx = len(x)
    lent = len(t)
    u = net(combin)
    du = autograd.grad(u, combin, grad_outputs=torch.ones_like(net(combin)), create_graph=True)[
        0]  # grad_outputs=torch.ones_like(net(combined_input))：这里的 grad_outputs 设置为 torch.ones_like(net(combined_input))，表示对于每个输出，
    # 初始的梯度为 1（即每个输出的偏导数为 1）。通常你会在多输出的情况下使用这个选项，在单输出时可以省略。
    u_x = du[:, 0].view(lenx, 1)
    u_t = du[:, 1].view(lent, 1)
    u_xx = autograd.grad(u_x, combin, grad_outputs=torch.ones_like((net(combin))), create_graph=True)[0]
    u_xx = u_xx[:, 1].view(lenx, 1)
    rate = 0.0001
    return u_t - rate * u_xx + 5 * (u ** 3 - u)

#循环需要的参数
losses = []
iterations = 10000
Nmax = 15
judge=0
tl=0.4
tols=0
judge_aix=0


for i in range(10):
    nr = 500
    ni = 128
    nb = 42
    dt = 0.1


    rx = np.random.uniform(low=-1, high=1, size=(nr, 1))
    rx = autograd.Variable(torch.from_numpy(rx).float(), requires_grad=True)
    rt = np.random.uniform(low=i * dt, high=(i + 1) * dt, size=(nr, 1))
    rt = autograd.Variable(torch.from_numpy(rt).float(), requires_grad=True)
    rx = rx.to(device)
    rt = rt.to(device)

    ix = torch.linspace(-1, 1, steps=ni).view(ni, 1)
    ix = autograd.Variable(ix.float(), requires_grad=True).to(device)
    it = i * dt * np.ones([ni, 1], dtype=np.float32)
    it = autograd.Variable(torch.from_numpy(it).float(), requires_grad=True).to(device)

    bx = (-1) * np.ones([nb, 1], dtype=np.float32)
    bx = autograd.Variable(torch.from_numpy(bx).float(), requires_grad=True).to(device)
    bx1 = np.ones([nb, 1], dtype=np.float32)
    bx1 = autograd.Variable(torch.from_numpy(bx1).float(), requires_grad=True).to(device)
    bt = np.random.uniform(low=i * dt, high=(i + 1) * dt, size=(nb, 1))
    bt = autograd.Variable(torch.from_numpy(bt).float(), requires_grad=True).to(device)


    for epoch in range(iterations):

        # define function loss
        pde_result = pde(rx, rt)
        reference_result = torch.zeros(nr, 1).to(device)
        lossr = mse_cost_function(pde_result, reference_result)

        # define initial loss

        if i == 0:
            initial_rel_result = initial_fun(ix)
        else:
            initial_rel_result = net1(torch.cat((ix, it), dim=1))
        initial_rel_result=autograd.Variable(initial_rel_result.float(), requires_grad=True).to(device)
        initial_net_result = net(torch.cat((ix, it), dim=1))
        lossi = mse_cost_function(initial_rel_result, initial_net_result)

        # define boundary loss


        bu = net(torch.cat((bx, bt), dim=1))
        bu1 = net(torch.cat((bx1, bt), dim=1))  # compute boundary result

        dbx = autograd.grad(bu, bx, grad_outputs=torch.ones_like((net(torch.cat((bx1, bt), dim=1)))), create_graph=True)[0]
        dbx1 = autograd.grad(bu1, bx1, grad_outputs=torch.ones_like((net(torch.cat((bx1, bt), dim=1)))), create_graph=True)[0]

        dbu = net(torch.cat((dbx, bt), dim=1))
        dbu1 = net(torch.cat((dbx1, bt), dim=1))

        bu_dbu = torch.cat((bu, dbu), dim=0)
        bu1_dbu1 = torch.cat((bu1, dbu1), dim=0)
        lossb = mse_cost_function(bu_dbu, bu1_dbu1)

        # 总loss
        loss = lossb +lossi*100+lossr
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # 梯度归0
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(epoch, "Traning Loss:", loss.data)
            if epoch !=0:
                lk=losses[-1]-losses[-2]
                lk1=losses[-2]-losses[-3]
                lk2=losses[-3] - losses[-4]
                Lk=(max(abs(lk),abs(lk1),abs(lk2)))
                print("Lk is :",Lk,"num nr is:",nr,"num ni is:",ni)
                judge=1 if Lk>=tols else 0


        #必须这么写，因为初始条件随机i的变换在变，因为时间适应性的存在
        if judge_aix==0:
            ani = round(ni * 0.2)  # 需要添加的大小
            aix = np.random.uniform(low=-1, high=1, size=(ani, 1))
            aix = autograd.Variable(torch.from_numpy(aix).float(), requires_grad=True).to(device)
            ait = i * dt * np.ones([ani, 1], dtype=np.float32)
            ait = autograd.Variable(torch.from_numpy(ait).float(), requires_grad=True).to(device)
            if i==0:
                a_initial_rel_result = initial_fun(aix)
            else:
                a_initial_rel_result=net(aix,ait)
            judge_aix=1

        if judge==1 :
            # 需要添加 rx rt ix
            anr = round(nr * 0.2)

            # 初始
            arx = np.random.uniform(low=-1, high=1, size=(anr, 1))
            arx = autograd.Variable(torch.from_numpy(arx).float(), requires_grad=True).to(device)
            art = np.random.uniform(low=i * dt, high=(i + 1) * dt, size=(anr, 1))
            art = autograd.Variable(torch.from_numpy(art).float(), requires_grad=True).to(device)

            a_lossr = torch.abs(pde(arx, art))
            a_lossr = autograd.Variable(a_lossr.float(), requires_grad=True).to(device)
            a_lossr = torch.cat((arx, art, a_lossr), dim=1)
            ar_x_t = find_min_subset_with_indices(a_lossr,tl)


            rx = torch.cat((rx, ar_x_t[:,0].view(-1,1)), dim=0)
            rx = autograd.Variable(rx.float(), requires_grad=True).to(device)
            rt = torch.cat((rt, ar_x_t[:,1].view(-1,1)), dim=0)
            rt = autograd.Variable(rt.float(), requires_grad=True).to(device)

            #INITIAL aix

            a_initial_net_result = net(torch.cat((aix, ait), dim=1))
            a_lossi = (a_initial_rel_result-a_initial_net_result)**2
            a_lossi = torch.cat((aix, ait, a_lossi), dim=1)
            a_lossi = autograd.Variable(a_lossi.float(), requires_grad=True).to(device)
            ai_x_t = find_min_subset_with_indices(a_lossi,tl)


            ix = torch.cat((ix, ai_x_t[:,0].view(-1,1)), dim=0)
            ix = autograd.Variable(ix.float(), requires_grad=True).to(device)
            it = torch.cat((it, ai_x_t[:,1].view(-1,1)), dim=0)
            ix = autograd.Variable(ix.float(), requires_grad=True).to(device)
            ni=len(ix)

            nr = len(rx)
            judge_aix == 0
            judge=0

    net1 = Net(2, 100, 1).to(device)
    torch.save(net.state_dict(), 'net_checkpoint.pth')
    net1.load_state_dict(torch.load('net_checkpoint.pth',weights_only=True))

    x = torch.linspace(-1, 1, steps=200).view(200, 1).to(device)
    t = torch.ones_like(x).to(device)
    t = t * (i + 1) * dt
    u = net(torch.cat((x, t), dim=1))
    plt.figure(i + 1)
    plt.plot(x.cpu().detach().numpy(), u.cpu().detach().numpy())
plt.show()

a=net()
