import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import time
# 记录开始时间
start_time = time.time()

def ode_1(x,t):
    u=torch.sin((torch.pi)*x)*torch.exp(-1*((torch.pi)**2)*t)
    return u


class Net(nn.Module):
    def __init__(self, NL, NN):  # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数，
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN, NN )  ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优。更多情况有待我实验验证。
        self.hidden_layer2= nn.Linear(NN,  NN )
        self.output_layer = nn.Linear(NN , 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out_final = self.output_layer(out)
        return out_final

device = torch.device('cpu')
net = Net(4, 100).to(device)  # 4层 20个
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error 均方误差求
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 优化器



def pde_2(x,t):
    combined_input = torch.cat((x, t), dim=1).to(device)
    u=net(combined_input)
    u_a = autograd.grad(u, combined_input, grad_outputs=torch.ones_like(net(combined_input)), create_graph=True)[0]

    u_t = u_a[:, 1].view(2000, 1).to(device)
    u_x = u_a[:, 0].view(2000, 1).to(device)
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    u_xx = u_xx.view(2000, 1)
    return u_t-u_xx

def initial_ful(x):
    return torch.sin((torch.pi)*x)


iterations=5000
losses = []
for epoch in range(iterations):
    optimizer.zero_grad()  # 梯度归0


    x_in = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True).to(device)
    z_in = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    pt_z_in = autograd.Variable(torch.from_numpy(z_in).float(), requires_grad=True).to(device)

    # 初始损失
    t_0 = torch.zeros(2000, 1, dtype=torch.float32).to(device)
    u_r1=initial_ful(pt_x_in).to(device)
    combined = torch.cat((pt_x_in, t_0), dim=1).to(device)
    u_p1=net(combined).to(device)
    mse_0=mse_cost_function(u_p1,u_r1).to(device)


    #方程损失
    u_p2=pde_2(pt_x_in,pt_z_in).to(device)
    u_r2 = torch.zeros(2000, 1, dtype=torch.float32).to(device)
    mse_f=mse_cost_function(u_p2,u_r2).to(device)

    #边界损失
    t_in = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))
    pt_t_in = autograd.Variable(torch.from_numpy(z_in).float(), requires_grad=True).to(device)
    x_b1= torch.ones(2000, 1, dtype=torch.float32).to(device)
    x_b2= torch.zeros(2000, 1, dtype=torch.float32).to(device)
    combined1 = torch.cat((x_b1, pt_t_in), dim=1).to(device)
    combined2 = torch.cat((x_b2, pt_t_in), dim=1).to(device)
    mse_b_1 = net(combined1).to(device)
    mse_b_2 = net(combined2).to(device)
    mse_b=mse_cost_function(mse_b_1,u_r2)+mse_cost_function(mse_b_2,u_r2)

    #总损失
    loss=mse_f+mse_0*5+mse_b

    #反向传播
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch%1000==0:
        print(epoch, "Traning Loss:", loss.data)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"执行时间：{elapsed_time} 秒")


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
x_p = torch.linspace(0, 1, 50).view(50,1)
t_p = torch.linspace(0, 1, 50).view(50,1)
combined3 = torch.cat((x_p, t_p), dim=1).to(device)
pred=net(combined3)
x_p = x_p.numpy()  # 转换为 numpy 数组
t_p = t_p.numpy()  # 转换为 numpy 数组
pred = pred.detach().numpy()  # 转换为 numpy 数组
X, Y = np.meshgrid(x_p, t_p)     # 创建网格
# 绘制三维表面图
ax.plot_surface(X, Y,pred , cmap='viridis')
# 设置标题和标签
ax.set_title("Surface plot of pre_f_p")
ax.set_xlabel('X axis')
ax.set_ylabel('t axis')
ax.set_zlabel('pre_f_p axis')

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
x_p = torch.linspace(0, 1, 50).view(50,1)
t_p = torch.linspace(0, 1, 50).view(50,1)
pred=ode_1(x_p,t_p)
x_p = x_p.numpy()  # 转换为 numpy 数组
t_p = t_p.numpy()  # 转换为 numpy 数组
pred = pred.detach().numpy()  # 转换为 numpy 数组
X, Y = np.meshgrid(x_p, t_p)     # 创建网格
# 绘制三维表面图
ax.plot_surface(X, Y,pred , cmap='viridis')
# 设置标题和标签
ax.set_title("Surface plot of rel_f_p")
ax.set_xlabel('X axis')
ax.set_ylabel('t axis')
ax.set_zlabel('rel_f_p axis')

plt.figure(3)
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()



