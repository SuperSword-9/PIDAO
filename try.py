import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义目标函数（梯度计算）
class ObjectiveFunction:
    def get_grad(self, x):
        return 2 * (x - 1)  # 这里用二次函数 (x-1)^2，梯度是 2(x-1)

# 定义分布式 PID 优化器
def PIDoptimizer(t, X, L, kp, ki, kd, momentum, f):
    delta = int(len(X) / 3)
    x, y, z = X[:delta], X[delta:2 * delta], X[2 * delta:]

    grad = f.get_grad(x)
    
    dx = -L @ x + y - kd * grad  # 加入分布式通信项 -L @ x
    dy = -momentum * y - (kp - momentum * kd) * grad - ki * z
    dz = grad  # 积分项

    return np.concatenate((dx, dy, dz))

# 定义带有 L · x 的 Momentum 优化器
def MomentumLxOptimizer(t, X, L, alpha, beta, f):
    delta = len(X) // 2
    x, v = X[:delta], X[delta:]

    grad = f.get_grad(x)
    
    dx = v - L @ x # 添加 L · x 以保持一致性
    dv = -alpha * grad - beta * v 
    
    return np.concatenate((dx, dv))

# 生成 10 个节点的连通图拉普拉斯矩阵 L
def generate_laplacian(n):
    A = np.random.randint(0, 2, (n, n))  # 生成随机邻接矩阵
    np.fill_diagonal(A, 0)  # 无自环
    A = np.maximum(A, A.T)  # 确保对称性
    D = np.diag(A.sum(axis=1))  # 度矩阵
    L = D - A  # 计算拉普拉斯矩阵
    return L

# **参数初始化**
num_nodes = 10
L = generate_laplacian(num_nodes)
f = ObjectiveFunction()

# **检查拉普拉斯矩阵的最大特征值**
lambda_max = np.max(np.linalg.eigvals(L))
print("L 的最大特征值:", lambda_max)

# **调试参数**
lr = 0.1  # 学习率
equivalent_momentum = 0.9
momentum = (1 / equivalent_momentum - 1) / lr  # 计算 momentum
kp, ki, kd = 0.5, 0.05, 0.05  # 调整 PID 参数，避免过大增益
alpha, beta = 0.5, 0.3  # Momentum_Lx 参数

# **初始条件**
X0_PID = np.random.rand(3 * num_nodes)  # 随机初始化 x, y, z
X0_Momentum = np.random.rand(2 * num_nodes)  # 随机初始化 x, v

t_span = (0, 30)  # 仿真 30 秒
t_eval = np.linspace(0, 30, 1000)  # 细化时间点

# **求解微分方程**
sol_PID = solve_ivp(PIDoptimizer, t_span, X0_PID, args=(L, kp, ki, kd, momentum, f), t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12)
sol_Momentum = solve_ivp(MomentumLxOptimizer, t_span, X0_Momentum, args=(L, alpha, beta, f), t_eval=t_eval, method='BDF', rtol=1e-10, atol=1e-12)

# **提取解**
x_sol_PID = sol_PID.y[:num_nodes, :]  # 提取 PID x 部分
x_sol_Momentum = sol_Momentum.y[:num_nodes, :]  # 提取 Momentum_Lx x 部分

# **检查数值范围**
print("PID x 部分的最小值：", np.min(x_sol_PID))
print("PID x 部分的最大值：", np.max(x_sol_PID))
print("Momentum_Lx x 部分的最小值：", np.min(x_sol_Momentum))
print("Momentum_Lx x 部分的最大值：", np.max(x_sol_Momentum))

# **绘图**
plt.figure(figsize=(10, 6))
for i in range(num_nodes):
    plt.plot(sol_PID.t, x_sol_PID[i, :], '--', label=f'PID Node {i}')
    plt.plot(sol_Momentum.t, x_sol_Momentum[i, :], label=f'Momentum_Lx Node {i}')
plt.xlabel("Time")
plt.ylabel("State x")
plt.title("PID vs Momentum")
plt.legend(fontsize=6, loc='lower right')
plt.show()
