# 消融实验的网络模型

import torch
import torch.nn as nn
from lib.pointops.functions import pointops

# 添加圆柱坐标转换函数
# [新增] 拓扑坐标转换函数
# [修改] 优化后的拓扑坐标转换函数
def get_cylindrical_features(xyz):
    """
    输入: xyz (N, 3) -> [x, y, z]
    输出: topo (N, 4) -> [r, sin_theta, cos_theta, z]
    使用 sin/cos 代替 theta 以解决 -pi/pi 处的周期性不连续问题
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = torch.sqrt(x ** 2 + y ** 2)  # 径向距离 r
    theta = torch.atan2(y, x)  # 周向角度

    # 返回4维特征，比单纯的theta更稳定
    return torch.stack([r, torch.sin(theta), torch.cos(theta), z], dim=1)


# 实现 T-LGAF (GraphAttentionLayer) - 包含 Curvature-Aware & Radial Encoding
# [修改] 符合老师要求的 LGAF: 显式建模 Normal, Theta, Z, R 四项差异


class GraphAttentionLayer(nn.Module):

    # 作为初学者，你可以把 __init__ 函数看作是 “装修清单”。
    # 在这里，我们不干活（不处理数据），只是把干活需要的工具（各种网络层）都买好、定义好。

    # 这段 __init__ 代码就像是搭建了一个精密的零件加工厂：
    # 原料：输入的点云特征。
    # Q/K/V 车间：把原料加工成查询、键、值三种形态。
    # 拓扑车间 (linear_explicit)：这是你的各种新武器。专门处理老师要求的法向量、角度、Z轴、半径这 4 个物理量，把它们变成机器能理解的高维特征。
    # 打分车间 (linear_w + softmax)：综合以上信息，给每个邻居打分，决定谁重要。
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, block_idx=0):
        super().__init__()

# nn.Module: 这是 PyTorch 中所有神经网络层的“基类”。继承它，PyTorch 才知道这是一个可以训练的网络。
        # 参数解释:
        # in_planes: 输入有多少种特征。比如上一层传过来每个点有 32 个特征。
        # out_planes: 输出要变成多少种特征。比如这一层处理完，每个点变成 64 个特征。
        # nsample: 邻居数。处理点云时，我们是一个点一个点处理的。这个参数决定了每个点“看”周围多少个邻居。
        # 保存传入的参数，供后面使用
        self.mid_planes = mid_planes = out_planes // 1  # 中间层维度，这里等于输出维度
        self.out_planes = out_planes # 输出特征的维度（比如 64, 128）
        self.share_planes = share_planes # 缩放因子，用于减少参数量（默认8）
        self.nsample = nsample # 邻居数量 (K-NN 的 K)，比如只看最近的 16 个点

# 定义“注意力三剑客” (Q, K, V)
        # 这是标准 Attention（注意力机制）的核心组件。
        # Q (Query, 查询): 中心点发出信号：“我要找什么样的邻居？”
        # K (Key, 键): 邻居发出信号：“我是什么类型的特征。”
        # V (Value, 值): 邻居实际包含的内容：“这是我要传给你的信息。”
        # 代码含义: nn.Linear 是全连接层（线性变换）。
        # 它把输入的特征（in_planes）通过矩阵乘法，分别变成了 Q、K、V 三种不同的特征表示。
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

# 定义“显式拓扑编码” (核心修改)
        # 对应老师公式: ai = MLP(dn, dtheta, dz, dr)
        # 输入维度为 4: [diff_n(1), diff_theta(1), diff_z(1), diff_r(1)]
        # diff_n: 法向量差异 (Normal difference)，代表表面弯曲程度。
        # diff_theta: 角度差异 (周向)。
        # diff_z: 轴向差异 (沿管道长度方向)。
        # diff_r: 径向差异 (离圆心距离的变化)。


        # nn.Sequential: 一个容器，把里面的层按顺序串起来。数据进去后，会依次经过这几层。
        # 流程解释:
        # nn.Linear(4, 16): 把这 4 个物理量，升维映射到 16 维，提取初步特征。
        # nn.BatchNorm1d(16): 归一化。把数据拉回到标准的正态分布，防止训练跑偏，能让模型学得更快。
        # nn.ReLU(inplace=True): 激活函数。把负数变成 0。没有它，网络就只是线性变换，不够聪明；加了它，网络才能拟合复杂的曲线。
        # nn.Linear(16, out_planes): 再次升维。把特征维度变成 out_planes，这样它的大小就和上面的 Q、K、V 一样了，方便后面把它们加在一起。
        self.linear_explicit = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, out_planes)
        )

# 定义“权重计算器” (MLP)
        # 这个模块用来计算最终的“关注度分数”。

        # 作用: 它接收融合后的特征（Q - K + 拓扑特征），经过几层计算，输出一个数值。这个数值代表**“这个邻居对中心点有多重要”**。
        # 为什么要除以 share_planes?
        # 比如特征有 64 维，如果直接算，参数量太大。
        # 这里先把它压缩到 64 // 8 = 8 维，再计算。这是一种瓶颈结构 (Bottleneck)，既能节省显存，又能防止过拟合。
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))

        # 定义 Softmax (归一化)
        # 作用: 把上面算出来的“重要性分数”转换成概率。
        # 举例: 假设一个点有 3 个邻居，分数分别是 [2.0, 1.0, 0.1]。经过 Softmax 后，可能变成 [0.7, 0.2, 0.1]。
        # 这意味着：第一个邻居贡献 70% 的信息，第二个 20%，第三个 10%。它们的和永远是 1。
        self.softmax = nn.Softmax(dim=1)

# 这段代码是 Graph Attention Layer (图注意力层) 的 forward 函数。
# 如果说 __init__ 是在“备料”，那么 forward 就是真正的 “烹饪过程”。

# 这段代码做了一件事： 在提取点云特征时，强制网络去“看”管道的几何结构（半径、角度、轴向、法向）。
# 这使得你的网络对下水道特有的缺陷（如裂缝、变形）比普通网络更敏感。

    # 它的核心任务是：对于管道上的每一个点，找到它周围的邻居，结合“几何特征”和“拓扑物理特征”（你的论文创新点），计算出新的特征表示。
    def forward(self, pnxo) -> torch.Tensor:
        # p (N, 3): 所有点的 XYZ 坐标
        # n (N, 3): 所有点的法向量
        # x (N, C): 所有点的特征向量（比如上一层输出的 32 维特征）
        # o (B): Offset，用于标记不同点云样本的边界
        p, n, x, o = pnxo

# 准备食材与寻找邻居
        # 1. 基础特征
        # Query (Q): 中心点发出的信号，代表“我在找什么样的邻居”。
        x_q = self.linear_q(x) # 把特征 x 变成 Query (N, mid_planes)

        # x_edge: [relative_xyz, feats]
        # 这一行是 GNN 的核心：KNN 找邻居

        # pointops.queryandgroup: 这是一个 CUDA 加速函数。
        # 作用: 对于每一个点（中心点），在 XYZ 空间中找到最近的 nsample (16) 个邻居。
        # 参数 edge=True: 表示返回的是 差值。即 邻居特征 - 中心点特征。
        # 返回 x_edge: 形状是 (N, 16, 3+C)。前 3 个数是相对坐标，后 C 个数是特征差值。
        # (展示中心点如何连接周围的 K 个点)
        x_edge, distance = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, edge=True)
        _, x_e = x_edge[:, :, 0:3], x_edge[:, :, 3:] # 把坐标差扔掉(_)，只保留特征差(x_e)

        # Key (K): 邻居的标签，“我是什么”。
        # Value (V): 邻居的内容，“我要传给你什么信息”。
        x_k = self.linear_k(x_e) # 把邻居特征变成 Key (N, 16, mid_planes)
        x_v = self.linear_v(x_e) # 把邻居特征变成 Value (N, 16, out_planes)

# 计算显式物理特征差异 (论文核心创新)
        # 2. 计算显式物理特征差异 (Explicit Modeling)
        # 代码在这里手动计算了坐标，而不是直接用 get_cylindrical_features，这是为了确保计算差值时的精确性。
        # A. 拓扑坐标 (r, sin, cos, z) -> 我们需要还原出 r, theta, z
        cyl = get_cylindrical_features(p)  # (N, 4) -> [r, sin, cos, z]

        # 这里为了计算差异，我们重新构建简化的 (r, theta, z) 用于相减
        # 注意: get_cylindrical_features 返回的是 [r, sin, cos, z]
        # 我们手动用 atan2 恢复一下 theta 用于计算差值 (或者直接在 get 函数里改，但为了不破坏之前的结构，在这里处理)
        # 简单做法：我们直接利用 get_cylindrical_features 返回的 4 维特征计算 "拓扑距离"
        # 但老师公式是分开的，所以我们需要分别提取

        # 重新计算纯粹的 (r, theta, z) 用于物理差异计算
        p_r = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2).unsqueeze(1)  # (N, 1) -> 半径 r
        p_theta = torch.atan2(p[:, 1], p[:, 0]).unsqueeze(1)  # (N, 1) -> 角度 theta
        p_z = p[:, 2].unsqueeze(1)  # (N, 1) -> 轴向 z
        p_phys = torch.cat([p_r, p_theta, p_z], dim=1)  # (N, 3) -> 拼成新的物理坐标

        # Grouping 物理坐标
        # phys_edge: (neighbor - center) -> [dr, dtheta, dz]
        # 再次调用 queryandgroup。这次我们不是找特征，而是找邻居的物理坐标，并计算它们与中心点的差值。
        # phys_edge: 形状 (N, 16, 3)。包含了每个邻居的 $[\Delta r, \Delta \theta, \Delta z]$。
        phys_edge, _ = pointops.queryandgroup(self.nsample, p, p, p_phys, None, o, o, use_xyz=False, edge=True)

        dr = torch.abs(phys_edge[..., 0:1])  # |ri - rj| -> 径向差异（是否凹陷/凸起）

        # 处理角度周期性 |theta_i - theta_j|
        # 为什么 dtheta 这么复杂？因为角度是圆的。
        # $359^\circ$ 和 $1^\circ$ 的数学差值是 $358$，但实际物理距离只有 $2$。
        # 这段代码用 remainder (取余) 解决了这个问题，确保计算的是最短弧长距离。
        dtheta_raw = phys_edge[..., 1:2]
        dtheta = torch.abs(torch.remainder(dtheta_raw + torch.pi, 2 * torch.pi) - torch.pi)

        dz = torch.abs(phys_edge[..., 2:3])  # |zi - zj| -> 轴向差异

        # B. 法向量差异 (Normal Diff)
        # n_edge 是 (neighbor_n - center_n)
        n_edge, _ = pointops.queryandgroup(self.nsample, p, p, n, None, o, o, use_xyz=False, edge=True)
        # dn: 计算法向量的差异模长。
        # 如果表面很平滑，法向量差异小；如果表面有裂缝或坑洼，法向量差异会很大。
        dn = torch.norm(n_edge, dim=-1, keepdim=True)  # |ni - nj| 法向量差异


# 特征融合与映射
# 现在我们手头有 4 个物理量：[dn, dtheta, dz, dr]。它们只是 4 个数字，通过 MLP 把它们变成高维特征，让神经网络能理解。

        # 3. 组合四项特征 [dn, dtheta, dz, dr]
        explicit_feat = torch.cat([dn, dtheta, dz, dr], dim=-1)  # (N, k, 4)

        # 通过 MLP 映射到 Attention 空间
        # explicit_emb: (N, nsample, out_planes)
        # linear_explicit: 这是你在 __init__ 里定义的 MLP。
        # 作用: 把 4 个物理数字，扩展成比如 32 维或 64 维的向量 (explicit_emb)，这样它的大小就和 Q、K 一样了，可以进行加减运算。
        explicit_emb = self.linear_explicit(explicit_feat.reshape(-1, 4)).reshape(x_q.shape[0], self.nsample, -1)

# 计算 Attention 并聚合
# 最后，根据物理特征和内容特征，决定“谁是重要的邻居”，并把信息汇总。
        # 4. 计算 Attention
        # w = Q - K + Explicit_Physics_Bias
        # x_q - x_k: 内容相似度（传统的 Attention）。
        # + explicit_emb: 物理结构修正（你的创新）。
        # 意思就是：如果两个点物理结构关系紧密（比如都在裂缝线上），即使它们原来的特征不太像，我也要关注它。
        w = x_q.unsqueeze(1) - x_k + explicit_emb

        # 通过 linear_w 网络处理权重
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w) # 归一化，变成概率（0~1）

        distance_weight = torch.exp(-distance.unsqueeze(-1)) # 距离衰减（离得越远权重越小）
        x_v = x_v * distance_weight

        num, nSample, c = x_v.shape
        s = self.share_planes

        # 聚合时也加上显式特征，增强表达
        # Output = sum( (Value + Physical_Info) * Attention_Weight )
        x1 = ((x_v + explicit_emb).view(num, nSample, s, c // s) * w.unsqueeze(2)).sum(1).view(num, c)

        # sum(1): 把 16 个邻居的信息加权求和，聚合成 1 个中心点的新特征。
        # x1: 输出结果，形状 (N, out_planes)。
        return x1

# 这是一个名为 EdgeConv 的类，它实现了边缘卷积 (Edge Convolution) 操作，这是在点云处理中非常经典且常用的模块（源自 DGCNN 论文）。
# 它的核心思想是：通过把点和它的邻居连接起来（构成边），来学习局部几何特征。

# EdgeConv 就像是一个**“局部特征提取器”**。
# 它查看每个点和它邻居的关系，把这些关系编码成新的特征，是处理点云局部几何结构（比如边角、平面、曲率）的神器。
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, nsample):
        super().__init__()
        self.nsample = nsample # 邻居数量 (K)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 激活函数
        # 为什么输入是 in_channels * 2？因为 EdgeConv 的核心操作是把 中心点特征 ($x_i$) 和 邻居特征差异 ($x_j - x_i$) 拼在一起。
        # 如果输入特征维度是 C，拼接后就变成了 2C。
        # nn.Conv2d: 这里虽然叫卷积，但 kernel_size=1，实际上就是一个对每个点独立进行的 MLP（多层感知机）。
        # 它负责把拼接后的特征融合起来。
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), self.relu)

        # 这是第二层卷积，进一步提取特征，增加网络的深度和非线性能力。
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), self.relu)
        # 这是一个小的 MLP，专门处理 XYZ 相对坐标。
        # 它的作用是把几何位置信息映射到高维特征空间，最后加到卷积出来的特征里去，增强对形状的感知。
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_channels))

# 前向传播#
    def forward(self, pnxo):
        # 解包输入数据：坐标(p)、法向(n)、特征(x)、偏移(o)。
        p, n, x, o = pnxo
        # 找邻居 (KNN)
        # edge=True: 这里非常关键。
        # queryandgroup 返回的 x_edge 包含了 相对坐标 $(x_j - x_i)$ 和 特征差值 $(f_j - f_i)$。
        # 结果 x_edge 形状：(N, nsample, 3+C)。
        x_edge, _ = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, edge=True)

        # 分离特征与位置
        # p_r: 相对坐标差值 (N, 16, 3)。
        # x_e: 特征差值 (N, 16, C)。
        p_r, x_e = x_edge[:, :, 0:3], x_edge[:, :, 3:]
        # 处理位置编码
        # 把相对坐标通过 linear_p 映射成高维位置特征。
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)  # (num,nsample,o_c)
        # 特征拼接 (EdgeConv 核心)
        # x.unsqueeze...: 把中心点特征复制 16 份，形状变成 (N, 16, C)。
        # x_e: 邻居的特征差值，形状也是 (N, 16, C)。
        # torch.cat: 把它们拼在一起，变成了 (N, 16, 2C)。
        # 意义: 既保留了全局信息（中心点特征），又引入了局部关系（特征差值）。
        x = torch.cat((x.unsqueeze(1).repeat(1, self.nsample, 1), x_e), dim=-1)  # (num, nsample, in_planes*2)

        # 卷积与聚合
        x = x.permute(2, 0, 1).unsqueeze(0)  # (num, nsample, in_planes*2) -> (1, in_planes*2, num, nsample)
        # 第一层 MLP
        x = self.conv1(x)  # (1, in_planes*2, num, nsample) -> (1, out_channels, num, nsample)
        # 第二层 MLP + 位置编码
        x = self.conv2(x) + p_r.permute(2, 0, 1).unsqueeze(0)  # (1, out_channels, num, nsample) -> (1, out_channels, num, nsample)

        # 把拼好的特征通过两个卷积层处理，并加上位置编码。
        # Max Pooling: x.max(dim=-1)。
        # 在 16 个邻居中取最大值（Max Pooling）。这意味着对于每个中心点，我们只保留它周围“最强”的特征。
        # 这一步把局部邻域的信息压缩成了一个点的特征。
        x = x.max(dim=-1, keepdim=False)[0].squeeze(0).transpose(0, 1).contiguous()  # (1, out_channels, num, nsample) -> (num, out_channels)

        return x


class AdaptiveConv(nn.Module):
    def __init__(self, feat_channels, out_channels, in_channels, nsample):
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.nsample = nsample

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(self.feat_channels, self.out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.out_channels), self.relu)
        self.conv2 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels * self.in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.out_channels * self.in_channels), self.relu)

    def forward(self, pnxo):
        p, n, x, o = pnxo
        p_x, _ = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)
        p_r, feat = p_x[:, :, 0:3].unsqueeze(0), p_x[:, :, 3:].permute(2, 0, 1).unsqueeze(0)  # (1, n, nsample, 3), (1, feat_channels, n, nsample)
        kernel = self.conv1(feat)  # (1, feat_channels, n, nsample) -> (1, out_channels, n, nsample)
        kernel = self.conv2(kernel)  # (1, out_channels, n, nsample) -> (1, out_channels * in_channels, n, nsample)



class MultiScale(nn.Module):
    def __init__(self, nsample, up_planes, out_planes, down_planes=None, share_planes=8, stride=4):
        super().__init__()
        self.nsample = nsample
        self.stride = stride
        self.share_planes = share_planes
        self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(up_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        self.linear_q1 = nn.Linear(out_planes, out_planes)
        self.linear_k2 = nn.Linear(out_planes, out_planes)
        self.linear_v2 = nn.Linear(out_planes, out_planes)
        self.linear_w2 = nn.Sequential(nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
                                       nn.Linear(out_planes, out_planes // share_planes),
                                       nn.BatchNorm1d(out_planes // share_planes), nn.ReLU(inplace=True),
                                       nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        if down_planes is not None:
            self.linear3 = nn.Sequential(nn.Linear(3 + down_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear_k3 = nn.Linear(out_planes, out_planes)
            self.linear_v3 = nn.Linear(out_planes, out_planes)
            self.linear_w3 = nn.Sequential(nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True),
                                           nn.Linear(out_planes, out_planes // share_planes),
                                           nn.BatchNorm1d(out_planes // share_planes), nn.ReLU(inplace=True),
                                           nn.Linear(out_planes // share_planes, out_planes // share_planes))
            self.pool = nn.MaxPool1d(nsample)

    def downsample(self, p1, pxo):
        p, x, o = pxo  # (4n, 3), (4n, c/2), (b)
        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i - 1].item()) // self.stride
            n_o.append(count)

        # torch.Tensor (PyTorch 的张量)
        n_o = torch.tensor(n_o, dtype=torch.int, device='cuda')
        # idx = pointops.furthestsampling(p, o, n_o)  # (m)
        # n_p = p[idx.long(), :]  # (m, 3)
        x, _ = pointops.queryandgroup(self.nsample, p, p1, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
        for i, layer in enumerate(self.linear3): x = layer(x).transpose(1, 2).contiguous() if i == 0 else layer(x)
        # x = self.relu(self.bn(self.linear_down(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
        x = self.pool(x).squeeze(-1)  # (m, c)
        return x

    def forward(self, pxo1, pxo2, pxo3=None):
        if pxo3 is None:
            p1, x1, o1 = pxo1  # x1:(n, c)
            p2, x2, o2 = pxo2  # x2:(n/4, 2c)
            x1 = self.linear1(x1)
            x2 = pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)  # (n, c)
            x_q, x_k, x_v = self.linear_q1(x1), self.linear_k2(x2), self.linear_v2(x2)
            x_k, _ = pointops.queryandgroup(self.nsample, p1, p1, x_k, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            x_v, _ = pointops.queryandgroup(self.nsample, p1, p1, x_v, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            w = x_k - x_q.unsqueeze(1)
            for i, layer in enumerate(self.linear_w2): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
            w = self.softmax(w)  # (n, nsample, c//share_planes)
            n, nsample, c = x_v.shape
            s = self.share_planes
            x2 = (x_v.view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
            feat = x1 + x2
        else:
            p1, x1, o1 = pxo1  # x1:(n, c)
            p2, x2, o2 = pxo2  # x2:(n/4, 2c)
            _, _, _ = pxo3  # x3:(4n, c/2)
            x1 = self.linear1(x1)
            x2 = pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)  # (n, c)
            x3 = self.downsample(p1, pxo3)  # (n, c)
            x_q = self.linear_q1(x1)
            x_k2, x_v2 = self.linear_k2(x2), self.linear_v2(x2)
            x_k3, x_v3 = self.linear_k3(x3), self.linear_v3(x3)
            x_k2, _ = pointops.queryandgroup(self.nsample, p1, p1, x_k2, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            x_v2, _ = pointops.queryandgroup(self.nsample, p1, p1, x_v2, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            x_k3, _ = pointops.queryandgroup(self.nsample, p1, p1, x_k3, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            x_v3, _ = pointops.queryandgroup(self.nsample, p1, p1, x_v3, None, o1, o1, use_xyz=False)  # (n, nsample, c)
            w2 = x_k2 - x_q.unsqueeze(1)
            w3 = x_k3 - x_q.unsqueeze(1)
            for i, layer in enumerate(self.linear_w2): w2 = layer(w2.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w2)
            for i, layer in enumerate(self.linear_w3): w3 = layer(w3.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w3)
            w2 = self.softmax(w2)  # (n, nsample, c//share_planes)
            w3 = self.softmax(w3)  # (n, nsample, c//share_planes)
            n, nsample, c = x_v2.shape
            s = self.share_planes
            x2 = (x_v2.view(n, nsample, s, c // s) * w2.unsqueeze(2)).sum(1).view(n, c)
            x3 = (x_v3.view(n, nsample, s, c // s) * w3.unsqueeze(2)).sum(1).view(n, c)
            feat = x1 + x2 + x3
        return [p1, feat, o1]


# 实现 Axial-Aware Pooling (TransitionDown)
# 沿管轴方向权重
# [修改] 符合老师要求的 Pooling: 融合 Curvature, Radial, Density

# 它的作用是 “预加载装备”。
# 在这个函数里，我们不处理具体的数据（那是 forward 函数的事），而是定义好这一层网络需要用到的所有组件（如线性层、池化层、以及你新加的权重网络）。
# 这一层对应你论文中的 SAG-Pooling (Spatial Attention Graph Pooling) 模块，负责在下采样（减少点数）的过程中，保留关键信息。
class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16, use_sag=True):
        # self: 类实例本身。
        # in_planes: 输入特征维度（例如 32）。
        # out_planes: 输出特征维度（例如 64）。下采样通常伴随着特征维度的增加。
        # stride: 步长/下采样倍率。
        # 如果 stride=4，表示点数要减少为原来的 1/4（这是主要情况）。
        # 如果 stride=1，表示不减少点数，只做特征变换。
        # nsample: 邻居数量（例如 16）。在下采样时，通过聚合周围 16 个点的信息来代表中心点。
        # use_sag: 开关。是否开启你论文里的“尺度/拓扑感知”功能。

        # TransitionDown 类的 __init__ 函数主要做了一件事： 根据是否需要下采样 (stride)，准备不同的“工具包”。
        # 如果要下采样，它准备了 线性层 + 池化层 + 你的权重网络 (weight_net)。
        # 如果不要下采样，它只准备一个 线性层。
        # 这个结构保证了网络可以在不同阶段灵活地调整点云的密度和特征维度。
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.use_sag = use_sag  # 保持接口兼容
# 分支判断：是否进行下采样？
        # 这是一个分叉路口。
        # 情况 A (stride != 1)：这是真正的下采样层（比如点数从 N 变 N/4）。我们需要复杂的池化操作。

        if stride != 1:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
            # A2. 定义权重网络 (你的创新点)
            # 权重计算 MLP: 输入 3 (Curvature, Radial, Density) -> 输出 1 (Weight)
            # 这是 SAG-Pooling 的核心。
            # 普通的 Pooling 是“盲目”的（只看数值最大）。你这里增加了一个**“打分器”**。
            # 输入 (3): 对应你设计的 3 个物理特征（曲率、径向距离、密度）。
            # 输出 (1): 输出一个 0 到 1 之间的分数。
            # Sigmoid(): 这是一个激活函数，它能把任何数字压缩到 (0, 1) 区间。
            # 接近 1: 说明这个点很重要（比如是裂缝点），保留！
            # 接近 0: 说明这个点不重要（比如是光滑管壁），抑制。

            # 权重计算 MLP: 输入 3 (Curvature, Radial, Density) -> 输出 1 (Weight)
            self.weight_net = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
                nn.Sigmoid()  # 输出 0-1 之间的权重
            )
        # 分支 B：不进行下采样
        # 情况 B (stride == 1)：如果步长是 1，说明点数不变。
        # 这时候只需要做一个简单的特征维度变换（线性层）即可，不需要池化，也不需要权重网络。
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        # 公共组件：归一化与激活
        # 不管走哪个分支，最后出来的特征都要经过这两个处理：
        # BatchNorm1d: 整理数据分布，让训练更稳定。
        # ReLU: 激活函数，增加非线性能力。
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    # 这是一个非常核心的 TransitionDown 模块（通常也叫 Downsample Layer）。
    # 它的作用是降低点的数量（下采样），同时增加特征的维度，类似于卷积神经网络中的 Stride=2 的卷积层。
    # 结合你之前提到的修改（SAG, Structure-Aware Grouping），
    # 这段代码在标准的 PointNet++ 下采样过程中，强行插入了一个基于物理几何特征（径向、密度、曲率）的注意力加权机制。

    # 这个函数是整个网络中承上启下的关键：
    # 它把点云变稀疏了（下采样）。
    # 它根据你设计的三个物理量（SAG），告诉网络哪些邻居点更重要（加权）。
    # 它把局部的一堆点融合成了一个特征向量（Max Pooling）。
    def forward(self, pnxo):
        # 函数签名与解包
        # 输入 pnxo：这是一个列表，包含四个核心数据。
        # p (Points): 点的坐标，通常形状是 (N, 3)，N 是这一层所有点的总数。
        # n (Normals): 点的法向量，形状 (N, 3)。
        # x (Features): 点的特征（比如颜色或上一层提取的深层特征），形状 (N, C)。
        # o (Offsets): 偏移量。这是一个针对点云批处理（Batch）的特殊设计。
        # 因为点云每个样本点数可能不同，这里把所有样本的点拼成一个超长的 (N, 3)，用 o 来记录每个样本在长数组中的结束位置。
        p, n, x, o = pnxo
        # 批处理偏移量计算（Batch Offset Calculation）
        # 逻辑：如果 stride 不为 1（通常是 4，表示点数减少 4 倍），我们需要计算下采样后每个样本不仅剩多少个点。
        # o[i].item() - o[i-1].item()：算出当前第 i 个样本原来有多少个点。
        # // self.stride：除以步长，算出下采样后该样本应该剩多少点。
        # n_o (New Offsets)：生成一个新的 offset 列表，用于告诉下一层每个样本的边界在哪里。
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.int, device='cuda')

            # FPS 采样中心点
            # FPS 采样与分组（Sampling & Grouping）
            # FPS 采样中心点
            # idx = ...：使用最远点采样 (FPS) 算法，选出具有代表性的关键点（中心点）。
            # n_p (New Points)：根据索引取出的新点坐标（即下采样后的点）。
            # n_n (New Normals)：对应的法向量。
            idx = pointops.furthestsampling(p, o, n_o)
            n_p, n_n = p[idx.long(), :], n[idx.long(), :]
            # queryandgroup：这是一个核心 CUDA 操作。
            # 它的作用是：对于每一个新选出的中心点 n_p，在原始点云 p 中寻找最近的 self.nsample 个邻居（KNN）。
            # 关键点：use_xyz=True。这意味着返回的 x_grouped 前 3 个通道是 相对坐标（邻居坐标 - 中心点坐标），后面接的是特征 x。
            # 形状：x_grouped 的形状通常是 (M, nsample, 3+C)，其中 M 是下采样后的点数。

            # Grouping
            x_grouped, _ = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            # x_grouped: [relative_xyz(3), feats(c)]

            # SAG 模块（Structure-Aware Grouping - 你的核心修改）
            # 这部分是你根据老师要求添加的“物理特征增强”。
            if self.use_sag:
                # --- 老师要求的三个特征计算 ---

                # 1. Radial Offset (径向偏移): r
                # 获取邻域点的真实坐标 (center + relative)

                # neighbor_xyz：x_grouped 前3维是相对坐标，加上中心点坐标 n_p，还原回绝对坐标。
                # torch.sqrt(...)：计算 $\sqrt{x^2 + y^2}$。
                # 物理含义：这是计算点到 Z 轴的水平距离（半径 $r$）。
                # 这在**圆柱体（如下水道管道）**场景中非常重要，因为它直接反映了点是在管道壁上，还是在管道中心。
                neighbor_xyz = x_grouped[..., 0:3] + n_p.unsqueeze(1)
                # 计算 r = sqrt(x^2 + y^2)
                radial_offset = torch.sqrt(neighbor_xyz[..., 0] ** 2 + neighbor_xyz[..., 1] ** 2).unsqueeze(
                    -1)  # (N, k, 1)

                # 2. Density (密度): 邻域点到中心的距离 (距离越小，密度越高)
                # relative_xyz 的模长
                # dist：计算邻域点到中心点的欧几里得距离。
                dist = torch.norm(x_grouped[..., 0:3], dim=-1, keepdim=True)
                # density：取倒数。距离越近，数值越大，模拟“密度”概念。
                # 作用：让网络更关注紧挨着中心点的那些特征，或者是区分稀疏/稠密区域（例如裂缝处的点云密度可能与平滑管壁不同）。
                # 简单模拟密度: 1 / (距离 + epsilon)
                # 简单模拟密度: 1 / (距离 + epsilon)
                density = 1.0 / (dist + 1e-6)

                # 3. Curvature (曲率): 近似为法向量的差异
                # 需要取邻居的法向量，这里需要重新 query 一次 n
                # n_grouped：这里再次调用了 queryandgroup，这次是为了把法向量 (n) 也聚合起来。
                n_grouped, _ = pointops.queryandgroup(self.nsample, p, n_p, n, None, o, n_o, use_xyz=False)
                # 计算邻居法向与中心法向的点积，点积越小(夹角越大)说明曲率越大
                # center_n: n_n (N, 3) -> (N, 1, 3)
                # dot_prod：计算 邻居法向量 与 中心点法向量 的点积。
                # 如果两个法向量平行（平坦表面），点积为 1，curvature 接近 0。
                # 如果两个法向量垂直（拐角、裂缝边缘），点积为 0，curvature 接近 1。
                # 作用：非常敏锐地捕捉表面凹凸不平的特征（如裂缝、腐蚀）。
                dot_prod = torch.matmul(n_grouped, n_n.unsqueeze(2))  # (N, k, 1)
                curvature = 1.0 - torch.abs(dot_prod)  # 1 - |cos_theta|

                # --- 计算权重 ---
                # 拼接 [Curvature, Radial, Density]
                # cat：把计算出的 3 个物理特征拼起来。
                # weight_net：这是一个小的 MLP（全连接层），把物理特征映射成一个权重值。
                physics_feat = torch.cat([curvature, radial_offset, density], dim=-1)  # (N, k, 3)
                weights = self.weight_net(physics_feat)  # (N, k, 1)

                # 加权特征
                # 只对特征部分加权，前3维是坐标
                # feats * (1 + weights)：这是一个残差注意力机制。
                # 如果权重是 0，特征保持原样。
                # 如果权重高，特征被放大。
                # 注意：这里你把 x 赋值为了 feats（仅包含特征通道），丢弃了坐标信息（这点我在上一个回复中提到过，可能会导致几何信息丢失）。
                feats = x_grouped[..., 3:]
                feats = feats * (1 + weights)  # 强化重要点 (Residual connection style)

                x = feats
    # 特征提取与聚合 (Feature Aggregation)
            else:
                x = x_grouped[..., 3:] # 如果不启用 SAG，直接取特征部分

            # self.linear(x)：通过全连接层，将特征维度升高（例如从 32 升到 64）。
            # x.transpose(1, 2)：PyTorch 的 BatchNorm 和 Conv1d 通常需要通道在第 1 维 (Batch, Channel, Length)，所以这里把特征维换到中间。
            # self.pool(x) (Max Pooling)：这是 PointNet 的精髓。
            # 输入形状：(N_new, Channel, nsample) (每个中心点有 nsample 个邻居)。
            # 输出形状：(N_new, Channel, 1)。
            # 含义：在每个局部邻域内，取特征最强的那个值，作为这个区域的代表特征。
            x = self.linear(x)
            x = self.relu(self.bn(x.transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, n, o = n_p, n_n, n_o
        else:
            # 如果 stride=1 (不下采样)，仅仅是一个 MLP 变换特征
            # 最后更新 p, n, o 为下采样后的新值，并将它们打包返回，供网络下一层使用。
            x = self.relu(self.bn(self.linear(x)))
        return [p, n, x, o]


class TransitionUp(nn.Module):

    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
            self.linear = nn.Linear(3 + in_planes // 2, in_planes, bias=False)
            self.bn = nn.BatchNorm1d(in_planes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear = nn.Linear(3 + out_planes // 2, out_planes, bias=False)
            self.bn = nn.BatchNorm1d(out_planes)
            self.relu = nn.ReLU(inplace=True)
        self.stride, self.nsample = 4, 8
        self.pool = nn.MaxPool1d(self.nsample)

    def downsample(self, p1, pnxo):
        p, n, x, o = pnxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            # n_o = torch.cuda.IntTensor(n_o)
            n_o = torch.tensor(n_o, dtype=torch.int, device='cuda')
            # idx = pointops.furthestsampling(p, o, n_o)  # (m)
            # n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.queryandgroup(self.nsample, p, p1, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            # p, o = n_p, n_o
            o = n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, n, x, o]

    def forward(self, pnxo1, pnxo2=None, pnxo3=None):
        if pnxo2 is None:
            p, n, x, o = pnxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
            if pnxo3 is not None:
                x = x + self.downsample(p, pnxo3)[2]
        else:
            p1, _, x1, o1 = pnxo1
            p2, _, x2, o2 = pnxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
            if pnxo3 is not None:
                x = x + self.downsample(p1, pnxo3)[2]
        return x


class GraphAttentionBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, block_idx=0):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = GraphAttentionLayer(planes, planes, share_planes, nsample, block_idx)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, pnxo):
        p, n, x, o = pnxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, n, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, n, x, o]


class GraphConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.edgeConv = EdgeConv(in_planes, out_planes, nsample)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.linear = nn.Linear(out_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pnxo):
        p, n, x, o = pnxo
        identity = x
        x = self.relu(self.bn1(self.edgeConv([p, n, x, o])))
        x = self.bn2(self.linear(x))
        x = self.relu(x + identity)
        return [p, n, x, o]


class GraphAttentionSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, use_lgaf=True, use_sag_pooling=True):
        super().__init__()
        self.c = c
        self.use_lgaf = use_lgaf  # 保存LGAF使用标志
        self.use_sag_pooling = use_sag_pooling  # 保存SAG-Pooling标志
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample, block_idx = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                   stride[0], nsample[0], block_idx[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                   stride[1], nsample[1], block_idx[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                   stride[2], nsample[2], block_idx[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes,
                                   stride[3], nsample[3], block_idx[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes,
                                   stride[4], nsample[4], block_idx[4])
        self.multi1 = MultiScale(nsample[0], planes[2], planes[1])
        self.multi2 = MultiScale(nsample[0], planes[3], planes[2], planes[1])
        self.multi3 = MultiScale(nsample[0], planes[4], planes[3], planes[2])
        self.multi4 = MultiScale(nsample[0], planes[2], planes[1])
        self.multi5 = MultiScale(nsample[0], planes[3], planes[2], planes[1])
        self.multi6 = MultiScale(nsample[0], planes[2], planes[1])
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True, block_idx=block_idx[5])  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], block_idx=block_idx[6])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2], block_idx=block_idx[7])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1], block_idx=block_idx[8])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0], block_idx=block_idx[9])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes,
                  stride, nsample, block_idx):
        """根据实验配置构建编码器层"""
        layers = []
        # 添加下采样层（使用正确的SAG配置）
        layers.append(
            TransitionDown(
                self.in_planes, planes, stride, nsample,
                use_sag=self.use_sag_pooling
            )
        )
        self.in_planes = planes

        # 添加特征提取块（根据LGAF配置）
        for _ in range(1, blocks):
            if self.use_lgaf:
                layers.append(
                    GraphAttentionBlock(
                        self.in_planes, self.in_planes,
                        share_planes, nsample, block_idx
                    )
                )
            else:
                layers.append(
                    GraphConvBlock(
                        self.in_planes, self.in_planes, nsample
                    )
                )
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=4, nsample=16, is_head=False, block_idx=0):
        layers = [TransitionUp(self.in_planes, None if is_head else planes)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(GraphConvBlock(self.in_planes, self.in_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pno):
        p0, n0, o0 = pno
        x0 = n0 if self.c == 3 else torch.cat((p0, n0), 1)
        p1, n1, x1, o1 = self.enc1([p0, n0, x0, o0])
        p2, n2, x2, o2 = self.enc2([p1, n1, x1, o1])
        p3, n3, x3, o3 = self.enc3([p2, n2, x2, o2])
        p4, n4, x4, o4 = self.enc4([p3, n3, x3, o3])
        p5, n5, x5, o5 = self.enc5([p4, n4, x4, o4])
        feat_21 = self.multi1([p2, x2, o2], [p3, x3, o3])[1]
        feat_31 = self.multi2([p3, x3, o3], [p4, x4, o4], [p2, feat_21, o2])[1]
        feat_41 = self.multi3([p4, x4, o4], [p5, x5, o5], [p3, feat_31, o3])[1]
        feat_22 = self.multi4([p2, feat_21, o2], [p3, feat_31, o3])[1]
        feat_32 = self.multi5([p3, feat_31, o3], [p4, feat_41, o4], [p2, feat_22, o2])[1]
        feat_23 = self.multi6([p2, feat_22, o2], [p3, feat_32, o3])[1]
        x5 = self.dec5[1:]([p5, n5, self.dec5[0]([p5, n5, x5, o5], pnxo2=None, pnxo3=[p4, n4, feat_41, o4]), o5])[2]
        x4 = self.dec4[1:]([p4, n4, self.dec4[0]([p4, n4, feat_41, o4], [p5, n5, x5, o5], [p3, n3, feat_32, o3]), o4])[2]
        x3 = self.dec3[1:]([p3, n3, self.dec3[0]([p3, n3, feat_32, o3], [p4, n4, x4, o4], [p2, n2, feat_23, o2]), o3])[2]
        x2 = self.dec2[1:]([p2, n2, self.dec2[0]([p2, n2, feat_23, o2], [p3, n3, x3, o3]), o2])[2]
        x1 = self.dec1[1:]([p1, n1, self.dec1[0]([p1, n1, x1, o1], [p2, n2, x2, o2]), o1])[2]
        x = self.cls(x1)
        return x


def graphAttention_seg_repro(experiment_type='IV', **kwargs):
    """创建不同实验的模型实例"""
    config = {
        'I': {'use_lgaf': False, 'use_sag_pooling': False},  # Baseline: GAC + 标准池化
        'II': {'use_lgaf': True, 'use_sag_pooling': False},  # LGAF (无SAG)
        'III': {'use_lgaf': False, 'use_sag_pooling': True},  # GAC + SAG-Pooling
        'IV': {'use_lgaf': True, 'use_sag_pooling': True}  # 完整LGASS
    }

    return GraphAttentionSeg(
        GraphAttentionBlock, [2, 3, 4, 6, 3],
        **config[experiment_type], **kwargs
    )