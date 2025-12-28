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
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, block_idx=0):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        # 对应老师公式: ai = MLP(dn, dtheta, dz, dr)
        # 输入维度为 4: [diff_n(1), diff_theta(1), diff_z(1), diff_r(1)]
        self.linear_explicit = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, out_planes)
        )

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pnxo) -> torch.Tensor:
        p, n, x, o = pnxo

        # 1. 基础特征
        x_q = self.linear_q(x)
        # x_edge: [relative_xyz, feats]
        x_edge, distance = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, edge=True)
        _, x_e = x_edge[:, :, 0:3], x_edge[:, :, 3:]
        x_k = self.linear_k(x_e)
        x_v = self.linear_v(x_e)

        # 2. 计算显式物理特征差异 (Explicit Modeling)
        # A. 拓扑坐标 (r, sin, cos, z) -> 我们需要还原出 r, theta, z
        cyl = get_cylindrical_features(p)  # (N, 4) -> [r, sin, cos, z]

        # 这里为了计算差异，我们重新构建简化的 (r, theta, z) 用于相减
        # 注意: get_cylindrical_features 返回的是 [r, sin, cos, z]
        # 我们手动用 atan2 恢复一下 theta 用于计算差值 (或者直接在 get 函数里改，但为了不破坏之前的结构，在这里处理)
        # 简单做法：我们直接利用 get_cylindrical_features 返回的 4 维特征计算 "拓扑距离"
        # 但老师公式是分开的，所以我们需要分别提取

        # 重新计算纯粹的 (r, theta, z) 用于物理差异计算
        p_r = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2).unsqueeze(1)  # (N, 1)
        p_theta = torch.atan2(p[:, 1], p[:, 0]).unsqueeze(1)  # (N, 1)
        p_z = p[:, 2].unsqueeze(1)  # (N, 1)
        p_phys = torch.cat([p_r, p_theta, p_z], dim=1)  # (N, 3)

        # Grouping 物理坐标
        # phys_edge: (neighbor - center) -> [dr, dtheta, dz]
        phys_edge, _ = pointops.queryandgroup(self.nsample, p, p, p_phys, None, o, o, use_xyz=False, edge=True)

        dr = torch.abs(phys_edge[..., 0:1])  # |ri - rj|

        # 处理角度周期性 |theta_i - theta_j|
        dtheta_raw = phys_edge[..., 1:2]
        dtheta = torch.abs(torch.remainder(dtheta_raw + torch.pi, 2 * torch.pi) - torch.pi)

        dz = torch.abs(phys_edge[..., 2:3])  # |zi - zj|

        # B. 法向量差异 (Normal Diff)
        # n_edge 是 (neighbor_n - center_n)
        n_edge, _ = pointops.queryandgroup(self.nsample, p, p, n, None, o, o, use_xyz=False, edge=True)
        dn = torch.norm(n_edge, dim=-1, keepdim=True)  # |ni - nj|

        # 3. 组合四项特征 [dn, dtheta, dz, dr]
        explicit_feat = torch.cat([dn, dtheta, dz, dr], dim=-1)  # (N, k, 4)

        # 通过 MLP 映射到 Attention 空间
        # explicit_emb: (N, nsample, out_planes)
        explicit_emb = self.linear_explicit(explicit_feat.reshape(-1, 4)).reshape(x_q.shape[0], self.nsample, -1)

        # 4. 计算 Attention
        # w = Q - K + Explicit_Physics_Bias
        w = x_q.unsqueeze(1) - x_k + explicit_emb

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)

        distance_weight = torch.exp(-distance.unsqueeze(-1))
        x_v = x_v * distance_weight

        num, nSample, c = x_v.shape
        s = self.share_planes

        # 聚合时也加上显式特征，增强表达
        x1 = ((x_v + explicit_emb).view(num, nSample, s, c // s) * w.unsqueeze(2)).sum(1).view(num, c)
        return x1

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, nsample):
        super().__init__()
        self.nsample = nsample
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), self.relu)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), self.relu)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_channels))

    def forward(self, pnxo):
        p, n, x, o = pnxo
        x_edge, _ = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, edge=True)
        p_r, x_e = x_edge[:, :, 0:3], x_edge[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)  # (num,nsample,o_c)
        x = torch.cat((x.unsqueeze(1).repeat(1, self.nsample, 1), x_e), dim=-1)  # (num, nsample, in_planes*2)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (num, nsample, in_planes*2) -> (1, in_planes*2, num, nsample)
        x = self.conv1(x)  # (1, in_planes*2, num, nsample) -> (1, out_channels, num, nsample)
        x = self.conv2(x) + p_r.permute(2, 0, 1).unsqueeze(0)  # (1, out_channels, num, nsample) -> (1, out_channels, num, nsample)
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


# 沿管轴方向权重
# [修改] 符合老师要求的 Pooling: 融合 Curvature, Radial, Density
class TransitionDown(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, nsample=16, use_sag=True):
        super().__init__()

        self.stride, self.nsample = stride, nsample
        self.use_sag = use_sag  # 保持接口兼容

        if stride != 1:
            # A1. 定义线性变换与池化
            # self.linear: 也就是 MLP。把特征从 in 映射到 out（升维，比如 32->64）。
            # self.pool: 最大池化。在 16 个邻居里挑一个最强的特征。这是传统的做法。
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)

            self.weight_net = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
                nn.Sigmoid()  # 输出 0-1 之间的权重
            )

        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)


        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pnxo):

        p, n, x, o = pnxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.int, device='cuda')

            idx = pointops.furthestsampling(p, o, n_o)
            n_p, n_n = p[idx.long(), :], n[idx.long(), :]

            x_grouped, _ = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            # x_grouped: [relative_xyz(3), feats(c)]

            if self.use_sag:
                # --- 老师要求的三个特征计算 ---

                # 特征 1： Radial Offset (径向偏移): r
                neighbor_xyz = x_grouped[..., 0:3] + n_p.unsqueeze(1)
                # 计算 r = sqrt(x^2 + y^2)
                radial_offset = torch.sqrt(neighbor_xyz[..., 0] ** 2 + neighbor_xyz[..., 1] ** 2).unsqueeze(
                    -1)  # (N, k, 1)

                # 特征 2：Density (密度) Density (密度): 邻域点到中心的距离 (距离越小，密度越高)
                dist = torch.norm(x_grouped[..., 0:3], dim=-1, keepdim=True)
                density = 1.0 / (dist + 1e-6)

                # 特征 3： Curvature (曲率): 近似为法向量的差异
                n_grouped, _ = pointops.queryandgroup(self.nsample, p, n_p, n, None, o, n_o, use_xyz=False)
                dot_prod = torch.matmul(n_grouped, n_n.unsqueeze(2))  # (N, k, 1)
                curvature = 1.0 - torch.abs(dot_prod)  # 1 - |cos_theta|

# 特征加权 (Attention Weighting)
                # --- 计算权重 ---
                # 拼接 [Curvature, Radial, Density]
                physics_feat = torch.cat([curvature, radial_offset, density], dim=-1)  # (N, k, 3)
                weights = self.weight_net(physics_feat)  # (N, k, 1)

                # 加权特征
                # 只对特征部分加权，前3维是坐标
                feats = x_grouped[..., 3:]
                feats = feats * (1 + weights)  # 强化重要点 (Residual connection style)
                x = feats
            else:
                x = x_grouped[..., 3:]
            x = self.linear(x)
            x = self.relu(self.bn(x.transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            # 更新变量与返回
            p, n, o = n_p, n_n, n_o
        else:
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
        'IV': {'use_lgaf': True, 'use_sag_pooling': True}  # 完整LGASS 极坐标
    }

    return GraphAttentionSeg(
        GraphAttentionBlock, [2, 3, 4, 6, 3],
        **config[experiment_type], **kwargs
    )