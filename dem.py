import torch
import numpy as np
import open3d as o3d
from model.pointtransformer.GraphAttention_multiscaleV1 import GraphAttentionSeg

# 1. 加载模型（使用strict=False避免参数不匹配）
model = GraphAttentionSeg(num_classes=8)
model.load_state_dict(torch.load('logs/weight_2025-04-19 00_13_39/best_model.pth'), strict=False)
model.eval()

# 2. 加载点云数据
points = np.loadtxt('data/sewer3d_semantic_segmentation/train/real_469.txt')
coords = torch.from_numpy(points[:,:3]).float()  # 前3列是xyz坐标
feats = torch.ones_like(coords)  # 如果没有特征，使用全1

# 3. 模型预测
with torch.no_grad():
    pred = model([coords.unsqueeze(0), feats.unsqueeze(0), torch.tensor([coords.shape[0]])])
    labels = torch.argmax(pred, dim=1).numpy()

# 4. 根据预测标签染色（使用您之前提供的配色方案）
color_map = np.array([
    [255,0,0],    # 类别0: 红色
    [0,255,0],     # 类别1: 绿色
    [0,0,255],     # 类别2: 蓝色
    [255,255,0],   # 类别3: 黄色
    [255,0,255],   # 类别4: 紫色
    [0,255,255],   # 类别5: 青色
    [128,128,128], # 类别6: 灰色
    [255,165,0]    # 类别7: 橙色
])

colors = color_map[labels]

# 5. 可视化
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(colors/255)
o3d.visualization.draw_geometries([pcd])

# 6. 保存结果
np.savetxt('predicted_labels.txt', labels, fmt='%d')
o3d.io.write_point_cloud('colored.ply', pcd)