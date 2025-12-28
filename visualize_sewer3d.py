import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# === 必须引入这三个模块，确保数据格式与训练一致 ===
from util.sewer3d import Sewer3dDataset
from util.data_util import collate_fn  # 必须使用这个collate_fn，否则没有offset
from model.pointtransformer.GraphAttention_multiscaleV1 import graphAttention_seg_repro as Model

# ---------------- 配置区域 ----------------
# 1. 模型路径：请修改为您要可视化的 .pth 路径
MODEL_PATH = 'logs/weight_2025-12-20 05:21:47/best_model.pth'
# 2. 实验类型：必须与训练时一致 ('IV' 是全功能模型, 'III' 是只有SAG)
EXP_TYPE = 'IV'
# 3. 输出文件夹
VISUAL_DIR = 'visualization_results'
# 4. GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ----------------------------------------

# 定义颜色映射 (RGB 0-255)
COLOR_MAP = {
    0: [192, 192, 192],  # Intact (灰色)
    1: [255, 0, 0],  # Spalling (红色 - 重点)
    2: [0, 255, 0],  # Blockage (绿色)
    3: [255, 255, 0],  # Corrosion (黄色 - 重点)
    4: [0, 0, 255],  # Misalignment (蓝色)
    5: [255, 0, 255],  # Deposit (洋红)
    6: [0, 255, 255],  # Displacement (青色)
    7: [255, 128, 0],  # RubberRing (橙色)
}


def prediction_to_color(pred_labels):
    """将标签转换为 RGB 颜色数组"""
    N = pred_labels.shape[0]
    colors = np.zeros((N, 3), dtype=np.int32)
    for cls_id, rgb in COLOR_MAP.items():
        indices = (pred_labels == cls_id)
        colors[indices] = rgb
    return colors


def save_point_cloud(coords, colors, filename):
    """保存为 .txt 文件"""
    # coords: (N, 3), colors: (N, 3)
    data = np.hstack((coords, colors))
    np.savetxt(filename, data, fmt='%.6f %.6f %.6f %d %d %d')
    print(f"Saved: {filename}")


def main():
    if not os.path.exists(VISUAL_DIR):
        os.makedirs(VISUAL_DIR)

    # 1. 加载数据集 (参数名已修正，与 train_sewer3d.py 保持一致)
    root = './data/sewer3d_semantic_segmentation'  # 请确认路径是否正确
    feat_dim = 6
    num_class = 8

    print("Loading Dataset...")
    # [修正] 参数改为 root 和 npoints，去除不支持的 data_root 和 test_area
    test_set = Sewer3dDataset(root=root, npoints=16384, split='test')

    # [修正] 必须加上 collate_fn=collate_fn，否则无法生成 offset
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)

    # 2. 初始化模型
    print(f"Loading Model (Experiment Type: {EXP_TYPE})...")
    # [修正] 参数需与训练一致: c=6 (feat_dim), k=8 (num_class)
    classifier = Model(c=feat_dim, k=num_class, experiment_type=EXP_TYPE).cuda()

    # 3. 加载权重
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH)
    # 处理可能的 state_dict 键名不匹配问题 (如带 module. 前缀)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)
    classifier.eval()

    # 4. 推理
    print("Start generating visualization files...")

    with torch.no_grad():
        # [修正] 这里的解包必须匹配 collate_fn 的输出
        for i, (coords, feats, target, offset) in tqdm(enumerate(test_loader), total=len(test_loader)):

            # 数据迁移到 GPU
            coords, feats, target, offset = coords.float().cuda(), feats.float().cuda(), target.long().cuda(), offset.cuda()

            # [修正] 模型输入必须是列表形式 [coords, feats, offset]
            pred = classifier([coords, feats, offset])

            # pred: (N, num_class) -> 取最大值索引
            pred_choice = pred.data.max(1)[1]

            # 转换回 CPU numpy
            # 注意：由于 batch_size=1，且经过了 collate_fn，coords 已经是 (N, 3) 形状
            points_np = coords.cpu().numpy()
            gt_labels = target.cpu().numpy()
            pred_labels = pred_choice.cpu().numpy()

            # 生成颜色
            gt_colors = prediction_to_color(gt_labels)
            pred_colors = prediction_to_color(pred_labels)

            # 保存
            save_point_cloud(points_np, gt_colors, os.path.join(VISUAL_DIR, f'Sample_{i}_GT.txt'))
            save_point_cloud(points_np, pred_colors, os.path.join(VISUAL_DIR, f'Sample_{i}_Pred.txt'))

            # 只保存前 10 个用于展示
            if i >= 10:
                break

    print(f"Done! Check the '{VISUAL_DIR}' folder.")


if __name__ == '__main__':
    main()