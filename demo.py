import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.pointtransformer.GraphAttention_multiscaleV1 import graphAttention_seg_repro as Model
from util.data_util import collate_fn


class PointCloudInferenceDataset(Dataset):
    def __init__(self, file_path, num_points=16384):
        self.file_path = file_path
        self.num_points = num_points
        self.points = self.load_and_preprocess(file_path)
        self.transform = ToTensor()  # 添加转换函数

    def load_and_preprocess(self, file_path):
        """加载并预处理点云数据 - 修复版本"""
        # 加载点云文件
        data = np.loadtxt(file_path)

        # 验证数据格式
        if data.shape[1] not in [3, 6, 7]:
            raise ValueError(
                f"点云格式错误: 应为3列(坐标)、6列(坐标+法向量)或7列(坐标+法向量+标签)，实际有{data.shape[1]}列")

        # 提取坐标和法向量
        coords = data[:, :3]

        # 自动添加缺失的法向量
        if data.shape[1] <= 3:
            normals = np.zeros_like(coords)
        else:
            normals = data[:, 3:6]

        # 随机采样
        if len(coords) > self.num_points:
            idx = np.random.choice(len(coords), self.num_points, replace=False)
            coords = coords[idx]
            normals = normals[idx]
        elif len(coords) < self.num_points:
            # 重复采样填充
            idx = np.random.choice(len(coords), self.num_points - len(coords), replace=True)
            coords = np.vstack([coords, coords[idx]])
            normals = np.vstack([normals, normals[idx]])

        return np.hstack([coords, normals])

    def __len__(self):
        return 1  # 单文件处理

    def __getitem__(self, idx):
        coords = self.points[:, :3]
        feats = self.points[:, 3:6]

        # 转换为Tensor - 修复的关键点
        coords, feats, _ = self.transform(coords, feats, np.zeros(len(coords)))
        return coords, feats, torch.zeros(len(coords), dtype=torch.long)  # 返回Tensor格式的虚拟标签


class ToTensor:
    """将NumPy数组转换为PyTorch Tensor"""

    def __call__(self, coords, feats, labels):
        coords = torch.from_numpy(coords).float()
        feats = torch.from_numpy(feats).float()
        labels = torch.from_numpy(labels).long()
        return coords, feats, labels


class PointCloudSegmenter:
    """点云分割处理器 - 完全修复版本"""

    def __init__(self, model_path, num_classes=8, feat_dim=6):
        """
        初始化点云分割器
        :param model_path: 预训练模型路径
        :param num_classes: 类别数量
        :param feat_dim: 特征维度
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, num_classes, feat_dim)
        self.class_names = {
            0: 'Normal',
            1: 'Spalling',
            2: 'Blockage',
            3: 'Corrosion',
            4: 'Misalign',
            5: 'Deposit',
            6: 'Displace',
            7: 'RubberRing'
        }

    def load_model(self, model_path, num_classes, feat_dim):
        """加载预训练模型"""
        model = Model(c=feat_dim, k=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def process_point_cloud(self, file_path, output_dir):
        """
        处理点云文件 - 完全修复版本
        :param file_path: 输入点云文件路径
        :param output_dir: 输出目录
        """
        # 创建数据集和加载器
        dataset = PointCloudInferenceDataset(file_path)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        # 处理数据
        for coords, feats, _, offset in loader:
            coords = coords.float().to(self.device)
            feats = feats.float().to(self.device)
            offset = offset.to(self.device)

            # 模型预测
            with torch.no_grad():
                seg_pred = self.model([coords, feats, offset])
                pred_labels = seg_pred.argmax(dim=1).cpu().numpy()

            # 保存结果 - 使用原始点云坐标
            original_points = dataset.points[:, :3]  # 获取原始点云坐标
            self.save_results(
                original_points,
                pred_labels,
                output_dir,
                os.path.basename(file_path)
            )

    def save_results(self, points, labels, output_dir, base_name):
        """保存分割结果 - 增强版"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(base_name)[0]  # 移除文件扩展名

        # 1. 保存完整分割结果
        full_output = np.hstack([points, labels[:, np.newaxis]])
        np.savetxt(
            os.path.join(output_dir, f"{base_name}_full.txt"),
            full_output,
            fmt="%.6f %.6f %.6f %d"
        )

        # 2. 按类别分别保存
        for class_id, class_name in self.class_names.items():
            class_mask = (labels == class_id)
            if np.any(class_mask):
                class_points = points[class_mask]
                class_file = os.path.join(output_dir, f"{base_name}_{class_name}.txt")
                np.savetxt(class_file, class_points, fmt="%.6f %.6f %.6f")

        # 3. 保存可视化文件 (PLY格式)
        self.save_ply_file(points, labels, os.path.join(output_dir, f"{base_name}_viz.ply"))
        print(f"处理完成! 结果保存至: {output_dir}")

    def save_ply_file(self, points, labels, filename):
        """保存带颜色的PLY文件"""
        header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        color_map = {
            0: [0.0, 0.0, 1.0],  # 正常 - 灰色
            1: [1.0, 0.0, 0.0],  # Spalling - 红色
            2: [1.0, 0.5, 0.0],  # Blockage - 蓝色
            3: [1.0, 1.0, 0.0],  # Corrosion - 黄色
            4: [0.0, 0.8, 0.0],  # Misalign - 绿色
            5: [0.63, 0.13, 0.94],  # Deposit - 橙色
            6: [0.0, 1.0, 1.0],  # Displace - 紫色
            7: [1.0, 0.0, 1.0],  # RubberRing - 青色
        }

        with open(filename, 'w') as f:
            f.write(header.format(len(points)))
            for pt, label in zip(points, labels):
                r, g, b = [int(c * 255) for c in color_map.get(label, [1, 1, 1])]
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")

if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "logs/4-weight_2025-06-17 12:15:59/best_model.pth"  # 替换为您的模型路径
    INPUT_FILE = "data/sewer3d_semantic_segmentation/train/real_459.txt"  # 输入点云文件
    OUTPUT_DIR = "demodata/2"  # 输出目录

    # 创建并运行分割器
    segmenter = PointCloudSegmenter(MODEL_PATH)
    segmenter.process_point_cloud(INPUT_FILE, OUTPUT_DIR)
    print("点云分割处理完成!")