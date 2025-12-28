import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.pointtransformer.GraphAttention_multiscaleV1 import graphAttention_seg_repro as Model
from util.data_util import collate_fn
from tqdm import tqdm


# 自定义collate函数
def custom_collate_fn(batch):
    coords_list, feats_list, metadata_list = [], [], []

    for coords, feats, metadata in batch:
        coords_list.append(coords)
        feats_list.append(feats)
        metadata_list.append(metadata)

    coords_batch = torch.cat(coords_list)
    feats_batch = torch.cat(feats_list)

    # 计算offset
    lengths = [len(c) for c in coords_list]
    offset = torch.cumsum(torch.IntTensor(lengths), dim=0)

    return coords_batch, feats_batch, metadata_list, offset
class PointCloudInferenceDataset(Dataset):
    """改进的点云推理数据集类 - 支持全分辨率输出"""

    def __init__(self, file_path, num_points=16384, device='cuda'):
        self.file_path = file_path
        self.device = device
        self.num_points = num_points
        self.full_point_cloud = self.load_full_point_cloud(file_path)
        self.transform = ToTensor()

    def load_full_point_cloud(self, file_path):
        """加载完整点云数据"""
        print(f"加载点云: {file_path}")
        data = np.loadtxt(file_path)

        # 验证数据格式
        if data.shape[1] not in [3, 6, 7]:
            raise ValueError(
                f"点云格式错误: 应为3列(坐标)、6列(坐标+法向量)或7列(坐标+法向量+标签)，实际有{data.shape[1]}列")

        self.original_point_count = len(data)
        print(f"原始点数: {self.original_point_count:,}")

        # 提取坐标和法向量
        coords = data[:, :3].astype(np.float32)

        # 自动添加缺失的法向量
        if data.shape[1] <= 3:
            normals = np.zeros_like(coords)
        else:
            normals = data[:, 3:6].astype(np.float32)

        # 保存完整的原始点云
        self.full_points = np.hstack([coords, normals])
        self.full_coords = coords
        self.full_feats = normals
        self.original_point_count = len(data)  # 关键修复
        print(f"原始点数: {self.original_point_count:,}")
        # 创建固定大小的处理块（用于模型推理）
        return self.create_processing_chunks()

    def create_processing_chunks(self):
        """
        创建处理块 - 使用滑动窗口方法保留全分辨率
        每个块覆盖部分点云，有50%重叠避免边界伪影
        """
        chunk_indices = []
        point_count = len(self.full_coords)
        chunk_size = self.num_points
        step_size = max(1, chunk_size // 2)  # 50% 重叠

        # 计算需要的块数量
        chunks = max(1, (point_count - chunk_size) // step_size + 1)
        print(f"创建 {chunks} 个处理块 ({chunk_size} 点/块, 步长 {step_size})")

        # 生成块索引
        for i in range(chunks):
            start_idx = i * step_size
            end_idx = min(start_idx + chunk_size, point_count)

            # 处理最后一小块
            if end_idx - start_idx < chunk_size:
                start_idx = max(0, point_count - chunk_size)
                end_idx = point_count

            chunk_indices.append((start_idx, end_idx))

        return chunk_indices

    def __len__(self):
        return len(self.full_point_cloud)

    def __getitem__(self, idx):
        start_idx, end_idx = self.full_point_cloud[idx]

        coords = self.full_coords[start_idx:end_idx]
        feats = self.full_feats[start_idx:end_idx]

        # 转换为Tensor
        coords, feats, _ = self.transform(coords, feats, np.zeros(len(coords)))

        # 返回三元组，将元数据作为第三个元素
        return coords, feats, torch.tensor([start_idx, end_idx], dtype=torch.long)


class PointCloudSegmenter:
    """高分辨率点云分割处理器"""

    def __init__(self, model_path, num_classes=8, feat_dim=6):
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
        # 添加GPU状态信息
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    def load_model(self, model_path, num_classes, feat_dim):
        """加载预训练模型 - 增加错误处理"""
        print(f"加载模型: {model_path}")
        try:
            model = Model(c=feat_dim, k=num_classes)
            checkpoint = torch.load(model_path, map_location=self.device)

            # 兼容不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)  # 直接是模型状态字典

            model = model.to(self.device)
            model.eval()

            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型加载成功! 参数量: {total_params:,}")
            return model
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

    def process_point_cloud(self, file_path, output_dir):
        # 创建数据集和加载器
        dataset = PointCloudInferenceDataset(file_path)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

        # 准备完整结果容器
        full_predictions = np.zeros(len(dataset.full_coords), dtype=np.int32)
        processed_indices = set()
        print("开始点云分割处理...")
        start_time = time.time()

        # 使用进度条处理每个块
        for batch in tqdm(loader, desc="处理点云块", unit="chunk"):
            # 解包批次数据
            coords, feats, metadata_list, offset = batch

            # 提取元数据
            metadata = metadata_list[0]  # batch_size=1
            start_idx = metadata[0].item()
            end_idx = metadata[1].item()
            global_indices = np.arange(start_idx, end_idx)

            # 关键修复：确保offset为Int32类型
            offset = offset.to(torch.int32)  # 显式转换为Int32

            # 移动到GPU并确保正确数据类型
            coords = coords.float().to(self.device)
            feats = feats.float().to(self.device)
            offset = offset.to(self.device)

            # 跳过已处理的索引
            if all(idx in processed_indices for idx in global_indices):
                continue

            # 模型预测
            with torch.no_grad():
                seg_pred = self.model([coords, feats, offset])
                pred_labels = seg_pred.argmax(dim=1).cpu().numpy()

            # 保存当前块的预测结果
            full_predictions[global_indices] = pred_labels
            processed_indices.update(global_indices)

        # 处理未覆盖的点（如果存在）
        missing_indices = set(range(len(full_predictions))) - processed_indices
        if missing_indices:
            print(f"警告: {len(missing_indices)} 点未被处理，使用相邻点模式")
            # 简单处理 - 使用最近的预测
            all_indices = np.array(sorted(missing_indices))
            closest = np.searchsorted(list(processed_indices), all_indices)
            closest = np.minimum(closest, len(full_predictions) - 1)
            full_predictions[all_indices] = full_predictions[closest]

        # 保存完整结果
        self.save_results(
            dataset.full_coords,
            full_predictions,
            output_dir,
            os.path.basename(file_path),
            original_count=dataset.original_point_count  # 添加缺失参数
        )
        elapsed = time.time() - start_time
        print(f"处理完成! 用时: {elapsed:.2f}秒 ({len(full_predictions) / elapsed:.0f} 点/秒)")

    def save_results(self, points, labels, output_dir, base_name, original_count):
        """保存高分辨率分割结果"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(base_name)[0]  # 移除文件扩展名

        # 保存完整预测结果
        full_output = np.hstack([points, labels[:, np.newaxis]])
        output_file = os.path.join(output_dir, f"{base_name}_segmented.txt")
        np.savetxt(output_file, full_output, fmt="%.6f %.6f %.6f %d")
        print(f"保存完整分割结果: {output_file} ({len(labels):,} 点)")

        # 按类别分割保存
        for class_id, class_name in self.class_names.items():
            class_mask = (labels == class_id)
            class_count = np.count_nonzero(class_mask)
            if class_count == 0:
                continue

            class_points = points[class_mask]
            class_file = os.path.join(output_dir, f"{base_name}_{class_name}.txt")
            np.savetxt(class_file, class_points, fmt="%.6f %.6f %.6f")
            print(f"  {class_name}: {class_count:,} 点 ({class_count / original_count * 100:.1f}%)")

        # 保存可视化文件
        viz_file = os.path.join(output_dir, f"{base_name}_viz.ply")
        self.save_ply_file(points, labels, viz_file)
        print(f"可视化文件已生成: {viz_file}")

    def save_ply_file(self, points, labels, filename):
        """生成高分辨率PLY可视化文件"""
        print(f"生成PLY可视化: {filename}")

        # 准备颜色映射
        color_map = {
            0: [200, 200, 200],  # 正常 - 灰色
            1: [255, 0, 0],  # Spalling - 红色
            2: [0, 0, 255],  # Blockage - 蓝色
            3: [255, 255, 0],  # Corrosion - 黄色
            4: [0, 255, 0],  # Misalign - 绿色
            5: [255, 165, 0],  # Deposit - 橙色
            6: [128, 0, 128],  # Displace - 紫色
            7: [0, 255, 255],  # RubberRing - 青色
        }

        # 创建文件头
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

        # 写入文件
        with open(filename, 'w') as f:
            f.write(header)
            for i, pt in enumerate(points):
                label = labels[i]
                color = color_map.get(label, [255, 255, 255])  # 未知类别默认白色
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {color[0]} {color[1]} {color[2]}\n")


# ToTensor类保持不变
class ToTensor:
    def __call__(self, coords, feats, labels):
        coords = torch.from_numpy(coords).float()
        feats = torch.from_numpy(feats).float()
        labels = torch.from_numpy(labels).long()
        return coords, feats, labels


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "logs/4-weight_2025-06-17 12:15:59/best_model.pth"  # 替换为您的模型路径
    INPUT_FILE = "real_459.txt"  # 输入点云文件
    OUTPUT_DIR = "demodata/2"  # 输出目录

    print("=" * 50)
    print("高分辨率点云分割处理")
    print("=" * 50)

    # 创建并运行分割器
    segmenter = PointCloudSegmenter(MODEL_PATH)
    segmenter.process_point_cloud(INPUT_FILE, OUTPUT_DIR)
    print("点云分割处理完成!")