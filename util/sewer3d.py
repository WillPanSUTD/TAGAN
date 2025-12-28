from torch.utils.data import Dataset
from util.transforme import *
from tqdm import tqdm
import os


class Sewer3dDataset(Dataset):
    def __init__(self, root='../data/normal', npoints=2048, split='train'):
        self.root = root
        self.split = split
        self.sewers = []
        # self.transform = Compose([RandomJitter(), RandomScale([0.9, 1.1]), ToTensor()])
        self.transform = Compose([ToTensor()])
        if self.split == 'train':
            self.root = os.path.join(self.root, 'train')
            sewers = os.listdir(self.root)
            weight = np.zeros(8)
            for index in tqdm(range(len(sewers)), desc='Load training set  ', total=len(sewers)):
                # self.sewers.extend([sewers[index]] * 9)
                point_root = os.path.join(self.root, sewers[index])
                points = np.loadtxt(point_root, dtype=np.float16)
                loop = points.shape[0] // npoints
                choice = np.random.choice(points.shape[0], npoints * loop, replace=False)
                points_choice = points[choice]
                for idx in range(loop):
                    points = points_choice[idx * npoints:(idx + 1) * npoints]
                    self.sewers.append(points)
                targets = points[:, -1]
                tmp, _ = np.histogram(targets, range(9))
                weight += tmp

            l_weight = np.power(np.amax(weight) / weight, 1 / 2)
            l_weight = l_weight / np.sum(l_weight)
            self.weight = weight / np.amax(weight)
            self.l_weight = l_weight

        else:
            self.root = os.path.join(self.root, 'test')
            sewers = os.listdir(self.root)
            for index in tqdm(range(len(sewers)), desc='Load test set      ', total=len(sewers)):
                # self.sewers.extend([sewers[index]] * 9)
                point_root = os.path.join(self.root, sewers[index])
                points = np.loadtxt(point_root, dtype=np.float16)
                choice = np.random.choice(points.shape[0], npoints, replace=False)
                points_choice = points[choice]
                self.sewers.append(points_choice)

    def __getitem__(self, index):
        points = self.sewers[index]
        points_coords = points[:, 0:3]
        points_feats = points[:, 3:6]
        points_labels = points[:, 6]
        coords, feats, targets = self.transform(points_coords, points_feats, points_labels)
        return coords, feats, targets

    def __len__(self):
        return len(self.sewers)
