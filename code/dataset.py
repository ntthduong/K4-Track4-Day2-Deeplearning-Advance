"""DeepWeeds datasets: fixed official splits, reproducible loaders."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia', 'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negatives']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_split(labels_dir, fold=0):
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    return tuple(pd.read_csv(Path(labels_dir) / f'{s}_subset{fold}.csv') for s in ('train', 'val', 'test'))

def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    sets, result = {}, {'n': {}, 'per_class': {}, 'overlap': {}}
    for name, df in frames.items():
        if not {'Filename', 'Label'} <= set(df):
            raise ValueError(f'{name}: missing Filename/Label')
        if df.Filename.isna().any() or df.Filename.duplicated().any() or not df.Label.isin(range(9)).all():
            raise ValueError(f'{name}: invalid filenames or labels')
        sets[name] = set(df.Filename)
        result['n'][name] = len(df)
        result['per_class'][name] = df.Label.value_counts().reindex(range(9), fill_value=0).to_dict()
        missing = [f for f in df.Filename if not (Path(images_dir) / f).is_file()]
        if missing:
            raise ValueError(f'{name}: {len(missing)} missing images, e.g. {missing[:3]}')
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        result['overlap'][f'{a}/{b}'] = len(sets[a] & sets[b])
    if any(result['overlap'].values()) or len(set.union(*sets.values())) != 17509:
        raise ValueError('Splits overlap or their union does not contain 17509 images')
    if any(abs(result['n'][s] / 17509 - r) > .01 for s, r in zip(frames, (.6, .2, .2))):
        raise ValueError('Split proportions differ from 60/20/20 by more than 1 percentage point')
    print(result)
    return result

def build_transforms(train, img_size=224, aug='basic', mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        steps = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
        if aug == 'color':
            steps.append(T.ColorJitter(.2, .2, .2, .05))
        elif aug == 'trivial':
            steps.append(T.TrivialAugmentWide())
        elif aug == 'randaug':
            steps.append(T.RandAugment(num_ops=2, magnitude=9))
        elif aug != 'basic':
            raise ValueError(f'Unknown augmentation: {aug}')
    else:
        steps = [T.Resize(round(img_size * 256 / 224)), T.CenterCrop(img_size)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])

class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df, self.images_dir, self.transform = df.reset_index(drop=True), Path(images_dir), transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as image:
            image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, int(row.Label), str(row.Filename)

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weighted = None
    if sampler is not None:
        if not train or sampler != 'balanced':
            raise ValueError('balanced sampler is supported only during training')
        counts = df.Label.value_counts()
        weights = torch.tensor([1 / counts[y] for y in df.Label], dtype=torch.double)
        weighted = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and weighted is None, sampler=weighted, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(df) >= batch_size,
                      worker_init_fn=seed_worker, generator=generator)
