import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, img_size, ep_len=3, img_glob='*.png'):
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)), key=lambda x: int(os.path.split(x)[-1][:-4]))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video
