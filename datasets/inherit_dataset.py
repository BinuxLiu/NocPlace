
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

import datasets.dataset_utils as dataset_utils

class InheritDataset(data.Dataset):
    def __init__(self, dataset_folder, image_size=512, resize_test_imgs=False):
        self.images_paths = dataset_utils.read_images_paths(dataset_folder, get_abs_path=True)

        self.dataset_name = os.path.basename(dataset_folder)
        
        subfolders = [f.name for f in os.scandir(dataset_folder) if f.is_dir()]
        for folder_name in subfolders:
            os.makedirs(dataset_folder + "_feat/" + folder_name, exist_ok=True)

        self.images_num = len(self.images_paths)

        transforms_list = []
        if resize_test_imgs:
            # Resize to image_size along the shorter side while maintaining aspect ratio
            transforms_list += [transforms.Resize(image_size, antialias=True)]
        transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.base_transform = transforms.Compose(transforms_list)
    
    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = InheritDataset.open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, image_path, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #db: {self.images_num} >"
