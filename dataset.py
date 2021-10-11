from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ImageCSVData(Dataset):
    
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        self.img_path = os.listdir(self.image_dir)
        label_data = pd.read_csv(label_file, names=['image_name', 'label'])
        self.label_data = {image_name:float(label) for image_name,label in zip(label_data['image_name'], label_data['label'])}
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_data[img_name.split('.')[0]]
        return img, label
    
    def __len__(self):
        return len(self.img_path)