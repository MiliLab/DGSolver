import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from PIL import Image
from torch import nn
import imageio
import cv2
import numpy as np
import random

# import torch_npu
# from torch_npu.contrib import transfer_to_npu

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def default(val, d):
    if exists(val) and (val is not None):
        return val
    return d() if callable(d) else d


def set_seed(SEED): 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

degradtion_cache = ['Enlighening', 'Desnowing', 'Deraining', 'Deblur', 'Dehazing']

class train_dataset(Dataset):
    def __init__(self, root_dir, task_folder, sub_folder = None, image_size = 256):
        super().__init__()
        assert task_folder in degradtion_cache
        self.root_path = root_dir
        self.task_path = os.path.join(root_dir,task_folder)
        if sub_folder is not None:
            self.sub_datasets = [sub_folder]
        else:
            self.sub_datasets = os.listdir(self.task_path)
        self.image_size = image_size   
        self.skip_datasets = []
        self.image_load_path,self.image_name_list = [],[]
        self.condi_load_path,self.condi_name_list = [],[] 

        for sub_dataset in self.sub_datasets:   
            assert sub_dataset in os.listdir(self.task_path)
            if sub_dataset in self.skip_datasets:
                continue
            else:
                if sub_dataset == 'ITS_v2' or sub_dataset == 'SOTS' or sub_dataset == 'RESIDE' :
                    dataset_path = os.path.join(self.task_path,sub_dataset,'train')
                    image_path = os.path.join(dataset_path,'label')
                    condi_path = os.path.join(dataset_path ,'condition')
                    condi_file_list = os.listdir(condi_path) 
                    condi_path_list = [os.path.join(condi_path,x) for x in condi_file_list]
                    image_file_list = os.listdir(condi_path) 
                    image_path_list = [os.path.join(image_path,x.split('_',1)[0]+x[-4:]) for x in condi_file_list]
                else:
                    dataset_path = os.path.join(self.task_path,sub_dataset,'train')
                    image_path = os.path.join(dataset_path,'label')
                    image_file_list = os.listdir(image_path)
                    condi_path = os.path.join(dataset_path ,'condition')
                    condi_file_list = os.listdir(condi_path)
                    image_path_list = [os.path.join(image_path,x) for x in image_file_list]
                    condi_path_list = [os.path.join(condi_path,x) for x in condi_file_list]
            
            self.image_load_path = self.image_load_path + image_path_list
            self.image_name_list = self.image_name_list + image_file_list
            self.condi_load_path = self.condi_load_path + condi_path_list
            self.condi_name_list = self.condi_name_list + condi_file_list
            
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        assert len(self.condi_name_list) == len(self.image_name_list),f'the number of label files does not match the number of condition files'
        return len(self.condi_name_list)

    def __getitem__(self, index):
        image_name_file = self.image_name_list[index]
        condi_name_file = self.condi_name_list[index]
        if condi_name_file != image_name_file:
            print(self.condi_load_path[index],'\n',self.image_load_path[index])
        assert condi_name_file == image_name_file, f'image pairs are not matched'
        image_file_path = self.image_load_path[index]
        condi_file_path = self.condi_load_path[index]
        image = imageio.imread(image_file_path)
        condi = imageio.imread(condi_file_path)
        if image.shape != condi.shape:
            print(image_file_path,'\n',condi_file_path)
        assert image.shape == condi.shape, 'image sizes are not matched'
        image,condi = self.random_crop_size(image,condi,self.image_size)
        
        if self.transform is not None:
            image_tf = self.transform(image)
            condi_tf = self.transform(condi)
        
        return condi_name_file,image_tf,condi_tf

    def resize_shape(self, image, short_side_length): 
        oldh, oldw, _ = image.shape[0],image.shape[1],image.shape[2]
        if min(oldh, oldw) < short_side_length:
            scale = short_side_length * 1.0 / min(oldh, oldw)
            newh, neww = oldh * scale, oldw * scale
            image = cv2.resize(image,(int(neww + 0.5),int(newh + 0.5))) 
        return image
    
    def random_crop_size(self,imageA,imageB,crop_size):
        imageA = self.resize_shape(imageA,crop_size)
        imageB = self.resize_shape(imageB,crop_size)
        assert imageA.shape == imageB.shape, f'image sizes are not matched'
        h,w,_ = imageA.shape
        h_start,w_start = np.random.randint(0,h-crop_size+1),np.random.randint(0,w-crop_size+1)
        imageA_crop = imageA[h_start:h_start+crop_size,w_start:w_start+crop_size,:]
        imageB_crop = imageB[h_start:h_start+crop_size,w_start:w_start+crop_size,:]
        return imageA_crop,imageB_crop


class test_dataset(Dataset):
    def __init__(self, root_dir, task_folder, sub_folder = None):
        super().__init__()
        assert task_folder in degradtion_cache
        self.root_path = root_dir
        self.task_path = os.path.join(root_dir,task_folder)
        if sub_folder is not None:
            self.sub_datasets = [sub_folder]
        else:
            self.sub_datasets = os.listdir(self.task_path)
        self.skip_datasets =  []
        self.image_load_path,self.image_name_list = [],[]
        self.condi_load_path,self.condi_name_list = [],[] 

        for sub_dataset in self.sub_datasets:   
            assert sub_dataset in os.listdir(self.task_path)
            if sub_dataset in self.skip_datasets:
                continue
            else:
                if sub_dataset == 'ITS_v2' or sub_dataset == 'SOTS':
                    dataset_path = os.path.join(self.task_path,sub_dataset,'test')
                    image_path = os.path.join(dataset_path,'label')
                    condi_path = os.path.join(dataset_path ,'condition')
                    condi_file_list = os.listdir(condi_path) 
                    condi_path_list = [os.path.join(condi_path,x) for x in condi_file_list]
                    image_file_list = os.listdir(condi_path) 
                    image_path_list = [os.path.join(image_path,x.split('_',1)[0]+x[-4:]) for x in condi_file_list]
                else:
                    dataset_path = os.path.join(self.task_path,sub_dataset,'test')
                    image_path = os.path.join(dataset_path,'label')
                    image_file_list = os.listdir(image_path)
                    condi_path = os.path.join(dataset_path ,'condition')
                    condi_file_list = os.listdir(condi_path)
                    image_path_list = [os.path.join(image_path,x) for x in image_file_list]
                    condi_path_list = [os.path.join(condi_path,x) for x in condi_file_list]
                    
            self.image_load_path = self.image_load_path + image_path_list
            self.image_name_list = self.image_name_list + image_file_list
            self.condi_load_path = self.condi_load_path + condi_path_list
            self.condi_name_list = self.condi_name_list + condi_file_list
            
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        assert len(self.condi_name_list) == len(self.image_name_list),f'the number of label files does not match the number of condition files'
        return len(self.condi_name_list)

    def __getitem__(self, index):
        image_name_file = self.image_name_list[index]
        condi_name_file = self.condi_name_list[index]
        if condi_name_file != image_name_file:
            print(self.condi_load_path[index],'\n',self.image_load_path[index])
        assert condi_name_file == image_name_file, f'image pairs are not matched'
        image_file_path = self.image_load_path[index]
        condi_file_path = self.condi_load_path[index]
        image = imageio.imread(image_file_path)
        condi = imageio.imread(condi_file_path)
        assert image.shape == condi.shape, 'image sizes are not matched'
        
        if self.transform is not None:
            image_tf = self.transform(image)
            condi_tf = self.transform(condi)
        
        return condi_file_path,image_tf,condi_tf