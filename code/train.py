import os
import numpy as np
import math
import time
import imageio
import json

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from tqdm.auto import tqdm
from ema_pytorch import EMA
from pathlib import Path
from skimage.metrics import structural_similarity
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.utils.data import DataLoader
from model import (ResidualDiffusion, Unet, UnetRes, set_seed)
from datasets_setting import train_dataset,test_dataset,set_seed
# import torch_npu
# from torch_npu.contrib import transfer_to_npu


parser = ArgumentParser()
parser.add_argument("--project_description", type=str, default="UniDiffSolver For Image Restoration", help="Name of Project")

parser.add_argument("--step_number", type=int, default=5000, help="step_number")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--image_size", type=int, default=512,  help="image_size")
parser.add_argument("--num_unet", type=int, default=1,  help="num_unet")
parser.add_argument("--objective", type=str, default='pred_res',  help="[pred_res_noise,pred_x0_noise,pred_noise,pred_res]")
parser.add_argument("--test_res_or_noise", type=str, default='res',  help="[res_noise,res,noise]")
parser.add_argument("--lr", type=float, default=0.0003, help="learning_rate")
parser.add_argument("--sampling_timesteps", type=int, default=1,  help="sampling_timesteps")

def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def cycle(dl): 
    while True: 
        for data in dl:
            yield data

def divisible_by(numer, denom):
    return (numer % denom) == 0

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
def create_empty_json(json_path):
    with open(json_path, 'w') as file:
        pass
 
def remove_json(json_path):
    os.remove(json_path)

def write_json(json_path,item):
    with open(json_path, 'a+', encoding='utf-8') as f:
        line = json.dumps(item)
        f.write(line+'\n')

def readline_json(json_path,key=None):
    data = []
    with open(json_path, 'r') as f:
        items = f.readlines()
    if key is not None:
        for item in items:
            data.append(json.loads(item)[key])
        return np.asarray(data).mean()
    else:
        for item in items:
            data.append(json.loads(item))
        return data

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_folder,
        eval_folder,
        train_num_steps = 100000,
        train_batch_size = 1,
        save_and_sample_every = 5000,
        save_best_and_latest_only = True,
        calculate_metric = True,
        results_folder = './results/',
        gradient_accumulate_every = 1,
        *,
        augment_horizontal_flip = True,
        train_lr = 8e-5,
        ema_update_every = 1,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_row = 10,
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        max_grad_norm = 1.,
    ):
        super().__init__()

        self.accelerator = accelerate.Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        self.model = diffusion_model 
        is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.save_row = save_row
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        self.train_folder = train_folder
        self.eval_folder = eval_folder

        self.ds_hazy = train_dataset(train_folder,task_folder='Dehazing',image_size = 256)
        self.ds_light = train_dataset(train_folder,task_folder='Enlighening',image_size = 256)
        self.ds_rain = train_dataset(train_folder,task_folder='Deraining',image_size = 256)
        self.ds_snow = train_dataset(train_folder,task_folder='Desnowing',image_size = 256)
        self.ds_blur = train_dataset(train_folder,task_folder='Deblur',image_size = 256)
 
        self.dl_hazy = cycle(self.accelerator.prepare(DataLoader(self.ds_hazy, batch_size = 1)))
        self.dl_light = cycle(self.accelerator.prepare(DataLoader(self.ds_light, batch_size = 1)))
        self.dl_rain = cycle(self.accelerator.prepare(DataLoader(self.ds_rain, batch_size = 1)))
        self.dl_snow = cycle(self.accelerator.prepare(DataLoader(self.ds_snow, batch_size = 1)))
        self.dl_blur = cycle(self.accelerator.prepare(DataLoader(self.ds_blur, batch_size = 1)))
  
        if self.accelerator.is_main_process:
            self.accelerator.print('Training Samplies : (hazy :{})'.format(len(self.ds_hazy)))
            self.accelerator.print('                  : (light:{})'.format(len(self.ds_light)))
            self.accelerator.print('                  : (rain :{})'.format(len(self.ds_rain)))
            self.accelerator.print('                  : (snow :{})'.format(len(self.ds_snow)))
            self.accelerator.print('                  : (blur :{})'.format(len(self.ds_blur)))

        self.ds_eval_hazy = test_dataset(eval_folder,task_folder='Dehazing')
        self.ds_eval_light = test_dataset(eval_folder,task_folder='Enlighening')
        self.ds_eval_rain = test_dataset(eval_folder,task_folder='Deraining')
        self.ds_eval_snow = test_dataset(eval_folder,task_folder='Desnowing')
        self.ds_eval_blur = test_dataset(eval_folder,task_folder='Deblur')
 
        self.dl_eval_hazy = self.accelerator.prepare(DataLoader(self.ds_eval_hazy, batch_size = 1))
        self.dl_eval_light = self.accelerator.prepare(DataLoader(self.ds_eval_light, batch_size = 1))
        self.dl_eval_rain = self.accelerator.prepare(DataLoader(self.ds_eval_rain, batch_size = 1))
        self.dl_eval_snow = self.accelerator.prepare(DataLoader(self.ds_eval_snow, batch_size = 1))
        self.dl_eval_blur = self.accelerator.prepare(DataLoader(self.ds_eval_blur, batch_size = 1))

        if self.accelerator.is_main_process:
            self.accelerator.print('Validation Samplies : (hazy :{})'.format(len(self.ds_eval_hazy)))
            self.accelerator.print('                    : (light:{})'.format(len(self.ds_eval_light)))
            self.accelerator.print('                    : (rain :{})'.format(len(self.ds_eval_rain)))
            self.accelerator.print('                    : (snow :{})'.format(len(self.ds_eval_snow)))
            self.accelerator.print('                    : (blur :{})'.format(len(self.ds_eval_blur)))

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.train_num_steps = train_num_steps
        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.calculate_metric = calculate_metric and self.accelerator.is_main_process

        if save_best_and_latest_only:
            self.best_metric = 0

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone = None):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        checkpoint_save_path = os.path.join(self.results_folder,f'model-{milestone}')
        if not os.path.exists(checkpoint_save_path):
            os.makedirs(checkpoint_save_path)
        torch.save(data, checkpoint_save_path + '/' +  f'model-{milestone}.pt')

    def load(self, milestone= None):
        accelerator = self.accelerator
        device = accelerator.device
        checkpoint_save_path = os.path.join(self.results_folder,f'model-{milestone}')
        data = torch.load('./ckpt/pretrained.pt', map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt']) 
        self.ema.load_state_dict(data["ema"])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def cal_psnr(self,img_ref, img_gen, data_range = 255.0):
        mse = np.mean((img_ref.astype(np.float32)/data_range - img_gen.astype(np.float32)/data_range) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def cal_ssim(self,img_ref, img_gen):
        ssim_val = 0
        for i in range(img_ref.shape[-1]):
            ssim_val = ssim_val + structural_similarity(img_ref[:,:,i], img_gen[:,:,i])
        return ssim_val/img_ref.shape[-1]
 
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        track_metric_json_path = os.path.join(self.results_folder,'metric.json') 
        if self.accelerator.is_main_process: 
            create_empty_json(track_metric_json_path)
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                hazy_name,hazy_target,hazy_input = next(self.dl_hazy)
                light_name,light_target,light_input = next(self.dl_light)
                rain_name,rain_target,rain_input = next(self.dl_rain)
                snow_name,snow_target,snow_input = next(self.dl_snow)
                blur_name,blur_target,blur_input = next(self.dl_blur)
                file_name = hazy_name + light_name + rain_name + snow_name + blur_name
                file_target = torch.cat([hazy_target,light_target,rain_target,snow_target,blur_target],dim=0)
                file_input = torch.cat([hazy_input,light_input,rain_input,snow_input,blur_input],dim=0)
                with self.accelerator.autocast():
                    loss = self.model(img = [file_target.to(device),file_input.to(device)])[0]
                    self.accelerator.backward(loss)
                    pbar.set_description(f'loss: {loss.item():.4f}')
                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    accelerator.wait_for_everyone()

                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    self.test(dataloader = self.dl_eval_rain, degradation = 'Deraining')    
                    self.test(dataloader = self.dl_eval_light, degradation = 'Enlighening')   
                    self.test(dataloader = self.dl_eval_blur, degradation = 'Deblur')                   
                    self.test(dataloader = self.dl_eval_snow, degradation = 'Desnowing') 
                    self.test(dataloader = self.dl_eval_hazy, degradation = 'Dehazing') 
                     
                if self.accelerator.is_main_process: 
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every): 
                        write_json(track_metric_json_path,f'model-{self.step} : ')
                        degradation_types = ['Enlighening', 'Desnowing', 'Deraining', 'Deblur', 'Dehazing'] 
                        for degradation in degradation_types:  
                            json_path = os.path.join(self.results_folder,f'model-{self.step}') + '/{}.json'.format(degradation)                          
                            psnr_val,ssim_val = readline_json(json_path,'psnr'),readline_json(json_path,'ssim')
                            accelerator.print('{}      -> (PSNR/SSIM) : {:.6f}/{:.6f} '.format(degradation,psnr_val,ssim_val))
                            write_json(track_metric_json_path,'{}      -> (PSNR/SSIM) : {:.6f}/{:.6f} '.format(degradation,psnr_val,ssim_val))

                accelerator.wait_for_everyone() 

                self.ema.update()
                if self.accelerator.is_main_process:
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):  
                        accelerator.print('save model checkpoint')  
                        self.save(self.step)  

                accelerator.wait_for_everyone()      
                self.step += 1
                pbar.update(1)
                
        accelerator.print('Training complete')

    def test(self,dataloader,degradation): 
        self.accelerator.wait_for_everyone() 
        if self.accelerator.is_main_process: 
            start_time = time.time()
            save_json_dir = os.path.join(self.results_folder,f'model-{self.step}')
            create_folder(save_json_dir) 
            save_json_path = save_json_dir + '/{}.json'.format(degradation)
            create_empty_json(save_json_path)
        self.accelerator.wait_for_everyone()
        save_json_path = os.path.join(self.results_folder,f'model-{self.step}') + '/{}.json'.format(degradation)
        self.ema.model.eval()
        for batch_id,batch in enumerate(dataloader): 
            name_path,image_tf,condi_tf = batch
            img_gen = self.ema.model.sample(condi_tf.to(self.device)) 
            for element_id in range(len(name_path)):
                image_np_ref = self.tf2img(image_tf[element_id,:,:,].unsqueeze(0))  
                image_np_gen = self.tf2img(img_gen[element_id,:,:,].unsqueeze(0))   

                psnr_val = self.cal_psnr(image_np_ref,image_np_gen) 
                ssim_val = self.cal_ssim(image_np_ref,image_np_gen)

                data_dump_info = {
                    'file_path' : name_path[element_id],
                    'psnr' : psnr_val,
                    'ssim' : ssim_val, 
                }            
                print(batch_id,name_path,'PSNR / SSIM : {:.6f} : {:.6f}'.format(psnr_val,ssim_val))
                write_json(save_json_path,data_dump_info)
                image_save_dir = os.path.join(self.results_folder,f'model-{self.step}',name_path[element_id].split('/')[-5],name_path[element_id].split('/')[-4])
                create_folder(image_save_dir)
                imageio.imwrite(os.path.join(image_save_dir,name_path[element_id].split('/')[-1]),image_np_gen)

        if self.accelerator.is_main_process: 
            end_time = time.time()
            test_time_consuming = end_time - start_time        
            self.accelerator.print('Test_time_consuming : {:.6} s'.format(test_time_consuming))

        self.accelerator.wait_for_everyone()   

    def tf2np(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        return image_np

    def tf2img(self,image_tf):
        image_np = self.tf2np(torch.clamp(image_tf,min=0.,max=1.))
        image_np = (image_np * 255).astype(np.uint8)
        return image_np

def train_ddp_accelerate(args):
    train_folder = ''
    eval_folder = ''
    print('Procedure Running: ',args.project_description)
    image_size = 256
    num_unet = 1
    objective = 'pred_res'
    ddim_sampling_eta = 0.0
    test_res_or_noise = "res"  
    sum_scale = 0.01
    delta_end = 2.0e-3 
    condition = True
    sampling_timesteps = 8
    model = UnetRes(dim=64, dim_mults=(1, 2, 4, 8),num_unet=num_unet, condition=condition, objective=objective, test_res_or_noise = test_res_or_noise)
    diffusion = ResidualDiffusion(model,image_size=image_size, timesteps=1000,delta_end = delta_end,sampling_timesteps=sampling_timesteps, objective=objective,ddim_sampling_eta= ddim_sampling_eta,loss_type='l1',condition=condition,sum_scale=sum_scale,test_res_or_noise = test_res_or_noise)
    diffusion_process_trainer = Trainer(
        diffusion_model = diffusion,
        train_folder = train_folder,
        eval_folder = eval_folder,
        train_num_steps = 500000,
        train_batch_size = 16,
        save_and_sample_every = 5000,
        save_best_and_latest_only = True,
        calculate_metric = True,
        results_folder = './save_folder',
        gradient_accumulate_every = 1,
    )
    diffusion_process_trainer.load() 
    diffusion_process_trainer.train() 
    print('Procedure Termination: (Finished)')

if __name__ == '__main__': 
    args = parser.parse_args() 
    set_seed(0)
    train_ddp_accelerate(args) 