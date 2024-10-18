import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
from utils import diff2clf, clf2diff, normalize
from PIL import Image
from tqdm import *
import math
from torchvision import transforms

data_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a).float().to(device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,)+(1,)*(len(x_shape)-1))
    return out


def save_image(x):
    """
    x:[B,C,H,W]
    [-1,1]
    """
    x = (x + 1)/2
    x = (x[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)


def get_ddim_steps(total_time_steps,sample_steps,strength):
    step = total_time_steps//sample_steps
    ddim_steps = np.arange(0,total_time_steps-1,step)
    ddim_steps = ddim_steps[:int(sample_steps*strength)+1]
    ddim_steps = np.flip(ddim_steps)
    return ddim_steps


def threshold_percent_area(pure_area, threshold_percent):
    B, H, W = pure_area.shape
    flattened_pure_area = pure_area.view(B, -1)  
    result = torch.zeros_like(flattened_pure_area)
    k = int(H * W * threshold_percent) 
    for i in range(B):
        topk_values, topk_indices = torch.topk(flattened_pure_area[i], k)
        result[i][topk_indices] = 1
    result = result.reshape(B, H, W)
    return result


class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion,strength_a,strength_b, is_imagenet,forward_noise_steps,threshold,threshold_percent,ddim_steps,device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.device = device
        self.is_imagenet = is_imagenet
        self.strength_a = strength_a
        self.strength_b = strength_b
        self.forward_noise_steps = forward_noise_steps

        self.num_train_timesteps = 1000

        self.sample_steps = ddim_steps
        self.timesteps = get_ddim_steps(self.num_train_timesteps,self.sample_steps,self.strength_a)
        self.eta=0.0

        self.betas = get_beta_schedule(1e-4, 2e-2, 1000)
        self.alphas = 1. - self.betas
        self.sqrt_alphas = np.sqrt(self.alphas) 
        self.sqrt_one_minus_alphas = np.sqrt(1. - self.alphas)
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)# \bar{alpha}_t
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # \bar{alpha}_t-1
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas*(1.0-self.alphas_cumprod_prev) / (1.0-self.alphas_cumprod) # \tilde{beta}_t
        # self.logvar = np.log(np.maximum(self.posterior_variance, 1e-20))

        self.activations = {}
        self.threshold = threshold
        self.threshold_percent = threshold_percent

    def compute_attention_map(self,activation):
        G_2sum = torch.sum(torch.abs(activation)**2, dim=1) 
        return G_2sum

    def get_activation(self,name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    
    def scale(self,attention_map):
        b,h,w = attention_map.shape
        attention_map = attention_map.reshape(b,-1)
        attention_map =  (attention_map - torch.min(attention_map,dim=1,keepdim=True).values) / (torch.max(attention_map,dim=1,keepdim=True).values - torch.min(attention_map,dim=1,keepdim=True).values)
        attention_map = attention_map.reshape(b,h,w)
        return attention_map

    def get_mask(self,x):
        batch_size,c,h,w = x.shape

        self.clf.block1.layer[-1].relu2.register_forward_hook(self.get_activation('block1'))
        self.clf.block2.layer[-1].relu2.register_forward_hook(self.get_activation('block2'))
        self.clf.block3.layer[-1].relu2.register_forward_hook(self.get_activation('block3'))
        self.clf.relu.register_forward_hook(self.get_activation('block4'))


        _ = self.clf(data_normalize(x))
        pure_area = torch.zeros(size=(batch_size,h,w))
        for i, block in enumerate(['block1', 'block2', 'block3','block4']):
            attention_map = self.compute_attention_map(self.activations[block]).cpu()
            attention_map = attention_map.unsqueeze(1) 
            attention_map = F.interpolate(attention_map, size=(h, w), mode='bilinear', align_corners=False)

            attention_map = attention_map.squeeze(1)

            attention_map = self.scale(attention_map)

            pure_area[attention_map>pure_area] = attention_map[attention_map>pure_area]

        #### method1
        pure_area[pure_area>=self.threshold] = 1
        pure_area[pure_area<self.threshold] = 0
        #### method2
        # pure_area = threshold_percent_area(pure_area,threshold_percent=self.threshold_percent)

        return pure_area.unsqueeze(1).cuda()



    def diffuse_t_steps(self, x0, t):
        # x is a torch tensor of shape (B,C,H,W)
        # t is a interger range from 0 to T-1
        alpha_bar = self.alphas_cumprod[t]
        xt = torch.sqrt(torch.tensor(alpha_bar)) * x0 + torch.sqrt(torch.tensor(1-alpha_bar)) * torch.randn_like(x0)
        # save_image(xt)
        return xt

    def diffuse_one_step(self,x,t):
        noise = torch.randn_like(x)
        return (
        extract(self.sqrt_alphas, t, x.shape) * x +
        extract(self.sqrt_one_minus_alphas, t, x.shape) * noise
    )


    def diffuse_one_step_from_now(self,x_t,t,steps):
        n = x_t.shape[0]
        for i in range(steps):
            x_t = self.diffuse_one_step(x_t, (torch.ones(n)*(t+i+1)).to(self.device))
        return x_t,t+steps

    def denoising_step(self,x, t):
        """
        Sample from p(x_{t-1} | x_t)
        """
        # instead of using eq. (11) directly, follow original implementation which,
        # equivalently, predicts x_0 and uses it to compute mean of the posterior
        t = (torch.ones(x.shape[0])*t).to(self.device)
        # 1. predict eps via model
        model_output = self.diffusion(x, t)
        # 2. predict clipped x_0
        # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
        pred_xstart = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape)*x -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
        pred_xstart = torch.clamp(pred_xstart, -1, 1)
        # 3. compute mean of q(x_{t-1} | x_t, x_0) (eq. (6))
        mean = (extract(self.posterior_mean_coef1, t, x.shape)*pred_xstart +
                extract(self.posterior_mean_coef2, t, x.shape)*x)

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        


        # sample - return mean for t==0
        noise = torch.randn_like(x)
        mask = 1-(t==0).float()
        mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
        sample = mean + mask*torch.sqrt(posterior_variance)*noise
        sample = sample.float()

        return pred_xstart,sample


    def denoise(self, x):

        mask = self.get_mask(x)
        time_steps_b = self.strength_b * self.num_train_timesteps
        n = x.shape[0]
        x_t = self.diffuse_t_steps(x,self.timesteps[0])
        for t, tau in list(zip(self.timesteps[:-1], self.timesteps[1:])):
            if not math.isclose(self.eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - self.alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - self.alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - self.alphas[t]
                sigma_t = self.eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(torch.tensor(self.alphas[0]))
            if tau >= time_steps_b:
                x_t_ori = self.diffuse_t_steps(x,t)
                x_t = x_t * mask + x_t_ori * (1.- mask)
                x_t,t = self.diffuse_one_step_from_now(x_t,t,steps=self.forward_noise_steps)
                
            pred_noise = self.diffusion(x_t,(torch.ones(n)*t).to(self.device))
            if self.is_imagenet:
                pred_noise, _ = torch.split(pred_noise, 3, dim=1)
            # first term of x_tau
            alphas_cumprod_tau = (extract(self.alphas_cumprod, (torch.ones(n)*tau).to(self.device), x.shape))
            sqrt_alphas_cumprod_tau = alphas_cumprod_tau ** 0.5
            alphas_cumprod_t = (extract(self.alphas_cumprod, (torch.ones(n)*t).to(self.device), x.shape))
            sqrt_alphas_cumprod_t = alphas_cumprod_t ** 0.5
            sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t) ** 0.5
            first_term = sqrt_alphas_cumprod_tau * (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t

            # second term of x_tau
            coeff = (1.0 - alphas_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise

            epsilon = torch.randn_like(x_t)
            x_t = first_term + second_term + sigma_t * epsilon
            

        x_0 = x_t
        return x_0

    def classify(self, x):
        logits = self.clf(x)
        return logits



    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        
        x_diff = self.denoise(x_diff)

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        # return logits
        return logits

    def get_img_logits(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        
        x_diff = self.denoise(x_diff)

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        # return logits
        return x_clf,logits

