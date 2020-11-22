# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter
from functools import wraps
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def stable_kl(logit, target, epsilon=1e-6):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    return (p* (rp- ry) * 2).sum() / bs

def kl(input, target):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction='batchmean')
    return loss

def adv_project(grad, norm_type='inf', eps=1e-8):
    if norm_type == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
    elif norm_type == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
    return direction

def alum_perturbation(model, batch, embed, logits, args):
    noise = generate_noise(embed, batch[1], args.adv_epsilon)
    adv_embed = embed.data.detach() + noise
    for k in range(args.adv_k + 1):
        adv_logits, _ = model(batch=batch, embed = adv_embed)
        adv_loss = kl(adv_logits, logits.detach())
        if k != args.adv_k:
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            adv_direct = adv_project(noise + args.adv_eta * delta_grad, norm_type=args.adv_project_type, eps=args.adv_noise_gamma)
            adv_embed = (embed.data + adv_direct * args.adv_step_size).detach()
    adv_loss_f = kl(adv_logits, logits.detach())
    adv_loss_b = kl(logits.detach(), adv_logits.detach())
    adv_loss = (adv_loss_f + adv_loss_b) * args.adv_alpha
    
    return adv_loss

# def parameter_based_perturbation(model, batch, embed, logits, args):
    

def random_perturbation(model, batch, embed, logits, args):
    noise = generate_noise(embed, batch[1], args.adv_epsilon)
    random_pert_embed = embed.data.detach() + noise
    random_pert_logit, _ = model(batch=batch, embed=random_pert_embed)
    random_pert_loss_f = kl(random_pert_logit, logits)
    random_pert_loss_b = kl(logits, random_pert_logit)
    random_pert_loss = (random_pert_loss_f + random_pert_loss_b) * args.adv_alpha

    return random_pert_loss

class SmartPerturbation():
    def __init__(self,
                 epsilon=1e-6,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='2',
                 k=1):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon 
        # eta
        self.step_size = step_size
        self.K = k
        # sigma
        self.noise_var = noise_var 
        self.norm_p = norm_p

    def _norm_grad(self, grad):
        if self.norm_p == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction

    def forward(self, model,
                logits,
                input_ids,
                token_type_ids,
                attention_mask):

        logits = logits.detach()

        # init delta
        embed = model(input_ids, attention_mask, token_type_ids, return_embed = True)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            adv_logits = model(input_ids, attention_mask, token_type_ids, input_embeds=embed + noise)
            adv_loss = stable_kl(adv_logits, logits)
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            delta_grad = self._norm_grad(delta_grad)
            embed = embed + delta_grad * self.step_size
            embed = embed.detach()
            embed.requires_grad_()
        adv_logits = model(input_ids, attention_mask, token_type_ids, input_embeds=embed)
        adv_loss = stable_kl(logits, adv_logits)
        return adv_loss 

