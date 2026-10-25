#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch       
import torch.nn as nn                                                      

@dataclass
class RGBDACTConfig:
    image_h:int=252; image_w:int=336; patch:int=14; num_joints:int=46; chunk_size:int=30
    dino_dim:int=384; depth_dim:int=128; d_model:int=384; nhead:int=8
    decoder_layers:int=4; cvae_layers:int=3; ffn_dim:int=1536; latent_dim:int=32
    dropout:float=.1; kl_weight:float=.01
    @property
    def grid_h(self): return self.image_h//self.patch
    @property
    def grid_w(self): return self.image_w//self.patch
    @property
    def n_spatial(self): return self.grid_h*self.grid_w

class DinoV2Spatial(nn.Module):
    def __init__(self,cfg:RGBDACTConfig,freeze=True):
        super().__init__(); self.cfg=cfg; self.frozen=freeze
        if cfg.image_h%cfg.patch or cfg.image_w%cfg.patch: raise ValueError('H/W must divide by 14')
        try: self.backbone=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True)
        except TypeError: self.backbone=torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
        if freeze:
            for p in self.backbone.parameters(): p.requires_grad_(False)
    def set_last_n_blocks_trainable(self,n:int):
        for p in self.backbone.parameters(): p.requires_grad_(False)
        if n<=0: self.frozen=True; return
        blocks=getattr(self.backbone,'blocks',None)
        if blocks is None: raise RuntimeError('DINO backbone lacks .blocks')
        for b in list(blocks)[-n:]:
            for p in b.parameters(): p.requires_grad_(True)
        if getattr(self.backbone,'norm',None) is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad_(True)
        self.frozen=False
    def forward(self,x):
        if self.frozen:
            with torch.no_grad(): out=self.backbone.forward_features(x)
        else: out=self.backbone.forward_features(x)
        if isinstance(out,dict):
            if 'x_norm_patchtokens' in out: tok=out['x_norm_patchtokens']
            elif 'x_prenorm' in out: tok=out['x_prenorm'][:,-self.cfg.n_spatial:,:]
            else: raise RuntimeError(f'unsupported DINO keys {list(out.keys())}')
        else: tok=out[:,-self.cfg.n_spatial:,:]
        if tok.shape[1]!=self.cfg.n_spatial: raise RuntimeError(f'DINO tokens {tok.shape[1]} != {self.cfg.n_spatial}')
        return tok

class DepthPatchEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.patch=nn.Conv2d(1,64,kernel_size=cfg.patch,stride=cfg.patch,bias=False)
        self.refine=nn.Sequential(nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,cfg.depth_dim,3,padding=1),nn.GroupNorm(8,cfg.depth_dim),nn.GELU(),nn.Conv2d(cfg.depth_dim,cfg.depth_dim,3,padding=1),nn.GELU())
    def forward(self,d): return self.refine(self.patch(d)).flatten(2).transpose(1,2)

class CVAEStyleEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.cls=nn.Parameter(torch.zeros(1,1,cfg.d_model)); self.q_proj=nn.Sequential(nn.Linear(cfg.num_joints,cfg.d_model),nn.GELU(),nn.Linear(cfg.d_model,cfg.d_model)); self.a_proj=nn.Linear(cfg.num_joints,cfg.d_model); self.pos=nn.Parameter(torch.randn(1,2+cfg.chunk_size,cfg.d_model)*.02)
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.nhead,cfg.ffn_dim,cfg.dropout,activation='gelu',batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,cfg.cvae_layers); self.mu=nn.Linear(cfg.d_model,cfg.latent_dim); self.logvar=nn.Linear(cfg.d_model,cfg.latent_dim)
    def forward(self,q,a):
        b=q.shape[0]; x=torch.cat([self.cls.expand(b,-1,-1),self.q_proj(q).unsqueeze(1),self.a_proj(a)],1)+self.pos; h=self.encoder(x)[:,0]; mu=self.mu(h); lv=self.logvar(h).clamp(-10,10); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return z,mu,lv

class RGBDACT(nn.Module):
    def __init__(self,cfg:RGBDACTConfig,freeze_dino=True):
        super().__init__(); self.cfg=cfg; self.rgb_encoder=DinoV2Spatial(cfg,freeze_dino); self.depth_encoder=DepthPatchEncoder(cfg)
        self.fusion=nn.Sequential(nn.Linear(cfg.dino_dim+cfg.depth_dim,cfg.d_model),nn.LayerNorm(cfg.d_model),nn.GELU()); self.q_token=nn.Sequential(nn.Linear(cfg.num_joints,cfg.d_model),nn.GELU(),nn.Linear(cfg.d_model,cfg.d_model)); self.z_token=nn.Linear(cfg.latent_dim,cfg.d_model); self.cvae=CVAEStyleEncoder(cfg)
        self.row_pos=nn.Parameter(torch.randn(1,cfg.grid_h,1,cfg.d_model)*.02); self.col_pos=nn.Parameter(torch.randn(1,1,cfg.grid_w,cfg.d_model)*.02); self.special_pos=nn.Parameter(torch.randn(1,2,cfg.d_model)*.02); self.action_queries=nn.Parameter(torch.randn(1,cfg.chunk_size,cfg.d_model)*.02); self.q_query_mlp=nn.Sequential(nn.Linear(cfg.num_joints,cfg.d_model),nn.LayerNorm(cfg.d_model),nn.GELU(),nn.Linear(cfg.d_model,cfg.d_model),nn.LayerNorm(cfg.d_model))
        layer=nn.TransformerDecoderLayer(cfg.d_model,cfg.nhead,cfg.ffn_dim,cfg.dropout,activation='gelu',batch_first=True,norm_first=True); self.decoder=nn.TransformerDecoder(layer,cfg.decoder_layers); self.out=nn.Linear(cfg.d_model,cfg.num_joints); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def _spatial_pos(self,b): return (self.row_pos+self.col_pos).reshape(1,self.cfg.n_spatial,self.cfg.d_model).expand(b,-1,-1)
    def encode_observation(self,rgb,depth,q,z):
        rt=self.rgb_encoder(rgb); dt=self.depth_encoder(depth)
        if rt.shape[:2]!=dt.shape[:2]: raise RuntimeError(f'grid mismatch {rt.shape} {dt.shape}')
        fused=self.fusion(torch.cat([rt,dt],-1))+self._spatial_pos(rt.shape[0]); specials=torch.cat([self.q_token(q).unsqueeze(1),self.z_token(z).unsqueeze(1)],1)+self.special_pos; mem=torch.cat([fused,specials],1)
        return mem,{'rgb_tokens':tuple(rt.shape),'depth_tokens':tuple(dt.shape),'fused_tokens':tuple(fused.shape),'memory':tuple(mem.shape)}
    def forward(self,rgb,depth,q,actions:Optional[torch.Tensor]=None):
        b=q.shape[0]
        if self.training and actions is not None: delta_actions=actions-q.unsqueeze(1); z,mu,lv=self.cvae(q,delta_actions)
        else:
            z=torch.zeros(b,self.cfg.latent_dim,device=q.device,dtype=q.dtype); mu=torch.zeros_like(z); lv=torch.zeros_like(z)
        mem,sh=self.encode_observation(rgb,depth,q,z); qcond=self.q_query_mlp(q); queries=self.action_queries.expand(b,-1,-1)+qcond.unsqueeze(1); h=self.decoder(queries,mem); delta=self.out(h); pred=q.unsqueeze(1)+delta; return {'pred':pred,'delta':delta,'mu':mu,'logvar':lv,'debug_shapes':sh}

def kl_divergence(mu,logvar): return -.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
