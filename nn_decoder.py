import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Decoder(nn.Module):
    def __init__(self, unit_no, t_dim, k_dim, h_dim, p_dim, f_dim):
        super(PartNonLinear, self).__init__()
        self.unit_no = unit_no
        self.t_dim = t_dim
        self.h_dim = h_dim
        self.p_dim = p_dim
        self.k_dim = k_dim
        self.f_dim = f_dim
        
        self.featurize = nn.ModuleList([nn.Linear(self.t_dim,
                                                  self.f_dim) for i in range(self.unit_no)]).cuda()
        
        self.hidden1 = nn.ModuleList([nn.Linear(self.k_dim*self.f_dim,
                                               self.h_dim) for i in range(self.p_dim)]).cuda()
        self.hidden1_act = nn.ModuleList([nn.PReLU() for i in range(self.p_dim)]).cuda()
        
        self.output_layer = nn.ModuleList([nn.Linear(self.h_dim,
                                                    1) for i in range(self.p_dim)]).cuda()
        
    def forward(self, S, pix_units):
        
        F = torch.empty(S.shape[0], self.unit_no * self.f_dim).cuda()
        for n in range(self.unit_no):
            feat_n = self.featurize[n](S[:, n*self.t_dim : (n+1)*self.t_dim])
            F[:, n*self.f_dim : (n+1)*self.f_dim] = feat_n
        
        I = torch.empty(S.shape[0] , self.p_dim).cuda()
        
        for x in range(self.p_dim):
            unit_ids = pix_units[x]
            feat_ids = torch.empty((self.k_dim * self.f_dim))
            for i in range(self.k_dim):
                feat_ids[i*self.f_dim : (i+1)*self.f_dim] = torch.arange(self.f_dim) + unit_ids[i]*self.f_dim
            
            pix_feat = self.hidden1[x](F[:, feat_ids.long()])
            pix_feat = self.hidden1_act[x](pix_feat)

            out = self.output_layer[x](pix_feat)
            
            I[:, x] = out.reshape(-1)
            
        return I            