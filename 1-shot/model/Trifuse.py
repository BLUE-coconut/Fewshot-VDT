import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.transformer import *
from .base.correlation import Correlation,cosine_similarity


def ConvModule(in_ch,out_ch,kernel,padding=None,mode="sigmoid"):
    if(padding is None):
        padding = kernel // 2
    if(mode=="sigmoid"):
        conv = nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel,padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.Sigmoid()
        )
    elif(mode=="relu"):
        conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel,padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    elif(mode=="bn"):
        conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel,padding=padding),
            nn.BatchNorm2d(out_ch)
        )
    else:   
        conv = nn.Conv2d(in_ch,out_ch,kernel,padding=padding)

    return conv


class TriCorr_v2_6(nn.Module):
    def __init__(self,in_ch, mid_ch = None, out_ch = None, dropout=0.5) -> None:
        super().__init__()
        # multimodal,QV\QD\QT attention, SV\SD\ST attention, then cos&max -> prior map
        # only V for main modal
        self.in_ch = in_ch
        if (mid_ch is None):
            self.mid_ch = in_ch
        else:
            self.mid_ch = mid_ch
        
        if (out_ch is None):
            self.out_ch = self.mid_ch
        else:
            self.out_ch = out_ch

        self.conv_DT = ConvModule(in_ch=self.in_ch*2,out_ch=self.mid_ch,kernel=1)
        self.reduct_V = ConvModule(in_ch=self.in_ch,out_ch=self.mid_ch,kernel=1)
        self.MHA_V = MultiHeadedAttention(num_heads=8, d_model=self.mid_ch, dropout=dropout)

        self.Q_sim_conv = ConvModule(1,1,1,mode="conv")
        self.S_sim_conv = ConvModule(1,1,1,mode="conv")

        self.V_corr = ConvModule(self.mid_ch*2,self.out_ch,3)


    def forward(self, query, support):
        # features = [fV,fD,fT]
        QV = query[0]
        QD = query[1]
        QT = query[2]
        SV = support[0]
        SD = support[1]
        ST = support[2]
        

        fused_Trimaps = []

        QV = self.reduct_V(QV)
        SV = self.reduct_V(SV)
        sim = cosine_similarity(QV,SV)
        REQ = self.Q_sim_conv(sim)*QV
        Q_en_V = self.MHA_V(self.conv_DT(torch.cat([QD,QT],dim=1)),REQ,REQ)
        
        RES = self.S_sim_conv(sim)*SV
        S_en_V = self.MHA_V(self.conv_DT(torch.cat([SD,ST],dim=1)),RES,RES)
        
        fused_Trimaps.append(self.V_corr(torch.cat([Q_en_V,S_en_V],dim=1)))  

        return fused_Trimaps
    




