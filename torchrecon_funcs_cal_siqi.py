#conda activate deeplearning2
#####import libraries##############
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io as scio
from torch.autograd import Variable
from torch.autograd import grad
from torch import Tensor
import scipy.ndimage as scimg

def loss_M(M,Q,Nb,Bpr,Bpi,thresh):
    Qr=Q[0][0:Nb]
    Qr=Qr[None,:]
    Qi=Q[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    Mgr=torch.mm((torch.mm(Qrt,Qr)).view(1,Nb**2)+(torch.mm(Qit,Qi)).view(1,Nb**2),Bpr)+torch.mm((torch.mm(Qit,Qr)).view(1,Nb**2)-(torch.mm(Qrt,Qi)).view(1,Nb**2),Bpi)
    #imaginary part is negligible (should be 0, but non zero due to numerical artifact, ignore)
    Mgr=Mgr/torch.sum(Mgr)
    #Mgr[Mgr<0.0002506] = 0
    #Mgr[Mgr<0.00037] = 0
    Mgr[Mgr<0.00032] = 0
    Mgr = Mgr/torch.sum(Mgr)
    Mgr[Mgr<thresh] = 0
    #Mgr = Mgr/torch.sum(Mgr)
    
    #image - scale*bkg
    # scale=Q[0][-1]
    # M=M-scale*bkg
    M[M<0]=0
    M = M*5000
    Mgr = Mgr*5000
    #Mgr = Mgr/torch.max(Mgr)*torch.max(M)
    return (torch.sum(torch.abs(M.view([1,64**2])-Mgr)**2))
    #return (torch.sum(torch.abs(M.reshape([1,64**2])-Mgr)))**2

def loss_M_half(M,Q,Nb,Bpr,Bpi):
    Qr=Q[0][0:Nb]
    Qr=Qr[None,:]
    Qi=Q[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    Mgr=torch.mm((torch.mm(Qrt,Qr)).view(1,Nb**2)+(torch.mm(Qit,Qi)).view(1,Nb**2),Bpr)+torch.mm((torch.mm(Qit,Qr)).view(1,Nb**2)-(torch.mm(Qrt,Qi)).view(1,Nb**2),Bpi)
    #imaginary part is negligible (should be 0, but non zero due to numerical artifact, ignore)
    Mgr = Mgr.view((64,64))
    Mgr = Mgr[:32,:]
    Mgr=Mgr/torch.sum(Mgr)
    #Mgr[Mgr<0.0002506] = 0
    #Mgr[Mgr<0.00037] = 0
    Mgr[Mgr<0.00032] = 0
    Mgr = Mgr/torch.sum(Mgr)
    #image - scale*bkg
    # scale=Q[0][-1]
    # M=M-scale*bkg
    M[M<0]=0
    M = M[:32,:]
    M = M/torch.sum(M)
    return (torch.sum(torch.abs(M-Mgr)**2))

def loss_M_32(M,Q,Nb,Bpr,Bpi):
    Qr=Q[0][0:Nb]
    Qr=Qr[None,:]
    Qi=Q[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    Mgr=torch.mm((torch.mm(Qrt,Qr)).view(1,Nb**2)+(torch.mm(Qit,Qi)).view(1,Nb**2),Bpr)+torch.mm((torch.mm(Qit,Qr)).view(1,Nb**2)-(torch.mm(Qrt,Qi)).view(1,Nb**2),Bpi)
    #imaginary part is negligible (should be 0, but non zero due to numerical artifact, ignore)
    Mgr = Mgr.view((32,32))
    Mgr=Mgr/torch.sum(Mgr)
    #Mgr[Mgr<0.0002506] = 0
    #Mgr[Mgr<0.00037] = 0
    Mgr[Mgr<0.00032] = 0
    Mgr = Mgr/torch.sum(Mgr)
    #image - scale*bkg
    # scale=Q[0][-1]
    # M=M-scale*bkg
    M[M<0]=0
    M = M/torch.sum(M)
    return (torch.sum(torch.abs(M-Mgr)**2))

def loss_w(spec, Q, alphaw_r, alphaw_i, Nb):
    #spec is already smoothed, baseline subtracted and zero thresholded
    Qr=Q[0][0:Nb]
    #print(Qr)
    #print(Q[0][-1])
    Qr=Qr[None,:] #[1x36]
    Qi=Q[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    eVshift = Q[0][-1]*50
    eV_idx = np.arange(2801)
    eV_idx = Tensor(eV_idx)
    eVshift_idx=torch.argmin((eV_idx-eVshift)**2)
    #print(Qi)
    #eVshift_idx = Tensor([eVshift*50])
    #eVshift_idx.requires_grad= True
   
    if eVshift_idx>1340: 
        eVshift_idx=40#40 is found empirically
    if eVshift_idx<-1340:
        eVshift_idx=-40
    print(eVshift_idx)
    EwEw = (torch.mm(Qr,alphaw_r)-torch.mm(Qi,alphaw_i))**2+(torch.mm(Qr,alphaw_i)+torch.mm(Qi,alphaw_r))**2
    EwEwlook = Tensor(EwEw[0][1200:-1200])#[1340:-1340])
    #EwEwlook = Tensor(EwEw[0][1340-eVshift_idx:-1340-eVshift_idx])
    EwEwlook=EwEwlook[None,:]
    EwEwlook = EwEwlook/torch.max(EwEwlook)
    
    #EwEwlook = EwEwlook*12.5
    #spec = spec*12.5
    spec = Tensor(spec[1200-eVshift_idx:-1200-eVshift_idx])#[1340:-1340])
    return torch.sum(torch.abs(14*(EwEwlook-spec))**2) + torch.sum(torch.abs(2*(EwEwlook-spec)/(spec+0.1))**2)

def loss_w_slide(spec, Q, alphaw_r, alphaw_i, Nb):
    #spec is already smoothed, baseline subtracted and zero thresholded
    q = np.zeros((1,2*Nb-1))
    q[0][1]=1
    Qq = Tensor(q)
    Qr=Qq[0][0:Nb]
    #print(Qr)
    #print(Q[0][-1])
    Qr=Qr[None,:] #[1x36]
    Qi=Qq[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    eVshift = Q[0]*50
    eV_idx = np.arange(2801)
    eV_idx = Tensor(eV_idx)
    eVshift_idx=torch.argmin((eV_idx-eVshift)**2)
    #eVshift_idx = Variable(eVshift_idx)# ,requires_grad=True)
    #print(Qi)
    #eVshift_idx = Tensor([eVshift*50])
    #eVshift_idx.requires_grad= True
   
    if eVshift_idx>1340: 
        eVshift_idx=40#40 is found empirically
    if eVshift_idx<-1340:
        eVshift_idx=-40
    print(eVshift_idx)
    EwEw = (torch.mm(Qr,alphaw_r)-torch.mm(Qi,alphaw_i))**2+(torch.mm(Qr,alphaw_i)+torch.mm(Qi,alphaw_r))**2
    EwEwlook = Tensor(EwEw[0][1200:-1200])#[1340:-1340])
    #EwEwlook = Tensor(EwEw[0][1340-eVshift_idx:-1340-eVshift_idx])
    EwEwlook=EwEwlook[None,:]
    EwEwlook = EwEwlook/torch.max(EwEwlook)
    
    #EwEwlook = EwEwlook*12.5
    #spec = spec*12.5
    spec = Tensor(spec[1200-eVshift_idx:-1200-eVshift_idx])#[1340:-1340])
    
    return torch.sum(torch.abs(14*(EwEwlook-spec))**2) + torch.sum(torch.abs(2*(EwEwlook-spec)/(spec+0.1))**2)


def loss_w_smooth(Q, alphaw_r, alphaw_i, Nb):
    #spec is already smoothed, baseline subtracted and zero thresholded
    Qr=Q[0][0:Nb]
    Qr=Qr[None,:] #[1x36]
    Qi=Q[0][Nb:2*Nb-1]
    Qi=Qi[None,:]
    x=Tensor([0])
    Qi=torch.cat([x[:,None],Qi],1)
    Qrt=torch.transpose(Qr,0,1)
    Qit=torch.transpose(Qi,0,1)
    eVshift = 0 #Q[0][-1]
    eVshift_idx = int(eVshift*50)  
    EwEw = (torch.mm(Qr,alphaw_r)-torch.mm(Qi,alphaw_i))**2+(torch.mm(Qr,alphaw_i)+torch.mm(Qi,alphaw_r))**2
    EwEwlook = Tensor(EwEw[0][1340-eVshift_idx:-1340-eVshift_idx])
    EwEwlook=EwEwlook[None,:]
    EwEwlook = EwEwlook/torch.max(EwEwlook)

    Ew1=torch.zeros(EwEwlook.size())
    for j in range(1,EwEwlook.size(1)):
        Ew1[0][j]=EwEwlook[0][j]-EwEwlook[0][j-1]
        
    Ew2=torch.zeros(EwEwlook.size())
    for j in range(1,EwEwlook.size(1)):
        Ew2[0][j]=Ew1[0][j]-Ew1[0][j-1]
    
    return torch.sum(torch.abs(Ew2)**2)

def loss_Q(Q, Nb):
    return torch.sum(torch.abs(Q[0][0:2*Nb-1]))

def loss_tot(M,Q,Nb,Bpr,Bpi,spec, alphaw_r, alphaw_i,w1,w2,w3):
#     print(w1*loss_M(M,Q,Nb,Bpr,Bpi),w2*loss_t(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),w3*loss_w(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),loss_Q(Q))
    return 0.1*loss_Q(Q,Nb)+w1*loss_M(M,Q,Nb,Bpr,Bpi)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb)+w3*loss_w_smooth(Q, alphaw_r, alphaw_i,Nb)

def loss_tot_noQ(M,Q,Nb,Bpr,Bpi,spec, alphaw_r, alphaw_i,w1,w2,w3,thresh):
#     print(w1*loss_M(M,Q,Nb,Bpr,Bpi),w2*loss_t(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),w3*loss_w(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),loss_Q(Q))
    return w1*loss_M(M,Q,Nb,Bpr,Bpi,thresh)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb) #+w3*loss_w_smooth(Q, alphaw_r, alphaw_i,Nb)

def loss_tot_32(M,Q,Nb,Bpr,Bpi,spec, alphaw_r, alphaw_i,w1,w2):
#     print(w1*loss_M(M,Q,Nb,Bpr,Bpi),w2*loss_t(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),w3*loss_w(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),loss_Q(Q))
    return w1*loss_M_32(M,Q,Nb,Bpr,Bpi)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb)#+w3*loss_w_smooth(Q, alphaw_r, alphaw_i,Nb)


def loss_tot_half(M,Q,Nb,Bpr,Bpi,spec, alphaw_r, alphaw_i,w1,w2,w3):
#     print(w1*loss_M(M,Q,Nb,Bpr,Bpi),w2*loss_t(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),w3*loss_w(Q,alpha_vn,ts,tn,wm,Nt,Nw,Nb),loss_Q(Q))
    return 0.1*loss_Q(Q,Nb)+w1*loss_M_half(M,Q,Nb,Bpr,Bpi)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb)+w3*loss_w_smooth(Q, alphaw_r, alphaw_i,Nb)

# def loss_half(M,Q,Nb,Bpr,Bpi,spec, alphaw_r, alphaw_i,w1,w2):
#     return w1*loss_M_half(M,Q,Nb,Bpr,Bpi)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb)

def loss_half(M,Q,Nb,Bpr,Bpi,spec,alphaw_r, alphaw_i,w1,w2):
    return w1*loss_M_half(M,Q,Nb,Bpr,Bpi)+w2*loss_w(spec, Q, alphaw_r, alphaw_i,Nb)

def rebin(arr, n=2):
    sy, sx = arr.shape
    shape = (int(sy/n), int(n), int(sx/n), int(n))
    return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)

def pix_2_eV(pixel, pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = 512):
    '''
    pixel: array of pixel indices
    pixpereV: calibration pixels per eV
    spectral_hw: (optional) central energy if you have absolutes
    spectra_pix0: where to put w = 0, 512 is at the center for 1024pix spectrometer 
    '''
    spectra_dhwdpix = 1./pixpereV
    spectra_hws_eV = spectra_hw0 + spectra_dhwdpix*(pixel -spectra_pix0)
    return spectra_hws_eV


# %Copyright (c) 2012, Thomas C. O'Haver
# % 
# % Permission is hereby granted, free of charge, to any person obtaining a copy
# % of this software and associated documentation files (the "Software"), to deal
# % in the Software without restriction, including without limitation the rights
# % to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# % copies of the Software, and to permit persons to whom the Software is
# % furnished to do so, subject to the following conditions:
# % 
# % The above copyright notice and this permission notice shall be included in
# % all copies or substantial portions of the Software.
# % 
# % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# % IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# % FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# % AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# % LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# % OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# % THE SOFTWARE.

def sa(Y,smoothwidth):
    w=int(smoothwidth + 0.5)
    SumPoints=np.nansum(Y[0:w])
    s=np.zeros(len(Y)) + SumPoints
    halfw=int(w/2 + 0.5)

    L=len(Y)
    for k in np.arange(0,L-w): 
        if k+halfw+1 < L-w:
            s[k+halfw+1]=SumPoints
        SumPoints=SumPoints-Y[k]
        SumPoints=SumPoints+Y[k+w]
    s[L-w-1:L]= s[L-w-1] - Y[L-w-1] + Y[L-w-1+halfw] #np.nansum(Y[L-w+halfw:L])
    SmoothY=s/w
    return SmoothY

def fastsmooth1(Y,w):
    SmoothY=sa(Y,w);
    return SmoothY