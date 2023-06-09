from __future__ import print_function
import sys

sys.path.append('/reg/d/psdm/tmo/tmox51020/results/paris/')
sys.path.append('/reg/d/psdm/tmo/tmox51020/results/paris/recon')
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
from scipy.interpolate import interp1d
import sys
sys.path.append('/reg/neh/home/tdd14/modules/cart2pol/cart2pol')
import cart2pol
from cart2pol import PolarRebin
from torchrecon_funcs_clean import *
import os


## NEED THE VLS/FZP calibration
## what is the pixels per eV
pixpereV = 22
####UPDATE FILENAMES##############
savefiledir = '/reg/d/psdm/tmo/tmox51020/results/paris/recon/torch_test_clean/' #where you are saving the output files
runnum = 125 #116 
numseeds = 3 #number of random seeds
#array of preprocessed images with shape (# shots, 64,64)
mdatname= '/cds/home/p/paris/cVMI/examples/Ms_test_clean.npy'
#array of preprocessed spectra with shape (# shots, 1024)
    # processing is:
        #smoothing with fastsmooth1 function and baseline subtraction
        #roll the spectra so that the central frequency of the average spectra (w0) is at the center of the array (pixel = 512) 
specdatname = '/cds/home/p/paris/cVMI/examples/specs_test_clean.npy'
mdat = np.load(mdatname, allow_pickle = True)[()]
specdat = np.load(specdatname, allow_pickle = True)[()] 
basesdir = '/reg/d/psdm/tmo/tmox51020/results/paris/circularpol_vNbases_dipole_Feb/'
PR = PolarRebin('/reg/d/psdm/tmo/tmox51020/results/paris/recon/PR_c32_r32_th32.h5')  # this should be correct for cart2pol, pol2cart for 64x64

######################
#update mask for your experiment
X,Y=np.meshgrid(np.arange(1024),np.arange(1024))
rs=np.sqrt((X-512)**2+(Y-512)**2)
maskhm=np.zeros((1024,1024))
maskhm[(rs>325-30) & (rs<420)]=1
mask = rebin(rebin(maskhm,4),4)

####################
Ups = [float(sys.argv[1])]

#correction factor if the spectra calibration are off. 0 and 1 if calibration is correct
eshift = 0  
sc = 1

#smoothing applied to the vNbases image to try to make it match the measured
gfsig = 0.985 
gfcart = 0.25 
mfcart = 1

thresh = 0.0005 #small number to set everything below to zero for M/sum(M)

Bps = []
alphas = []
vNaxiss = []

Np = 128
N_w=6
N_t=6 
hbar=6.6e-16 #eV*s
Nb=N_t*N_w
mfn = 25
gfn = 15 


Up = Ups[0]  
filename = 'Bpbasis_Np' + str(Np) + '_Nw' + str(N_w) +'_Nt' + str(N_t) + '_Up' + str(Up)+'.npy'
br = np.load(basesdir+filename, allow_pickle=True)
Breal = br[()]['Breal']
Bimag = br[()]['Bimag']
breal = br[()]['breal']
bimag = br[()]['bimag']
vNaxis = br[()]['vNaxis']
alpha = br[()]['alpha']
Bp_basis = Breal+1j*Bimag

#rebin to 64x64
tempr = np.zeros((Breal.shape[0], 64**2))
tempi = np.zeros((Breal.shape[0], 64**2))
for i in range(len(Breal)):
    tempr[i,:] = np.reshape(rebin(np.reshape(Breal[i,:], [128,128])), [64**2])
    tempi[i,:] = np.reshape(rebin(np.reshape(Bimag[i,:], [128,128])), [64**2])

Breal = tempr
Bimag = tempi

alphw = np.zeros(alpha['t_sample'].shape, dtype = complex)
for i in range(len(alphw)):
    f = interp1d(vNaxis['t_sample'], alpha['t_sample'][i], fill_value = 0, bounds_error = False)
    N = 2801 #1024
    ts = np.linspace(vNaxis['t_sample'][0],-vNaxis['t_sample'][0], N)
    alphw[i] = np.fft.ifftshift(np.fft.ifft(f(ts)))


# alphw = np.zeros(alpha['t_sample'].shape, dtype = complex)
# for i in range(len(alphw)):
#     alphw[i] = np.fft.ifftshift(np.fft.ifft(alpha['t_sample'][i]))

alphaw_r=Variable(Tensor(np.real(alphw)))#real part
alphaw_i=Variable(Tensor(np.imag(alphw)))#imaginary part

Np = 64
for j in range(Breal.shape[0]):
    temp=np.reshape(Breal[j,:],[Np,Np])
    polimg = PR.cart2pol(temp,32,32)
    bb = np.zeros((64,64))
    bb[1:64,1:64]= PR.pol2cart(scimg.gaussian_filter(polimg,[gfsig,0]))
    Breal[j,:]=Variable(Tensor(np.reshape(bb*mask,newshape=(1,Np**2))))
    #temp=scimg.filters.gaussian_filter(temp,0.8889)
    #Bpr[j,:]=Variable(Tensor(np.reshape(temp,newshape=(1,Np**2))))
    temp=np.reshape(Bimag[j,:],[Np,Np])
    polimg = PR.cart2pol(temp,32,32)
    bb = np.zeros((64,64))
    bb[1:64,1:64]= PR.pol2cart(scimg.gaussian_filter(polimg,[gfsig,0]))
    Bimag[j,:]=Variable(Tensor(np.reshape(bb*mask,newshape=(1,Np**2))))
    #temp=scimg.filters.gaussian_filter(temp,0.8889)
    #Bpi[j,:]=Variable(Tensor(np.reshape(temp,newshape=(1,Np**2))))
    
Bpr=Variable(Tensor(Breal))
Bpi=Variable(Tensor(Bimag))

############################
#loop through the images in preprocessed array
for k in range(mdat.shape[0]):
    for t in range(numseeds):
        tt = np.copy(t)
        savefilename = os.path.join(savefiledir, 'pytorchrecon_'+str(k)+'_Up'+str(Ups[0])+'_seed'+str(tt)+'.npy')
        if os.path.isfile(savefilename):
            print(savefilename+' exists')
            continue
        else:
            torch.manual_seed(t)
        
            m = mdat[k]*mask
            spec = specdat[k]
            outdict = {}
            outdict['thresh'] = thresh
            outdict['gfsig'] = gfsig
            outdict['gfcart'] = gfcart
            outdict['mfn'] = mfn
            outdict['gfn'] = gfn
            outdict['eshift'] = eshift
            outdict['sc'] = sc
            outdict['mask'] = mask
            outdict['maskhm'] = maskhm

            M=Variable(Tensor(m))
            M=M/torch.sum(M)
            #M=M.transpose(0,1)
            t=Variable(Tensor(vNaxis['t']))
            ts=Variable(Tensor(vNaxis['t_sample']))
            #ts=torch.linspace(t[0][0],t[0][-1],int(1e3))
            w=Variable(Tensor(vNaxis['w']))
            alphat_r=Variable(Tensor(np.real(alpha['t_sample'])))#real part
            alphat_i=Variable(Tensor(np.imag(alpha['t_sample'])))#imaginary part

            N = 2801
            xf = np.arange(-N/2-1/2,N/2-1/2,1)/np.abs(vNaxis['t_sample'][0])/2
            eV = 1239.84*(2*np.pi*xf)/2.9979E8/(2*np.pi)*1e-9 
            #Interpolate spec outside of the loop to match eV axis of the vNbases
            xeV = pix_2_eV(np.arange(1024), pixpereV = pixpereV, spectra_hw0 = 0, spectra_pix0 = 512)
            xeV = xeV - eshift #correction if calibration is off
            f = interp1d(xeV,spec, bounds_error=False, fill_value = 0)
            spec_in = f(eV[1340:-1340])
            
            spec_in= Variable(Tensor(spec_in))
            spec_in=spec_in/torch.max(spec_in)

            q = np.zeros((1,2*Nb-1))
            q[:,tt] = 1
            Q = Tensor(q)
            Q.requires_grad = True
            
            n_epochs=1.3e3
            lr=1e-2
            optimizer=torch.optim.Adam([Q],lr=lr)
            costs = []
            costM = []
            costW = []
            costS = []
            costQ = []
            eps = []
            M_wght = 1
            Spec_wght = 12 #1 #0.75 #0.8
            Smooth_wght = 0.00001
            for epoch in range(int(n_epochs)):
                
                ll=loss_tot_noQ(M,Q,Nb,Bpr,Bpi,spec_in, alphaw_r, alphaw_i,M_wght,Spec_wght,Smooth_wght,thresh)
                loss_tot_noQ(M,Q,Nb,Bpr,Bpi,spec_in, alphaw_r, alphaw_i,M_wght,Spec_wght,Smooth_wght,thresh).backward()
                #ll=loss_tot(M,Q,Nb,Bpr,Bpi,bkg,alpha_vn,ts,tn,wm,Nt,Nw,5,1,1)
                #loss_tot(M,Q,Nb,Bpr,Bpi,bkg,alpha_vn,ts,tn,wm,Nt,Nw,5,1,1).backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch%100==0:
                    #print(epoch,'total loss = '+str(ll),str(Q[0][0]))
                    costs.append(ll.detach().numpy()[()])
                    costM.append(loss_M(M,Q,Nb,Bpr,Bpi,thresh).detach().numpy()[()]*M_wght)
                    costW.append(loss_w(spec_in, Q, alphaw_r, alphaw_i, Nb).detach().numpy()[()]*Spec_wght)
                    #costS.append(loss_w_smooth(Q, alphaw_r, alphaw_i, Nb).detach().numpy()[()]*Smooth_wght)
                    #costQ.append(0.1*loss_Q(Q,Nb).detach().numpy()[()])
                    costS.append(0)
                    costQ.append(0)
                    eps.append(epoch)
                if ll < lr:
                    break

            outdict['ID'] = k
            outdict['Up'] = Up
            outdict['Qs'] = Q[0].detach().numpy()
            outdict['costs'] = costs
            outdict['costM'] = costM
            outdict['costW'] = costW
            outdict['costS'] = costS
            outdict['costQ'] = costQ
            outdict['M_wght'] = M_wght
            outdict['Spec_wght'] = Spec_wght
            outdict['Smooth_wght'] = Smooth_wght
            outdict['seed'] = tt
            outdict['runnum'] = runnum
            outdict['mdatname'] = mdatname
            outdict['specdatname'] = specdatname
            outdict['basesdir'] = basesdir
            outdict['pixpereV'] = pixpereV
            
            np.save(savefilename, outdict)
            print('one done')

