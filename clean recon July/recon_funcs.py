from __future__ import print_function
import sys
sys.path.append('/reg/d/psdm/tmo/tmox51020/results/paris/')
sys.path.append('/reg/d/psdm/tmo/tmox51020/results/paris/recon')
import analyzers as az  # this raises a few runtime errors which require some updating
sys.path.append('/reg/neh/home/tdd14/modules/cart2pol/cart2pol')
import numpy as np
import h5py
import vNfunctions as vN
from scipy import optimize
from scipy.optimize import minimize
import cmath
from scipy import special
import time, datetime
from datetime import timedelta
import requests
import json
import psana as ps
from tqdm import tqdm
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import median_filter as mf
from scipy import interpolate
import cart2pol
import cv2
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cart2pol
from cart2pol import PolarRebin

energy_x = 54 #eV
hbar=6.6e-16 #eV*s

0,1,2

def gaus(x, a, mu, sig):
    return a*1/sig/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sig)**2)


def gaus_fit(t,spec):
    #spec = Et*Et.conj()
    popt, pcov = curve_fit(gaus,t,spec, bounds=([0.1, -50, 0.01], [100, 50, 30]))
    a, mu, sig = popt
    return a, mu, sig #FWHM is 2.355*sig


def plot_unds(Ks, Kds, und0 = 26, undf = 48, color = 'tab:blue', ax = None, alpha = 1, label = ''):
    undnum = []
    kplot = []
    for ii,i in enumerate(range(und0,undf)):
        undnum += [i-0.4, i-0.4, i+0.4, i+ 0.4]
        kplot += [0,Ks[ii],Kds[ii], 0]
    if type(ax) == type(None):
        fig,ax = plt.subplots()
        
    ax.plot(undnum, kplot, color = color, alpha = alpha, label = label)
    return ax

def rebin(arr, n=2):
    sy, sx = arr.shape
    shape = (int(sy/n), int(n), int(sx/n), int(n))
    return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)

def plotring(x0 =517,y0 =484, r = 360, pix = 1024, col = 'white', ax = None):
    binn = 1024/pix #32
    phi_rad = np.arange(0,360)*np.pi/180
    
    if type(ax) == type(None):
        fig,ax = plt.subplots()

    ax.plot((x0+r*np.cos(phi_rad))/binn, (y0+r*np.sin(phi_rad))/binn, c=col, alpha = 0.5)

def sin_fit(x, a, b, c):
    return a * np.sin((x+b)/180*np.pi)+c

def circpol(p, x):
    """
    Input: 
    p: streaking amplitude, photoline radius, streaking angle
    x: Polar angles in units of degrees.
    """
    return p[0]*np.cos((x-p[2])*(np.pi/180.)) + np.sqrt(p[1]**2 - (p[0]**2)*(np.sin((x-p[2])*(np.pi/180.))**2))

def lstSqFit(func2fit, xData, yData, p_i, return_resid=False):
    
    errFunc = lambda p, x, y: func2fit(p, x) - y #error function, difference
    #between data point and fitted curve
    p_f = optimize.least_squares(errFunc, p_i[:], args=(xData, yData)) 
    
    if p_f.x[1]<0:
        p_f.x[1] = -p_f.x[1]
    if p_f.x[0]<0:
        p_f.x[0] = -p_f.x[0]
        p_f.x[-1] = p_f.x[-1]+180
        
    p_f = optimize.least_squares(errFunc, p_f.x, args=(xData, yData))
    
    #final p,
    #either a result or the last value attempted if the call was unsucessful
    if return_resid:
        return p_f.x, (errFunc(p_f.x, xData, yData)**2).sum()
    else:
        return p_f.x
    
def center_optimization_1st_polar(ims, init_center):
    """
    This function finds the photoline of 1st harmonics in the polar coordinate. 
    Returned are R and Phi of points on the photoline.
    Input:
    ims: An 1024-by-1024 array, which is a VMI image.
    init_center: The location of the center of unstreaked VMI images.
    Output:
    R_vs_phi: An 1d array of length N. Radii of points on the photoline of 1st harmonics in the polar coordinate.
    phi: An 1d array of length N. Angles of points on the photoline of 1st harmonics in the polar coordinate.
    """
    x0 = init_center[0]
    y0 = init_center[1]
    cvmiana=az.CVMIAnalyzer(center=(x0, 1024-y0))
    
    # Normalize the input image
    ims2plot = np.copy(ims/np.sum(np.ravel(ims)))
    ims_polar=cvmiana.polarRebin(gf(ims2plot, [4,4]), center=cvmiana.center)
    
    ims_polar = gf(ims_polar, [1, 2])
    
    # Search for the edge
    phi0, phi1, phi2, phi3 = 40, 140, 220, 320
    phi = np.concatenate([np.arange(phi0, phi1), np.arange(phi2, phi3)])
    
    #r_start, r_end = 300, 432 #for bg
    r_start, r_end = 310, 400 #for streaking
    R_vs_phi = np.zeros(len(phi))
    for ind in range(len(phi)):
        
        ####################################################################################
        # The idea in this block is to find the radius where the signal starts to have the #
        # largest drop. This point is considered as the edge of the 1st harmonic photoline #
        # at this polar angle.                                                             #
        ####################################################################################
        
        sig_vs_r = ims_polar[r_start:r_end, phi[ind]]
        local_extrema = np.where(( np.diff(sig_vs_r)*np.diff(np.roll(sig_vs_r,1)) )<0)[0]

        sig_jump = sig_vs_r[local_extrema[:-1]]-sig_vs_r[local_extrema[1:]]
        sig_jump = np.append(sig_jump, (sig_vs_r[local_extrema[-1]]-sig_vs_r[-1]) )
        #print(sig_jump)
        if len(sig_jump)>0:
            try:
                R_vs_phi[ind] = r_start+int(np.mean([local_extrema[np.argmax(sig_jump)],local_extrema[np.argmax(sig_jump)+1]]))
            except:
                R_vs_phi[ind] = r_start+local_extrema[np.argmax(sig_jump)]
        else:
            R_vs_phi[ind] = r_start+np.argmax(sig_vs_r)
            
        ####################################################################################
        #                                      End                                         #
        ####################################################################################
    
    return R_vs_phi, phi

def streaking_finding_polar(R_vs_phi, phi):
    """
    Input:
    R_vs_phi: An 1d array of length N. Radii of points on the photoline of 1st harmonics in the polar coordinate.
    phi: An 1d array of length N. Angles of points on the photoline of 1st harmonics in the polar coordinate.
    Output: 
    a0: The streaking amplitude in pixels
    phi0: The streaking angle in rads.
    """
    fit = lstSqFit(circpol, phi, R_vs_phi, [5,300,100], return_resid=True)
    
    a0 = fit[0][0]
    phi0 = fit[0][-1]
    
    return a0, phi0

# def px_2_nm(pixel, vls_pitch, order):
#     pitch_slope = -1.37220068e+03 # negative of pitch shift to px shift proportionality constant
#     pitch_0 = 4.98 #The VLS pitch where the calibration is done 
#     order_0 = 4    
#     pitch_diff = pitch_0-vls_pitch
#     effective_pixel = pixel + (pitch_slope*pitch_diff)
#     offset, slope, slope_2 =  2.38324323e+00, -3.92516982e-04,  1.44317419e-08
#     return (offset + slope*effective_pixel + slope_2*effective_pixel**2)*(order_0/order)

# def get_time_2(spec, vls_pitch =5.399977, order = 3):
#     N = len(spec)
#     xlambda = px_2_nm(np.arange(2048), vls_pitch, order)*1e-9
#     xf = 2.9979E8/xlambda
    
#     f = xf[-1] - xf[0]
#     t = np.linspace(-N/f/2,N/f/2, N) 

#     FFT = np.fft.fftshift(np.fft.fft(spec))
#     return FFT,t

def get_time_2(spec, pixpereV = 101, spectra_hw0 = 0):
    N = len(spec)
    spectra_dhwdpix = 1./pixpereV

    spectra_hws_eV = spectra_hw0 + spectra_dhwdpix*spec
    xlambda = 1239.84/spectra_hws_eV
    
    xf = 2.9979E8/xlambda
    
    f = xf[-1] - xf[0]
    t = np.linspace(-N/f/2,N/f/2, N) 

    FFT = np.fft.fftshift(np.fft.fft(spec))
    return FFT,t


def pix_2_eV(pixel, pixpereV = 101, spectra_hw0 = 0, spectra_pix0 = 512):
    '''
    pixel: array of pixel indices
    pixpereV: calibration pixels per eV
    spectral_hw: (optional) central energy if you have absolutes
    spectra_pix0: where to put w = 0, 512 is at the center for 1024pix spectrometer 
    '''
    spectra_dhwdpix = 1./pixpereV
    spectra_hws_eV = spectra_hw0 + spectra_dhwdpix*(pixel -spectra_pix0)
    return spectra_hws_eV


# def make_bg_fixed(ss, ee, vNaxis, alpha, Bp_basis, pixpereV = 101):
#     #ss is a list of the VLS intensity (EwEw*) for multiple shots
#     spec = np.mean(np.sqrt(ss), axis = 0)  #sqrt of intensity spectra to get E(w)
#     xeV = pix_2_eV(np.arange(2048), pixpereV) + ee
    
#     #define widow on VLS axis so that we can shift (close to) the zero frequency to index 0
#     #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
#     n1 = 220  
#     n2 = n1 + 398
#     xeV = xeV[n1:n2] 
#     spec = spec[n1:n2] 
#     spec = np.fft.fftshift(spec)
#     f = interpolate.interp1d(xeV, spec, fill_value = 'extrapolate')

#     NN = 512*2 #number of sample points in w, pick even number #resample so same number of points later when iFFT back
#     N = len(spec)
#     fx = interpolate.interp1d(np.arange(N), xeV, fill_value = 'extrapolate') #xeV is NOT linspaced
#     xeV = fx(np.linspace(0, N-1, NN))
#     spec = f(xeV)
#     xf = 2.9979E8*xeV/1239.84e-9
#     f = xf[-1] - xf[0]
#     N = len(spec)
#     t = np.linspace(-N/f/2,N/f/2, N) 
#     FFT = np.fft.fftshift(np.fft.fft(spec))

#     #car = np.exp(-1.j*(energy_x)/hbar*t) 
#     car3 = np.exp(-1.j*(ee)/hbar*t) #global offset due to where the KE = XRAY - IP is 
#     FFT = FFT*car3 #add in the frequency due to offset from zero frequency
#     f = interpolate.interp1d(t, FFT, fill_value = 'extrapolate')
#     x = f(vNaxis['t_sample'])
#     #car2 = np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t_sample'],(36,1))) #alpha_t = alpha['t_sample']*car2/np.max(np.abs(alpha['t_sample']))
#     alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) #since we have shifted to right off zero frequency, do not add carrier to alpha

#     Q = np.reshape(np.matmul(x, np.linalg.pinv(alpha_t)), [36,1])
#     Msol=np.real(np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis))
#     #Msol = rebin(np.reshape(Msol1, [int(np.sqrt(Msol1.shape[0])),int(np.sqrt(Msol1.shape[0]))]))
    
#     return Msol

def make_bg_fixed(ss, ee, vNaxis, alpha, Bp_basis, pixpereV = 101, spectra_pix0 = 500):
    #ss is NOT a list of the VLS intensity (EwEw*)
    #do averaging before passing in for multiple shots   
    # ee should be offset from w0 = 0 (small number)
    spec = ss
    pp = 1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = pp+spectra_pix0)
    xeV = xeV 
    
    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 256*2) #from M7

    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    n1 = np.argmin((eV[1] - xeV)**2)#1450#0
    n2 = np.argmin(xeV**2) # where zero is #1950#3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    
    xeV = xeV[n1:n2+(n2-n1)+1] 

    # n1 = 0
    # n2 = 3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
    #                 #also need it to be centered around where you are calling zero or there will be an offset
    
#     t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) 
#     t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)
    #n1 = 1450#0
    #n2 = 1950

    spec = spec[n1:n2+(n2-n1)+1] 
    spec = np.fft.fftshift(spec)
    f = interpolate.interp1d(xeV, spec, fill_value = 'extrapolate') #xeV is evenly spaced so this should be fine

    #sample more points from the relevant region
    NN = 1024
    NNN = 0
    N = len(spec)
    fx = interpolate.interp1d(np.arange(N), xeV, fill_value = 'extrapolate')
    xeV = fx(np.linspace(0-NNN, N-1+NNN, NN)) 
    spec = f(xeV) #evenly spaced
    xf = 2.9979E8*xeV/1239.84e-9

    f = xf[-1] - xf[0]
    N = len(spec)
    t = np.linspace(-N/f/2,N/f/2, N) 

    FFT = np.fft.fftshift(np.fft.fft(spec))

    
    car3 = np.exp(-1.j*(ee)/hbar*t)
    FFT = FFT*car3 #add in the frequency due to offset from zero frequency

    f = interpolate.interp1d(t, FFT, fill_value = 'extrapolate')
    x = f(vNaxis['t_sample'])
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) #since we have shifted to right off zero frequency, do not add carrier to alpha

    Q = np.reshape(np.matmul(x, np.linalg.pinv(alpha_t)), [36,1])
    Msol=np.real(np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis))
    
    return Msol, xeV, spec



def costM6(P2D,Bp_basis,bg_spec,Qfull,alpha,vNaxis,spec,vls_pitch =5.399977,order = 3, eshift = 398):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=Qfull[N_basis:2*N_basis]
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    #eshift = Qfull[-1]
    #eshift = e_shift 
    bg_s = bg_spec/np.sum(bg_spec)
    bg_s = bg_s/np.max(bg_s)*np.max(P2D)*2
    Mguess = np.reshape(rebin(np.reshape(Mguess-bg_s, [64, 64])), [Np**2,])
    diff=Mguess-P2D
    
    #spectra cost
    #simulated spectra
    
    alpha_t = alpha['t']/np.max(np.abs(alpha['t']))
    alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))
    st = np.fft.ifftshift(np.fft.ifft(Et)) 
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))
     
    xf = np.arange(-len(vNaxis['t'])/2,len(vNaxis['t'])/2,1)/(vNaxis['t'][-1] - vNaxis['t'][0])  #Hz???
    f = interpolate.interp1d(2*np.pi*xf,specc, fill_value="extrapolate")
    #specc = f(vNaxis['w'])
    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 100)
    specc = f(ws)
    
    xlambda = px_2_nm(np.arange(2048), vls_pitch, order)*1e-9
    xeV = 1239.84/xlambda*1e-9
    #eV = 1239.84*(vNaxis['w'])/2.9979E8/(2*np.pi)*1e-9 
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 

    #measured spectra
    #spec = np.sqrt(spec - np.mean(spec[1400:1600]))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[1400:1600]))
    f = interpolate.interp1d(xeV - eshift, spec, fill_value="extrapolate")
    x = f(eV + energy_x) #shift the measured spectra and interp so same amount of sample points 
    x = x/np.max(np.abs(x))
    #specc[specc<0.02] = 0
    #x[x<0.02] = 0
    diff_spec = specc - x
    
    #cost=np.sum(np.abs(diff)**2) + np.sum(np.abs(500*diff_spec)**2) #oct 24
    cost=np.sum(np.abs(diff)**2) + np.sum(np.abs(300*diff_spec)**2) #oct 28
    return cost

def pullData(pv, datetime_gmt, end=None):
    #datetime in gmt, luckily I think this is what event.datetime() gives you
    start = datetime_gmt.strftime('%Y-%m-%dT%H:%M:%S') #'<year>-<mon>-<day>T<hr>:<min>:<sec>'
    if type(end) == type(None):
        end = start
    else:
        end = end.strftime('%Y-%m-%dT%H:%M:%S')
    url = "http://lcls-archapp.slac.stanford.edu/retrieval/data/getData.json?pv="+pv+"&from="+start+"Z&to="+end+"Z"
#     print(url)
#     response = urllib2.urlopen(url)
    response = requests.get(url)
    time, value = datArrange(response)
    # returns the time in pacific time
    return time,value

def datArrange(response):
#     data = json.load(response)
    data = response.json()
    realData = data[0]["data"]
    seconds=np.array([x.get('secs') for x in realData[:]],dtype =float)
    nanosUse = 1e-9*np.array([x.get('nanos') for x in realData[:]], dtype = float)
    value = np.array([x.get('val') for x in realData[:]])
    time = np.array([sum(x) for x in zip(seconds, nanosUse)],dtype = float)
    return time, value


def costM2(P2D,Bp_basis, PR, gfsig, mfcart, gfcart, thresh, Qfull):
    #P2D: image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
# #     Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate((Qfull[N_basis::],[0]))
   # Qimag=Qfull[N_basis::]
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    #Mguess = np.reshape(rebin(np.reshape(Mguess, [64, 64])), [Np**2,]).shape
    #Mguess = np.reshape(rebin(np.reshape(Mguess, [128, 128]),int(128/Np)), [Np**2,])
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32)
    bb = np.zeros((64,64))
    bb[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(bb, mfcart), gfcart)
    b = rebin(b)
    Mguess = np.reshape(b, [Np**2,])
    
    Mguess[Mguess < thresh] = 0
    diff=Mguess-P2D
    
    cost=np.sum(np.abs(diff)**2)
    return cost

def costM_der2(P2D,Bp_basis,PR, gfsig, mfcart, gfcart,thresh, Qfull):
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0]))
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
#     Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate((Qfull[N_basis::],[0]))
    #Qimag=Qfull[N_basis::]
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    Mguess = np.reshape(rebin(np.reshape(Mguess, [128, 128]),int(128/Np)), [Np**2,])
    # polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32)
    # bb = np.zeros((64,64))
    # bb[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    # b = gf(mf(bb, mfcart), gfcart)
    # Mguess = np.reshape(b, [Np**2,])
    # Mguess[Mguess < thresh] = 0
    diff=Mguess-P2D
    
    g=np.zeros((N_basis,Np**2))+1j*np.zeros((N_basis,Np**2))
    for j in range(N_basis):
        temp = 2*np.matmul(Qguess.transpose(),Bp_basis[(j*N_basis):(j+1)*N_basis,:])
        #g[j,:]= np.reshape(rebin(np.reshape(temp, [64, 64])), [Np**2,]) 
        g[j,:]= np.reshape(rebin(np.reshape(temp, [128, 128]),int(128/Np)), [Np**2,]) 
    
    Slope_Re=(2*np.sum(np.tile(diff,[N_basis,1])*np.real(g),axis=1))
    Slope_Im=(2*np.sum(np.tile(diff,[N_basis,1])*np.imag(g),axis=1))
    
#     gradient=np.concatenate((Slope_Re,Slope_Im[1::]),axis=0)
    gradient=np.concatenate((Slope_Re,Slope_Im),axis=0)
    
    return gradient.flatten()[:-1]

def costM32(P2D, Bp_basis, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b) #bin down to 32x32
    
    b = np.reshape(b,[32**2,])
    b = b*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    cost=np.sum(np.abs(diff)**2)*10
    return cost

def costM322(P2D, Bp_basis, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b*nmask) #bin down to 32x32
    
    b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    cost=np.sum(np.abs(diff)**2)*10
    return cost


def costM323(P2D, Bp_basis, vNaxis, alpha, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b*nmask) #bin down to 32x32
    
    b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    
    
    
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) ##take out carrier
#alpha_t = alpha['t']/np.max(np.abs(alpha['t'])) #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))


    
    
    N = 1024
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    Etss = f(ts)
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))

    N = len(specc)
    xf = np.arange(-N/2,N/2,1)/((ts[-1] - ts[0]))  #Hz???
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    #ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    ws = np.linspace(vNaxis['w'][0]-1.5e16, vNaxis['w'][-1]+1.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[700:900]))  #sqrt of intensity spectra to get E(w)
    pp =1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = pp+500)
    xeV = xeV + eVshift

    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 256*2) #from M7

    eV2 = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    n1 = np.argmin((eV2[0] - xeV)**2)#1450#0
    n2 = np.argmin((eV2[-1] - xeV)**2)+1#1950#3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    xeV = xeV[n1:n2]
    spec = spec[n1:n2] 
    #spec = np.fft.fftshift(spec)


    f = interpolate.interp1d(xeV, spec, bounds_error=False, fill_value=0)
    x = f(eV)
    
    cost_s = (np.abs(x/np.max(np.abs(x)) - specc/np.max(np.abs(specc)))*2000)**2
    
    cost=np.sum(np.abs(diff)**2)*10 + 0.1*np.sum(cost_s) #0.05*np.sum(cost_s)
    return cost


def costM64(P2D, Bp_basis, vNaxis, alphw, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
            
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = b*nmask
    b = np.reshape(b,[64**2,])
    #b = rebin(b*nmask) #bin down to 32x32
    #b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    
    
    ###Spectra check#######################
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[800:900]))  #sqrt of intensity spectra to get E(w)
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = 512)
    xeV = xeV + eVshift

    
    
    Ew = np.squeeze(np.matmul(Qguess.T,alphw)) #alphw = FFT[alpha['t_sample']]
    
    N = len(Ew)
    xf = np.arange(-N/2,N/2,1)/((vNaxis['t_sample'][-1] - vNaxis['t_sample'][0])) #using tsample range
    eV = 1239.84*(2*np.pi*xf)/2.9979E8/(2*np.pi)*1e-9 
    
    EwEw = np.real(Ew*np.conj(Ew))

    f = interpolate.interp1d(eV,EwEw, bounds_error=False, fill_value = 0)
    specc = f(xeV)
    
    
    
    cost_s = (np.abs(spec/np.max(np.abs(spec)) - specc/np.max(np.abs(specc)))*2000)**2
    
    cost=np.sum(np.abs(diff)**2)*10 + 0.5*np.sum(cost_s) #0.05*np.sum(cost_s)
    return cost

def costM64_smooth(P2D, Bp_basis, vNaxis, alphw, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
            
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = b*nmask
    b = np.reshape(b,[64**2,])
    #b = rebin(b*nmask) #bin down to 32x32
    #b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    
    
    ###Spectra check#######################
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[800:900]))  #sqrt of intensity spectra to get E(w)
    spec[spec<0] = 0
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = 512)
    xeV = xeV + eVshift

    
    
    Ew = np.squeeze(np.matmul(Qguess.T,alphw)) #alphw = FFT[alpha['t_sample']]
    
    N = len(Ew)
    xf = np.arange(-N/2,N/2,1)/((vNaxis['t_sample'][-1] - vNaxis['t_sample'][0])) #using tsample range
    eV = 1239.84*(2*np.pi*xf)/2.9979E8/(2*np.pi)*1e-9 
    
    EwEw = np.real(Ew*np.conj(Ew))

    f = interpolate.interp1d(eV,EwEw, bounds_error=False, fill_value = 0)
    specc = f(xeV)
    cost_s = (np.abs(spec/np.max(np.abs(spec)) - specc/np.max(np.abs(specc)))*2000)**2
    
    
    #add cost for smoothness of spectra phase
    phi = np.unwrap(np.angle(Ew))
    e1 = np.min((np.argmin(np.abs(eV+26)),np.argmin(np.abs(eV-26))))
    e2 = np.max((np.argmin(np.abs(eV+26)),np.argmin(np.abs(eV-26))))
    y = phi[e1-10:e2+10]
    smooth = np.diff(np.diff(y))*2
    #smooth = np.diff(np.diff(y))*10
    #y = (phi[e1-10:e2+10]-np.min(phi[e1-10:e2+10]))/(np.max(phi[e1-10:e2+10])-np.min(phi[e1-10:e2+10]))
    #smooth = np.diff(np.diff(y))*500
    #smooth = smooth/np.max(np.abs(smooth))*10 #normalize so the diff isn't afftected by the actual value of the phase
    cost_smooth = smooth**2 
    w = interpolate.interp1d(xeV,spec,bounds_error=False, fill_value = 0)
    w = (w(eV[e1-8:e2+10])) #**0.75
    w = w/np.max(w) +1 #+ 0.3
    cost_smooth = cost_smooth*w #weigh by amplitude of spectrum
    
    #cost=np.sum(np.abs(diff)**2)*10/10/10 + 0.05/10*np.sum(cost_s)+500*np.sum(cost_smooth) # 500*np.sum(np.sum((np.diff(np.diff(y))*10)**2)) #0.05*np.sum(cost_s)

    #cost=np.sum(np.abs(diff)**2) + np.sum(cost_s) + 150*np.sum(cost_smooth)  #0.05*np.sum(cost_s)
    #cost=np.sum(np.abs(diff)**2) + np.sum(cost_s)/4 + 100*np.sum(cost_smooth)  #0.05*np.sum(cost_s)
    cost=np.sum(np.abs(diff)**2)/10 + 0.05/2*np.sum(cost_s)+ 500*np.sum(cost_smooth)
    return cost

def costM32_smooth(P2D, Bp_basis, vNaxis, alphw, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
            
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = b*nmask
    b[b<thresh] = 0
    b = rebin(b)
    b = np.reshape(b,[32**2,])
    Mguess = b
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    
    
    ###Spectra check#######################
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[800:900]))  #sqrt of intensity spectra to get E(w)
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = 512)
    xeV = xeV + eVshift

    
    
    Ew = np.squeeze(np.matmul(Qguess.T,alphw)) #alphw = FFT[alpha['t_sample']]
    
    N = len(Ew)
    xf = np.arange(-N/2,N/2,1)/((vNaxis['t_sample'][-1] - vNaxis['t_sample'][0])) #using tsample range
    eV = 1239.84*(2*np.pi*xf)/2.9979E8/(2*np.pi)*1e-9 
    
    EwEw = np.real(Ew*np.conj(Ew))

    f = interpolate.interp1d(eV,EwEw, bounds_error=False, fill_value = 0)
    specc = f(xeV)
    cost_s = (np.abs(spec/np.max(np.abs(spec)) - specc/np.max(np.abs(specc)))*2000)**2
    
    
    #add cost for smoothness of spectra phase
    phi = np.unwrap(np.angle(Ew))
    e1 = np.min((np.argmin(np.abs(eV+26)),np.argmin(np.abs(eV-26))))
    e2 = np.max((np.argmin(np.abs(eV+26)),np.argmin(np.abs(eV-26))))
    y = phi[e1-10:e2+10]
    cost_smooth = np.sum(np.diff(np.diff(y))**2)
    
    
    cost=np.sum(np.abs(diff)**2)*10/10/5 + 0.05/5*np.sum(cost_s)+5000*cost_smooth #0.05*np.sum(cost_s)
    return cost


def costM323dot(P2D, Bp_basis, vNaxis, alpha, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b*nmask) #bin down to 32x32
    
    b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    diff=(1-np.dot(Mguess/np.linalg.norm(Mguess),P2D/np.linalg.norm(P2D)))*5e7
    #diff = Mguess - P2D
    
    
    
    
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) ##take out carrier
#alpha_t = alpha['t']/np.max(np.abs(alpha['t'])) #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))


    
    
    N = 1024
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    Etss = f(ts)
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))

    N = len(specc)
    xf = np.arange(-N/2,N/2,1)/((ts[-1] - ts[0]))  #Hz???
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    #ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    ws = np.linspace(vNaxis['w'][0]-1.5e16, vNaxis['w'][-1]+1.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[700:900]))  #sqrt of intensity spectra to get E(w)
    pp =1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = pp+500)
    xeV = xeV + eVshift

    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 256*2) #from M7

    eV2 = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    n1 = np.argmin((eV2[0] - xeV)**2)#1450#0
    n2 = np.argmin((eV2[-1] - xeV)**2)+1#1950#3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    xeV = xeV[n1:n2]
    spec = spec[n1:n2] 
    #spec = np.fft.fftshift(spec)


    f = interpolate.interp1d(xeV, spec, bounds_error=False, fill_value=0)
    x = f(eV)
    
    cost_s = (np.abs(x/np.max(np.abs(x)) - specc/np.max(np.abs(specc)))*2000)**2
    
    cost=np.sum(np.abs(diff)**2) + 0.05*np.sum(cost_s)
    return cost

def costM323i(P2D, Bp_basis, vNaxis, alpha, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-1]
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b*nmask) #bin down to 32x32
    b = b-np.flip(np.flip(b,axis = 0),axis = 1)
    
    b = np.reshape(b,[32**2,])
    b = b#*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    #Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    
    
    
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) ##take out carrier
#alpha_t = alpha['t']/np.max(np.abs(alpha['t'])) #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))


    
    
    N = 1024
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    Etss = f(ts)
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))

    N = len(specc)
    xf = np.arange(-N/2,N/2,1)/((ts[-1] - ts[0]))  #Hz???
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    #ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    ws = np.linspace(vNaxis['w'][0]-1.5e16, vNaxis['w'][-1]+1.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[700:900]))  #sqrt of intensity spectra to get E(w)
    pp =1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = pp+500)
    xeV = xeV + eVshift

    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 256*2) #from M7

    eV2 = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    n1 = np.argmin((eV2[0] - xeV)**2)#1450#0
    n2 = np.argmin((eV2[-1] - xeV)**2)+1#1950#3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    xeV = xeV[n1:n2]
    spec = spec[n1:n2] 
    #spec = np.fft.fftshift(spec)


    f = interpolate.interp1d(xeV, spec, bounds_error=False, fill_value=0)
    x = f(eV)
    
    cost_s = (np.abs(x/np.max(np.abs(x)) - specc/np.max(np.abs(specc)))*2000)**2
    
    cost=np.sum(np.abs(diff)**2)*10 + 0.05*np.sum(cost_s)
    return cost

def costM323sub(P2D,fzp_bg, Bp_basis, vNaxis, alpha, fzp, PR,gfsig, gfcart, mfcart, nmask,thresh, vNaxis0, alpha0, Bp_basis0,  pixpereV, spectra_pix0, Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    eVshift = Qfull[-4]
    scale = Qfull[-3]
    scale2 = Qfull[-2]
    deebg = Qfull[-1]
    
    
    bg_spec,xx,ss =  make_bg_fixed(fastsmooth1(fzp_bg,10), deebg, vNaxis0, alpha0, Bp_basis0,  pixpereV = pixpereV, spectra_pix0 = spectra_pix0)
    polimg = PR.cart2pol(rebin(np.reshape(bg_spec, [128,128])),32,32)
    bb = np.zeros((64,64))
    bb[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    bb = gf(mf(bb,mfcart), gfcart)
    bg_spec = bb

    
    
    
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(rebin(np.reshape(Mguess, [128,128])),32,32) #128x128 still
    b = np.zeros((64,64))
    #b = np.zeros((128,128))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    #b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    #b = rebin(b*nmask) #bin down to 32x32
    Mguess = b
    Mguess[np.abs(Mguess) < thresh] = 0
    Mguess = Mguess/np.linalg.norm(Mguess)
    scimbg = bg_spec
    scimbg = scimbg/np.linalg.norm(scimbg)*scale
    #scimbg = scimbg/np.mean(scimbg[38:41,38:41])*np.mean(Mguess[38:41,38:41])*scale
    #scimbg = scimbg/np.sum(scimbg)*np.sum(Mguess)*scale  #simulated shouldn't have all that noise, so instead make the magnitudes match? idk
    Mguess = rebin((Mguess - scimbg)*nmask)
    
    #Mguess = np.reshape(Mguess,[32**2,])
    Mguess = np.reshape(Mguess,[32**2,])/np.linalg.norm(Mguess)*np.linalg.norm(P2D)
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    #diff = diff*10000
    diff = Mguess*scale2 - P2D
    
    
    
    
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) ##take out carrier
#alpha_t = alpha['t']/np.max(np.abs(alpha['t'])) #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))


    
    
    N = 1024
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    Etss = f(ts)
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))

    N = len(specc)
    xf = np.arange(-N/2,N/2,1)/((ts[-1] - ts[0]))  #Hz???
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    #ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    ws = np.linspace(vNaxis['w'][0]-1.5e16, vNaxis['w'][-1]+1.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    
    spec = np.squeeze(fastsmooth1(fzp,10))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[700:900]))  #sqrt of intensity spectra to get E(w)
    pp =1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 22, spectra_hw0 = 0, spectra_pix0 = pp+500)
    xeV = xeV + eVshift

    ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 256*2) #from M7

    eV2 = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    n1 = np.argmin((eV2[0] - xeV)**2)#1450#0
    n2 = np.argmin((eV2[-1] - xeV)**2)+1#1950#3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    xeV = xeV[n1:n2]
    spec = spec[n1:n2] 
    #spec = np.fft.fftshift(spec)


    f = interpolate.interp1d(xeV, spec, bounds_error=False, fill_value=0)
    x = f(eV)
    
    cost_s = (np.abs(x/np.max(np.abs(x)) - specc/np.max(np.abs(specc)))*2000)**2
    
    cost=np.sum(np.abs(diff)**2)*10 + np.sum(cost_s)
    return cost

def costM22(P2D, Bp_basis, PR,gfsig, gfcart, mfcart, nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(np.reshape(Mguess, [128,128]),64,64) #128x128 still
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(mf(b, mfcart), gfcart)
    b = rebin(b) #bin down to 64x64
    
    b = np.reshape(b,[64**2,])
    b = b*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    diff=(Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D))*10
    #diff = Mguess - P2D
    
    cost=np.sum(np.abs(diff)**2)*10
    return cost


def costM3(P2D,Bp_basis, PR, gfsig,nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(np.reshape(Mguess, [128,128]),64,64) #128x128 still
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(b, 2)
    b = rebin(b) #bin down to 64x64
    
    b = np.reshape(b,[64**2,])
    b = b*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    #Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    #diff = Mguess - P2D
    
    cost=np.sum(np.abs(diff)**2)*10
    return cost
def costM4(P2D,Bp_basis, PR, gfsig,nmask,thresh,Qfull):
    #P2D is the streaked image unwrapped into an array (not bg substracted)
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(np.reshape(Mguess, [128,128]),64,64) #128x128 still
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(b, 2)
    b = rebin(b) #bin down to 64x64
    
    b = np.reshape(b,[64**2,])
    b = b*nmask #/np.sum(b*nmask)*1e5 
    Mguess = b
    Mguess[Mguess < thresh] = 0
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    #diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    diff = Mguess - P2D
    
    cost=np.sum(np.abs(diff)**2)*10
    return cost

def costM_der4(P2D,Bp_basis,PR, gfsig,nmask,Qfull):
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0]))
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
#     Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    polimg = PR.cart2pol(np.reshape(Mguess, [128,128]),64,64) #128x128 still
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(b, 2)
    b = rebin(b) #bin down to 64x64
    
    b = np.reshape(b,[64**2,])
    b = b*nmask/np.sum(b*nmask)*1e5 
    Mguess = b
    
    diff=Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
        
    g=np.zeros((N_basis,Np**2))+1j*np.zeros((N_basis,Np**2))
    g0=np.zeros((N_basis,Np**2))+1j*np.zeros((N_basis,Np**2))
    for j in range(N_basis):
        temp = 2*np.matmul(Qguess.transpose(),Bp_basis[(j*N_basis):(j+1)*N_basis,:])
        g[j,:]= np.reshape(rebin(np.reshape(temp, [Np*2, Np*2])), [Np**2,])
        
    Slope_Re=(2*np.sum(np.tile(diff,[N_basis,1])*np.real(g),axis=1))
    Slope_Im=(2*np.sum(np.tile(diff,[N_basis,1])*np.imag(g),axis=1))
    
#     gradient=np.concatenate((Slope_Re,Slope_Im[1::]),axis=0)
    gradient=np.concatenate((Slope_Re,Slope_Im[1::]),axis=0)
    
    return gradient.flatten()


def costM5(P2D,Bp_basis,bg_spec,Qfull):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
        
    
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    bg_s = bg_spec/np.sum(bg_spec)
    bg_s = bg_s/np.max(bg_s)*np.max(P2D)*2
     
    Mguess = np.reshape(rebin(np.reshape(Mguess-bg_s, [Np*2, Np*2])), [Np**2,])
    diff=Mguess-P2D
    
    cost=np.sum(np.abs(diff)**2)
    return cost

def costM_der5(P2D,Bp_basis,bg_spec,Qfull):
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0]))
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
#     Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    bg_s = bg_spec/np.sum(bg_spec)
    bg_s = bg_s/np.max(bg_s)*np.max(P2D)*2
    Mguess = np.reshape(rebin(np.reshape(Mguess-bg_s, [Np*2, Np*2])), [Np**2,])

    diff=Mguess-P2D
     
    
    g=np.zeros((N_basis,Np**2))+1j*np.zeros((N_basis,Np**2))
    g0=np.zeros((N_basis,Np**2))+1j*np.zeros((N_basis,Np**2))
    for j in range(N_basis):
        temp = 2*np.matmul(Qguess.transpose(),Bp_basis[(j*N_basis):(j+1)*N_basis,:])
        g[j,:]= np.reshape(rebin(np.reshape(temp, [Np*2, Np*2])), [Np**2,])
        
    Slope_Re=(2*np.sum(np.tile(diff,[N_basis,1])*np.real(g),axis=1))
    Slope_Im=(2*np.sum(np.tile(diff,[N_basis,1])*np.imag(g),axis=1))
    
#     gradient=np.concatenate((Slope_Re,Slope_Im[1::]),axis=0)
    gradient=np.concatenate((Slope_Re,Slope_Im[1::]),axis=0)
    
    return gradient.flatten()

def costM7_Ring(P2D,Bp_basis,bg_spec,Qfull,alpha,vNaxis,spec,mask, vls_pitch =5.399977,order = 3, eshift = 398):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    Np=int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
# #     Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    #eshift = e_shift
    #eshift = Qfull[-1]
    
    #X,Y=np.meshgrid(np.arange(64),np.arange(64))
    #rs=np.sqrt((X-512/16)**2+(Y-512/16)**2)
    #maskhmm=np.zeros((64,64))
    #maskhmm[rs>4*52/16]=1
    
    bg_s = bg_spec/np.sum(bg_spec)
    bg_s = bg_s/np.max(bg_s)*np.max(P2D)*2
    Mguess = np.reshape(rebin(np.reshape(Mguess-bg_s, [64, 64])), [Np**2,])
    
    P2D = np.reshape(np.reshape(P2D, [Np,Np])*mask, [Np**2,])
    #diff=Mguess-P2D
    diff = 1- np.dot(Mguess/np.linalg.norm(Mguess),P2D/np.linalg.norm(P2D))
    
    
    #spectra cost
    #simulated spectra
    
#     alpha_w=alpha['w']/np.max(np.abs(alpha['w']))*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['w'],(N_basis,1)))
#     Ew= np.squeeze(np.matmul(Qguess.T,alpha_w))
#     specc = Ew*np.conj(Ew)
#     specc = specc/np.max(np.abs(specc))
    
    alpha_t = alpha['t']/np.max(np.abs(alpha['t']))
    alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))
    
    st = np.fft.ifftshift(np.fft.ifft(Et)) 
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))
     
    xf = np.arange(-len(vNaxis['t'])/2,len(vNaxis['t'])/2,1)/(vNaxis['t'][-1] - vNaxis['t'][0])  #Hz???
    f = interpolate.interp1d(2*np.pi*xf,specc, fill_value="extrapolate")
    #specc = f(vNaxis['w'])
    #ws = np.linspace(vNaxis['w'][0], vNaxis['w'][-1], 100)
    ws = np.linspace(2*np.pi*xf[0]-.05e16, 2*np.pi*xf[-1]+.05e16, 100) #from M7
   
    specc = f(ws)
    
    

    xlambda = px_2_nm(np.arange(2048), vls_pitch, order)*1e-9
    xeV = 1239.84/xlambda*1e-9
    #eV = 1239.84*(vNaxis['w'])/2.9979E8/(2*np.pi)*1e-9 
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 

    #measured spectra
    #spec = np.sqrt(spec - np.mean(spec[1400:1600]))
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[1400:1600]))
    f = interpolate.interp1d(xeV - eshift, spec, fill_value="extrapolate")
    x = f(eV + energy_x) #shift the measured spectra and interp so same amount of sample points 
    x = x/np.max(np.abs(x))
    #diff_spec = specc - x
    diff_spec = 1- np.dot(specc/np.linalg.norm(specc),x/np.linalg.norm(x))
    
    #cost=np.sum(np.abs(diff)**2) + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*5e6 + np.sum(np.abs(500*diff_spec)**2)
    cost= np.abs(diff)*5e6 + np.abs(diff_spec)*3e6
    return cost


def costM8(P2D,Bp_basis,bg_spec, Qfull,alpha,vNaxis,spec, ee, scale, nmask, PR, gfsig, vls_pitch =5.399977,order = 3):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    N = 512*2 #points for FFT
    Np= int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
# #   Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    
    bg_s = bg_spec/np.sum(bg_spec)
    polimg = PR.cart2pol(np.reshape(bg_s, [64,64]),32,32)
    b = np.zeros((64,64))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = np.reshape(b,[64**2,])
    b = (b*nmask)/np.sum(nmask*b)*1e5
    bg_s = b*scale
    #bg_s = (bg_s*nmask)/np.sum((bg_s*nmask))*1e5*scale
    Mguess = np.reshape(rebin(np.reshape(Mguess, [128, 128])), [Np**2,])
    polimg = PR.cart2pol(np.reshape(Mguess, [64,64]),32,32)
    b = np.zeros((64,64))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = np.reshape(b,[64**2,])
    b = b*nmask/np.sum(b*nmask)*1e5 
    Mguess = b #do smoothing on bases
    #Mguess =  Mguess*nmask #/np.sum(Mguess*nmask)*1e5
    Mguess = Mguess - bg_s #32 by 32
    
    # a = np.max(P2D)
    # b = np.min(P2D)
    # c = np.max(Mguess)
    # d = np.min(Mguess)
    # Mguess = Mguess*(a*c+b*d)/(c**2+d**2)
    
    #Mguess = np.reshape(np.reshape(Mguess-bg_s, [64, 64]), [Np**2,]) #64 by 64
    
    P2D = np.reshape(np.reshape(P2D, [Np,Np]), [Np**2,])
    diff= Mguess-P2D
    #diff = 1- np.dot(Mguess/np.linalg.norm(Mguess),P2D/np.linalg.norm(P2D))
    
 
    #### Get Et from Q Guess's #####
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample']))
    #alpha_t = alpha['t']/np.max(np.abs(alpha['t']))
    #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))
    
    
    #define the indexes to truncate t_sample to the vNaxis t
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    Et = Et[t1:t2] 
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    
    
    #### Get Ew from Et #####
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))
    
    
    xf = np.arange(-N/2,N/2,1)/(ts[-1] - ts[0])  #Hz???
    #xxeV = 1239.84*(xf*2*np.pi)/2.9979E8/(2*np.pi)*1e-9
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    

    #### shift central energy of measured VLS spectra #######  
    
    xlambda = px_2_nm(np.arange(2048), vls_pitch, order)*1e-9
    xeV = 1239.84/xlambda*1e-9
    xeV = xeV - 397.1 - (energy_x + 12) + ee  #ee from calibration
   
    
    #measured spectra
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[1500:1700]))
    spec[spec<0] = 0
    f = interpolate.interp1d(xeV, spec, fill_value="extrapolate")
    x = f(eV)
    vlss = x/np.max(np.abs(x))
    idx = np.where((vlss<0.005) & (np.arange(len(vlss)) < np.argmax(vlss)))[0][-1]
    vlss[:idx] = 0
    idx = np.where((vlss<0.005) & (np.arange(len(vlss)) > np.argmax(vlss)))[0][0]
    vlss[idx:] = 0
    diff_spec = specc - vlss
    #diff_spec = 1- np.dot(specc/np.linalg.norm(specc),x/np.linalg.norm(x))
    
#     plt.plot(eV, x, label = 'vls')
#     plt.plot(eV,specc, label = 'fit')
#     plt.xlim(-20,20)
#     plt.legend()
    
    
    cost=np.sum(np.abs(diff)**2)/10 + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.abs(diff_spec)*3e6
    return cost


def costM9(P2D,Bp_basis,bg_spec, Qfull,alpha,vNaxis,spec, ee, scale, nmask, PR, gfsig, vls_pitch =5.399977,order = 3):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    N = 512*2 #points for FFT
    Np= int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
# #   Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    
    bg_s = bg_spec/np.sum(bg_spec)
    polimg = PR.cart2pol(np.reshape(bg_s, [64,64]),32,32)
    b = np.zeros((64,64))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = np.reshape(b,[64**2,])
    b = (b*nmask)/np.sum(nmask*b)*1e5
    bg_s = b*scale
    #bg_s = (bg_s*nmask)/np.sum((bg_s*nmask))*1e5*scale
    Mguess = np.reshape(rebin(np.reshape(Mguess, [128, 128])), [Np**2,])
    polimg = PR.cart2pol(np.reshape(Mguess, [64,64]),32,32)
    b = np.zeros((64,64))
    b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = np.reshape(b,[64**2,])
    b = b*nmask/np.sum(b*nmask)*1e5 
    Mguess = b #do smoothing on bases
    #Mguess =  Mguess*nmask #/np.sum(Mguess*nmask)*1e5
    Mguess = Mguess - bg_s #32 by 32
    
    # a = np.max(P2D)
    # b = np.min(P2D)
    # c = np.max(Mguess)
    # d = np.min(Mguess)
    # Mguess = Mguess*(a*c+b*d)/(c**2+d**2)
    
    #Mguess = np.reshape(np.reshape(Mguess-bg_s, [64, 64]), [Np**2,]) #64 by 64
    
    P2D = np.reshape(np.reshape(P2D, [Np,Np]), [Np**2,])
    diff= Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    #diff = 1- np.dot(Mguess/np.linalg.norm(Mguess),P2D/np.linalg.norm(P2D))
    
 
    #### Get Et from Q Guess's #####
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample']))
    #alpha_t = alpha['t']/np.max(np.abs(alpha['t']))
    #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))
    
    
    #define the indexes to truncate t_sample to the vNaxis t
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    Et = Et[t1:t2] 
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    
    
    #### Get Ew from Et #####
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))
    
    
    xf = np.arange(-N/2,N/2,1)/(ts[-1] - ts[0])  #Hz???
    #xxeV = 1239.84*(xf*2*np.pi)/2.9979E8/(2*np.pi)*1e-9
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    
    

    #### shift central energy of measured VLS spectra #######  
    
    xlambda = px_2_nm(np.arange(2048), vls_pitch, order)*1e-9
    xeV = 1239.84/xlambda*1e-9
    xeV = xeV - 397.1 - (energy_x + 12) + ee  #ee from calibration
   
    
    #measured spectra
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[1500:1700]))
    spec[spec<0] = 0
    f = interpolate.interp1d(xeV, spec, fill_value="extrapolate")
    x = f(eV)
    vlss = x/np.max(np.abs(x))
    idx = np.where((vlss<0.005) & (np.arange(len(vlss)) < np.argmax(vlss)))[0][-1]
    vlss[:idx] = 0
    idx = np.where((vlss<0.005) & (np.arange(len(vlss)) > np.argmax(vlss)))[0][0]
    vlss[idx:] = 0
    diff_spec = specc - vlss
    #diff_spec = 1- np.dot(specc/np.linalg.norm(specc),x/np.linalg.norm(x))
    
#     plt.plot(eV, x, label = 'vls')
#     plt.plot(eV,specc, label = 'fit')
#     plt.xlim(-20,20)
#     plt.legend()
    
    
    cost=np.sum(np.abs(1.5e3*diff)**2) + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.abs(diff_spec)*3e6
    return cost


def costM10(P2D,Bp_basis,bg_spec, Qfull,alpha,vNaxis,spec, ee, scale, nmask, PR, gfsig, spectra_pix0):
    #P2D is the background subtracted image unwrapped into an array
    #this function is based on calculating M using Bp_basis
    #x3=np.matmul(np.matmul(Q,Q.conj().transpose()).transpose().flatten(),Bp_basis)
    N = 512*2 #points for FFT
    Np= int(np.sqrt(np.shape(P2D)[0])) 
    N_basis=int(np.sqrt(np.shape(Bp_basis)[0]))
    Qreal=Qfull[0:N_basis]
# #   Qimag=np.concatenate((np.zeros((1,1)),Qfull[N_basis::]),axis=0)
#     Qimag=np.reshape(np.append(0,Qfull[N_basis::]),[N_basis,1])
    Qimag=np.concatenate(([0],Qfull[N_basis:2*N_basis-1]))
    Qguess=Qreal+1j*Qimag
    Qguess=np.reshape(Qguess,[N_basis,1])
    Mguess=np.real(np.matmul(np.matmul(Qguess,Qguess.conj().transpose()).transpose().flatten(),Bp_basis))
    
    
    bg_s = bg_spec/np.sum(bg_spec)
    polimg = PR.cart2pol(np.reshape(bg_s, [128,128]),64,64)
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = rebin(b)
    b = np.reshape(b,[64**2,])
    b = (b*nmask)/np.sum(nmask*b)*1e5
    bg_s = b*scale
    
    #Mguess = np.reshape(rebin(np.reshape(Mguess, [128, 128])), [Np**2,])
    polimg = PR.cart2pol(np.reshape(Mguess, [128,128]),64,64) #128x128 still
    #b = np.zeros((64,64))
    b = np.zeros((128,128))
    #b[1:64,1:64]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b[1:128,1:128]= PR.pol2cart(gf(polimg,[gfsig,0]))
    b = gf(b, 2)
    b = rebin(b) #bin down to 64x64
    
    b = np.reshape(b,[64**2,])
    b = b*nmask/np.sum(b*nmask)*1e5 
    Mguess = b #do smoothing on bases
    #Mguess =  Mguess*nmask #/np.sum(Mguess*nmask)*1e5
    Mguess = Mguess - bg_s 
    
    P2D = np.reshape(np.reshape(P2D, [Np,Np]), [Np**2,])
    diff= Mguess/np.linalg.norm(Mguess)-P2D/np.linalg.norm(P2D)
    #diff = 1- np.dot(Mguess/np.linalg.norm(Mguess),P2D/np.linalg.norm(P2D))
    
    '''
    #### Get Et from Q Guess's #####
    alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample']))
    #alpha_t = alpha['t']/np.max(np.abs(alpha['t']))
    #alpha_t = alpha_t*np.exp(-1.j*(energy_x)/hbar*np.tile(vNaxis['t'],(N_basis,1)))
    Et = np.squeeze(np.matmul(Qguess.T,alpha_t))
    
    
    #define the indexes to truncate t_sample to the vNaxis t
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) #772
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)#4828 [386:2414]
    
    f = interpolate.interp1d(vNaxis['t_sample'], Et, bounds_error=False, fill_value = 0)
    Et = Et[t1:t2] 
    ts = np.linspace(vNaxis['t_sample'][t1:t2][0],vNaxis['t_sample'][t1:t2][-1], N)
    
    
    #### Get Ew from Et #####
    st = np.fft.ifftshift(np.fft.ifft(f(ts)))
    specc = st*np.conj(st)
    specc = specc/np.max(np.abs(specc))
    
    
    xf = np.arange(-N/2,N/2,1)/(ts[-1] - ts[0])  #Hz???
    #xxeV = 1239.84*(xf*2*np.pi)/2.9979E8/(2*np.pi)*1e-9
    
    f = interpolate.interp1d(2*np.pi*xf,specc, bounds_error=False, fill_value = 0)
    ws = np.linspace(vNaxis['w'][0]-2.5e16, vNaxis['w'][-1]+2.5e16, 256*2) #from M7
    specc = f(ws)
    eV = 1239.84*(ws)/2.9979E8/(2*np.pi)*1e-9 
    
    

    #### shift central energy of measured VLS spectra #######  
    
    spec = np.array(np.squeeze(spec) - np.mean(np.squeeze(spec)[700:900]))  #sqrt of intensity spectra to get E(w)
    pp =1200
    spec = np.pad(spec, (pp,), 'constant',constant_values=(0))
    N = len(spec)
    xeV = pix_2_eV(np.arange(N), pixpereV = 101, spectra_hw0 = 0, spectra_pix0 = pp+spectra_pix0)
    xeV = xeV + ee

    n1 = 0
    n2 = 3400 #want this to span 33.40369027550054 eV or less (range of vNaxis['w'])
                    #also need it to be centered around where you are calling zero or there will be an offset
    t1 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][0])**2) 
    t2 = np.argmin((vNaxis['t_sample'] -vNaxis['t'][-1])**2)
    xeV = xeV[n1:n2]
    spec = spec[n1:n2] 
    spec = np.fft.fftshift(spec)
   

    #f = interpolate.interp1d(xeV, spec, fill_value="extrapolate")
    f = interpolate.interp1d(xeV, spec, bounds_error=False, fill_value=0)
    x = f(eV)
    vlss = x/np.max(np.abs(x))
    #idx = np.where((vlss<0.005) & (np.arange(len(vlss)) < np.argmax(vlss)))[0][-1]
    #vlss[:idx] = 0
    #idx = np.where((vlss<0.005) & (np.arange(len(vlss)) > np.argmax(vlss)))[0][0]
    #vlss[idx:] = 0
    diff_spec = specc - vlss
    #diff_spec = 1- np.dot(specc/np.linalg.norm(specc),x/np.linalg.norm(x))
    
#     plt.plot(eV, x, label = 'vls')
#     plt.plot(eV,specc, label = 'fit')
#     plt.xlim(-20,20)
#     plt.legend()
    '''
    
    cost=np.sum(np.abs(100*diff)**2)
    #cost=np.sum(np.abs(1.5e3*diff)**2)+ np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.sum(np.abs(500*diff_spec)**2)
    #cost= np.abs(diff)*3e6 + np.abs(diff_spec)*3e6
    return cost



def rad_smooth_basis(Bp_basis, gfsig, PR = None):
    N = int(np.sqrt(Bp_basis.shape[1]))
    if type(PR) == type(None):
        PR = PolarRebin(cbins=int(N/2),rbins=64,thbins=64)  #cbin is the cartesian bin number, should be half of the final image size 
    Bp_smooth = np.zeros(Bp_basis.shape)
    for i in range(Bp_basis.shape[0]):
        polimg = PR.cart2pol(np.reshape(np.real(Bp_basis[i]), [N,N]),64,64)
        b = np.zeros((N,N))
        b[1:N,1:N]= PR.pol2cart(gf(polimg,[gfsig,0]))
        b = np.reshape(b,[N**2,])
        Bp_smooth[i] = b
    
    return Bp_smooth
 
def spectra_centroid(spec, n1 = 250, n2 =750):
    spec = np.squeeze(spec)    
    w2 = (spec - spec[n2+100:n2+400].mean())[n1:n2]

    M = np.sum(w2)
    centroid = n1 + 1/M*np.sum(w2*np.arange(0,n2-n1))
    
    return int(centroid)

def spectra_centroid_eV(spec):
    #return the spectra centroid in eV
    centroid = spectra_centroid(spec, n1 = 250, n2 =750)
    centroids_bg_eV = px_2_nm(np.array(centroid), vls_pitch = 5.399977, order = 3)*1e-9
    return 1239.84/centroids_bg_eV*1e-9


def get_centroidish(x):
    x = np.squeeze(np.copy(x))   
    w2 = x
    w2[w2< 0.25*np.max(w2)] = 0

    M = np.sum(w2)
    centroid = 1/M*np.sum(w2*np.arange(0,len(x)))
    
    return int(centroid)


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