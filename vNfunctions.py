import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io as scio
import scipy.ndimage as scimg
import scipy.integrate as scint

def dipole_M_cos(Px, Py, Pz, config):
    d = Px/np.sqrt(Px**2+Py**2+Pz**2)
    return d

def dipole_M_iso(Px, Py, Pz, config):
    d = np.ones(np.shape(Px))
    return d


def streak_au_nonlinear(E_X,A,t_X,config):
    E_X = np.reshape(E_X, [len(E_X), 1])
    bp = Photoelectron_Momentum_Distribution(E_X,A,t_X,config)
    P3D = np.abs(bp)**2; #dimension [pz,px*py]
    P2D = np.trapz(P3D,axis=0);#dimension [px*py,1]
    return P2D,P3D,bp

def Photoelectron_Momentum_Distribution(E_X,A,t_X,config, dipole_M = dipole_M_cos):
    #t_X in fs.
    
    #Constants used in calculations
    T_AU = 0.024189; #fs to a.u.
    E_AU = 27.2114; #eV to a.u.

    #Apply Configurations and convert units
    I_p = config['Ip'] / E_AU; #Change to a.u.
    K_max = config['Kmax'];
    K_min = config['Kmin'];
    N_p = config['Np']/2; 
    t_X = t_X / T_AU; # Change fs to a.u.
    
    N_t = len(t_X); #Number of time points
    T=t_X[-1]-t_X[0];#time window in a.u.##########may need to add another step to make 5000 points######
    dt = T / ( len(t_X) - 1 )#Time step in a.u.
    
    #Setup momentum grid
    P = np.sqrt(2*K_max);  #The largest momentum.
    dp = (2*P)/(2*N_p);  #Momentum step

    P_xy = np.arange(-P+dp/2.,P-dp/2.+dp,dp) #vector for x,y momentum
#     P_xy=np.linspace(-P+dp/2.,P-dp/2,int(2*N_p))
    P_z = np.arange(-P+dp/2.,P-dp/2.+dp,dp)#vector for z momentum
#     P_z=np.linspace(-P+dp/2.,P-dp/2,int(2*N_p))
    
    P_xm,P_ym = np.meshgrid(P_xy,P_xy); #Mesh Grid for momentum
    
    N_pz = len(P_z); # number of z-points
    N_pi = int((2*N_p)**2); # number of x,y points
    
    #Prealocate memory for b_p 
    b_p = np.zeros([N_pz,N_pi])+1j*np.zeros([N_pz,N_pi]); # size should be [N_pz,(N_px*N_py)]
    
    #Canonical Momentum, used for calculating the Action
    V_x=np.matmul(np.ones([N_t,1]),P_xm.transpose().flatten()[np.newaxis,:])-np.matmul(A[0,:][:,np.newaxis],np.ones([1,N_pi]))
    V_y=np.matmul(np.ones([N_t,1]),P_ym.transpose().flatten()[np.newaxis,:]) - np.matmul(A[1,:][:,np.newaxis],np.ones([1,N_pi]))

    I_xy = (1/2) * ( np.abs(V_x)**2 +np.abs(V_y)**2 ); #Integrand for Action
    
    S_xy =np.matmul(np.ones([N_t,1]),np.trapz(I_xy,t_X,axis=0)[np.newaxis,:])-scint.cumulative_trapezoid(I_xy,t_X,axis=0,initial=0);
    Phase_xy = np.exp(-1.0j*S_xy )#Phase part of time integral 
    
    for ind_z in range(int(np.ceil(N_pz/2))):
#     for ind_z in range(2):
        p = P_z[ind_z];
        K_z = (1/2) * p**2;
        if K_min>0:# % Only do indexing if neccessary, speeds up calc.
            ind_p = ( (P_xm.flatten()**2 + P_ym.flatten()**2 + p**2) > 2*K_min );
            N_i =ind_p.sum()
    
            Phase_z=np.matmul(np.exp(-1j*(K_z+I_p)*(t_X[-1] - t_X))[:,np.newaxis],np.ones([1,N_i]))
            #Calculate dipole d(p_x-A_x,p_y-A_y,p_z)
            dipole = dipole_M(V_x[:,ind_p],V_y[:,ind_p],p,config);
            
            #Int[E_X*d * Exp(phi_xy) * Exp(phi_z)]
            b_p[ind_z,ind_p] = np.trapz(np.matmul(E_X[:,np.newaxis],np.ones([1,N_i]))*dipole*Phase_xy[:,ind_p]*Phase_z,t_X,axis=0)
            b_p[-1-ind_z,ind_p]=b_p[ind_z,ind_p]#Extend using symmetry around z-axis 
        else:
            Phase_z=np.matmul(np.exp(-1j*(K_z+I_p)*(t_X[-1] - t_X))[:,np.newaxis],np.ones([1,N_pi]))

            #Calculate dipole d(p_x-A_x,p_y-A_y,p_z)
            dipole = dipole_M(V_x,V_y,p,config);

            #Int[E_X*d * Exp(phi_xy) * Exp(phi_z)]
            b_p[ind_z,:]=np.trapz(np.matmul(E_X[:,np.newaxis],np.ones([1,N_pi]))*dipole*Phase_xy*Phase_z,t_X,axis=0)
            b_p[-1-ind_z,:]=b_p[ind_z,:]#Extend using symmetry around z-axis
    
    return b_p

def vNbasis_nonlinear(T,N_w,N_t,nt_tot,nt_x,ndata,Ttot):
    N_basis=N_t*N_w#Total number of vN lattice points
    
    #vN basis functions
    T_Basis=lambda t,tn,wm,alpha: (2*alpha*np.pi)**(-1/4)*np.exp(-(t-tn)**2/(4*alpha))*np.exp(-1j*wm*t)
    W_Basis=lambda w,tn,wm,alpha: (2*alpha/np.pi)**(1/4)*np.exp(-alpha*(w + wm)**2)*np.exp(-1j*tn*(w + wm));
    #Time Axis
    Dt = T / N_basis; #Time step in time rep.
    t = np.arange(-T/2 + Dt/2,T/2-Dt/2+Dt,Dt)
#     t=np.linspace(-T/2 + Dt/2,T/2-Dt/2,N_basis)#Time axis in time domain
    Dt_sample=Ttot/nt_tot#Time step in temporal integration
    t_tot=np.arange(-Ttot/2+Dt_sample/2,Ttot/2-Dt_sample/2+Dt_sample,Dt_sample)
#     t_tot=np.linspace(-Ttot/2+Dt_sample/2,Ttot/2-Dt_sample/2,nt_tot)
    dt_x=T/nt_x
    t_X=np.arange(-T*1.3/2,T*1.3/2+dt_x,dt_x)#time step within x-ray pulse (1.3 is to rid edge effects)
#     t_X=np.linspace(-T*1.3/2,T*1.3/2+dt_x,nt_x)#time step within x-ray pulse (1.3 is to rid edge effects)
    
    #Frequency Axis
    Omega=2*np.pi*(N_basis / T)#frequency range
    DOmega=Omega/N_basis#Frequency step in freq rep.
    wmin=-Omega/2;
    w=np.linspace(wmin + DOmega/2,wmin+Omega-DOmega/2,N_basis)#Frequency axis in frequency domain
    
    #Resample axes
    t_sample=np.sort(np.concatenate((t_tot,t_X),axis=0))
    w_sample=np.linspace(w[0],w[-1],ndata)
    
    #vN alpha
    alpha_vN=T/(2*Omega)#in t^2
    
    #vN lattice
    dt = T / N_t#Time step for vN grid
    dOmega = Omega / N_w#Freq step for vN grid
    #tn=np.linspace(-T/2 + dt/2,T/2-dt/2,N_t)#vN lattice points in time
    tn=np.arange(-T/2+dt/2,T/2-dt/2+dt,dt)
#     wm=np.linspace(wmin+dOmega/2,wmin+Omega-DOmega/2,N_w)#vN lattice points in frequency
    wm=np.arange(wmin+dOmega/2,wmin+Omega-DOmega/2+dOmega,dOmega)

    #Basis Functions
    alpha_t=np.zeros([N_basis,N_basis])+1.0j*np.zeros([N_basis,N_basis]); #time domain basis function
    alpha_w=np.zeros([N_basis,N_basis])+1.0j*np.zeros([N_basis,N_basis]); #Frequency domain basis function
    alpha_t_sample=np.zeros([N_basis, len(t_sample)])+1.0j*np.zeros([N_basis, len(t_sample)]);#Resampled time domain basis function
    alpha_w_sample = np.zeros([N_basis, ndata])+1.0j*np.zeros([N_basis, ndata]); #Resampled frequency domain basis function

    ind = -1;
    for ind_t in range(N_t):
        for ind_w in range(N_w):
            ind = ind+1;
            alpha_t[ind,:] = np.conj(T_Basis(t, tn[ind_t], wm[ind_w], alpha_vN))
            alpha_w[ind,:] = np.conj(W_Basis(w, tn[ind_t], wm[ind_w], alpha_vN))
            alpha_t_sample[ind,:] = np.conj(T_Basis(t_sample, tn[ind_t], wm[ind_w], alpha_vN))
            alpha_w_sample[ind,:] = np.conj(W_Basis(w_sample, tn[ind_t], wm[ind_w], alpha_vN))
     
    
    #Build Output Structures
    #Basis Fns
    alpha={'t':alpha_t,'w':alpha_w,'t_sample':alpha_t_sample,'w_sample':alpha_w_sample}
    #Axes
    axis={'t':t,'w':w,'t_sample':t_sample,'w_sample':w_sample}
    #Grid
    vNgrid =np.concatenate((tn,wm),axis=0)
    
    return alpha,axis,vNgrid
#     return t_tot,t_X,dt_x