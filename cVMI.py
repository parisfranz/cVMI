class Point():
    def __init__(self,spec, pixpereV, w0, a1, a2, a3):
        self.pixpereV = pixpereV
        self.spec = spec
        self.streaked = None
        self.xeV = None #goes with spec
        self.w0 = w0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.Qs = None
        self.eV = None #goes with Ew
        self.tpreQ = None
        
    def make_Qs(self):
        '''
        spec - spectra profile with w0 at pix 512 ALREADY ROLLED
        w0 - where to shift center eV to
        a1,a2,a3 1st, 2nd, 3rd order chirps
        '''
        spec = self.spec
        spec[spec<0] = 0
        Ew0 = np.sqrt(spec)

        #make corresponding spectrum frequency axis in eV
        xeV = pix_2_eV(np.arange(len(Ew0)), pixpereV = self.pixpereV, spectra_hw0 = 0, spectra_pix0 = 512)
        #self.xeV = xeV
        xeV = xeV + self.w0

        # n1 = 2 #254 # spans range of vNaxis['w'] and centered around zero, in this case pix 512
        # n2 = -n1+1 
        # xeV = xeV[n1:n2] 
        # Ew0 = Ew0[n1:n2]
        
        n1 = 1# I do not know why this works
        xeV = xeV[n1:] 
        Ew0 = Ew0[n1:] 

        f = interpolate.interp1d(xeV, Ew0, fill_value = 'extrapolate')
        NN = 512*2 #how many points you want to use for the FFT, power of 2
        N = len(Ew0) #current length
        fx = interpolate.interp1d(np.arange(N), xeV, fill_value = 'extrapolate')
        xeV = fx(np.linspace(0, N-1, NN)) #make sure your points are evenly spaced for the FFT and you have the number of points you want
        Ew0 = f(xeV)

       # Ew_in = Ew0*np.exp(1j*(self.a1*(np.pi*2*(xeV-self.w0))+self.a2*(np.pi*2*(xeV-self.w0))**2+self.a3*(np.pi*2*(xeV-self.w0))**3)) #add in the phase
        Ew_in = Ew0*np.exp(1j*(self.a1*((xeV-self.w0))+self.a2*((xeV-self.w0))**2+self.a3*((xeV-self.w0))**3)) #add in the phase

        xf = 2.9979E8*xeV/1239.84e-9
        #f = np.abs(xf[-1])*2
        f = xf[-1] - xf[0]
        

        N = len(Ew_in) #this should be N = NN now
        #t = np.linspace(-N/f/2+1/f/2,N/f/2+1/f/2, N) 
        t = np.linspace(-N/f/2,N/f/2, N) 
        FFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Ew_in))) #shift so central frequency is at 0
        car3 = np.exp(-1.j*(self.w0)/hbar*t)
        FFT = FFT*car3 #add in the frequency due to offset from zero frequency
        f = interpolate.interp1d(t, FFT, fill_value = 0, bounds_error = False) #fill_value = 'extrapolate')
        x = f(vNaxis['t_sample'])
        self.tpreQ = x*x.conj()
        alpha_t = alpha['t_sample']/np.max(np.abs(alpha['t_sample'])) #since we have shifted to right off zero frequency, do not add carrier to alpha

        self.Qs = np.linalg.lstsq(np.matmul(alpha_t.conj(),alpha_t.T)*(vNaxis['t_sample'][1] - vNaxis['t_sample'][0]),np.matmul(alpha_t.conj(),np.reshape(x,(2801,1))*2*np.pi*(t[1]-t[0])))[0]
        #*(vNaxis['t_sample'][1] - vNaxis['t_sample'][0]) is just a scalar, doesn't need to be evenly spaced?    
    
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


    
class Run():
    def __init__(self,runnum, center, stretch):
        self.runnum = runnum
        self.points = get_points(self)
        self.calibration = None
        self.center = center #[x,y]? 
        if stretch = True: 
            self.stretch_matrix = get_stretch_matrix(512-self.center[0], 512-self.center[1])
        else: 
            self.stretch_matrix = make_MM([0,0, 0, 0, 0, 0, 0, 0, 512-self.center[0], 512-self.center[1]]) #make_MM([0,0, 0, 0, 0, 0, 0, 0, 512-535, 512-512])
        self.bg = None #make the dipole subtracted bg with Tarans code - do you want to do quads?
        
    def get_stretch_matrix(self,x0,y0):
        
        return make_MM(X)
    
    def get_points(self):
        w0 = 0
        a1 = 0.5
        a2 = 0
        a3 = 0.004
        for i in range(5):
            p = Point(sss, pixpereV = 22, w0 = w0, a1 = a1, a2 = a2, a3 = a3)
            p.make_Qs()
            
    def make_bg(self):
    
        
    def calibrate(self):
        self.calibration = None
        
    def generate_vNbases(self):
        
    def make_MM(x0):
        xshift, yshift =  x0[8], x0[9] 

        # Coordinates that you want to Perspective Transform
        pts1 = np.float32([[0,0],\
                           [0,1024],\
                           [1024,0],\
                           [1024, 1024]])


        # Size of the Transformed Image
        pts2 = np.float32([[0+xshift-x0[0],0 + yshift-x0[1]],\
                           [0+xshift-x0[2],1024 + yshift-x0[3]],\
                           [1024+xshift-x0[4],0 + yshift-x0[5]],\
                           [1024+xshift-x0[6],1024 + yshift-x0[7]]])

        MM = cv2.getPerspectiveTransform(pts1,pts2)
        return MM

    def MM_error(x0, im):
        mm = make_MM(x0)
        imm = cv2.warpPerspective(im,mm,(1024,1024))
        diffud = rebin(imm*maskhole-np.flipud(imm)*maskhole,4)*2
        difflr = rebin(imm*maskhole-np.fliplr(imm)*maskhole,4)*2
        #SSE = np.sum(diffud**2)
        #return np.sum(diffud**2)/np.linalg.norm(diffud) + np.sum(difflr**2)/np.linalg.norm(difflr)
        return np.sum(diffud**2)/np.linalg.norm(diffud) + np.sum(difflr**2)/np.linalg.norm(difflr)+(np.max(diffud) + np.max(difflr))**2
        #return np.max(diffud) + np.max(difflr)
        #return SSE/np.linalg.norm(diffud)