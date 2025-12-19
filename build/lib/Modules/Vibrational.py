import os, sys

import numpy as np

import ase, ase.io
from ase import Atoms

__JULIA__ = False
try:
    from julia import Julia
    # Avoid precompile issues
    jl = Julia(compiled_modules=False)  
    
    # Import a Julia module
    from julia import Main

    # current_dir = os.path.dirname(__file__)
    # print(os.path.dirname(__file__))
    Main.include("/home/antonio/IOcp2k/Modules/time_correlation.jl")
    # Main.include(os.path.join(os.path.dirname(__file__), "time_correlation.jl"))
    __JULIA__ = True
except:
    __JULIA__ = False
    print("Vibrational module| Sorry no Julia found. In case you want to use it try with python-jl")


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
matplotlib.use('tkagg')

import AtomicSnap
from AtomicSnap import AtomicSnapshots

import scipy, scipy.optimize, scipy.fft

import copy

import time

import warnings

# Conversions
BOHR_TO_ANGSTROM = 0.529177249 
ANGSTROM_TO_BOHR = 1/BOHR_TO_ANGSTROM
THZ_TO_CM = 33
# Dipoles
DEBEYE_TO_eANG = 0.20819433622621247
DEBEYE_TO_AU = DEBEYE_TO_eANG * ANGSTROM_TO_BOHR

matplotlib.use('tkagg')

class Vibrational:
    """
    GET THE DIPOLE DIPOLE CORRELATION FUNCTION
    ==========================================
    """
    
    def __init__(self, **kwargs):
        """
        DEFINE THE ATTRIBUTES OF THE CLASS
        ==================================

        Uses PICOSECONDS and eANGSTROM
        """
        # PICOSECOND
        self.dt = None

        # PICOSECOND
        self.t = None
        
        # Tempeartue in Kelvin
        self.T = None

        # The volumes in ANGSTROM3
        self.V = None

        # All the dipoles in ATOMIC UNITS (N, 3)
        self.d = None

        self.N = None

        # The time dependent correlation function and error
        self.correlations = None
        
        self.error_correlations = None
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
        
    def init(self, d = None, dt = 0.5, show_dipoles = False):
        """
        INITIALIZE
        ==========

        Paramters:
        ----------
            -ase_atoms: list of ase atoms objects
            -dt: time step in FEMPTOSECOND
            -T: temperature in Kelvin
        """
        print("\n\n\n=========== WELCOME TO THE VIBRATIONAL ANALYSIS TOOL ===========\n")
        # The legth of the trajectory
        self.N = len(d[:,0])
        
        # in PICOSECOND
        self.dt = dt * 1e-3

        # IN PICOSECOND
        self.t = np.arange(self.N) * self.dt

        # Get all the currents in eANGSTROM
        self.d = np.copy(d) * DEBEYE_TO_AU**-1 * DEBEYE_TO_eANG


        # The time dependent correlations
        self.correlations = np.zeros(self.N, dtype = type(d[0,0]))

        self.error_correlations = np.zeros(self.N, dtype = type(d[0,0]))

        # Subtract the average
        cmpts = ["X", "Y", "Z"]

        for i in range(3):
            print("DIPOLES comp {} subtracting average {:.2f} eANG".format(cmpts[i],  np.average(self.d[:,i])))
            self.d[:,i] -= np.average(self.d[:,i])
            if show_dipoles:
                plt.plot(self.d[:,i])
                plt.show()
        print()
    


    def get_time_correlation_fft(self, normalize = False):
        """
        RETURN THE TIME CORREALTION FUNCTION USING FFT
        ==============================================

        Alternative implementation of the windowed average to get the time correlation function

        Here we use scipy FFT and IFFT. 
        
        We use the padding of the signal, i.e. we double the trajectory and we set the second half to zero

        Parameters:
        -----------
            -normalize: bool, if True we impose the normalization, useful to compare with the Julia implementation

        Returns:
        --------
            -C_t_real: np.array with shape self.N: the time correaltion function of the dipole
        """
        print(' We compute the dipole dipole time correlation function using PYTHON FFT!')
        # Duplicate in size but set all zeros
        d = np.zeros((2 * self.N, 3), dtype = type(self.d[0,0]))
        
        # Set the inital values and the others will be zeros
        d[:self.N,:] = self.d[:,:]
        
        # The P P correlation in Fourier
        d_omega  = scipy.fft.fft(d[:,:], axis = 0)
        
        # Get the modulus squre and perform the dot product
        d2_omega = np.einsum('ia, ia -> i', np.conjugate(d_omega), d_omega)
        
        # Perform the inverse Fourier transform
        C_t = scipy.fft.ifft(d2_omega)

        # Check the imaginary part
        if np.imag(C_t).max() > 1e-5:
            warnings.warn(" WARNING: Discarting imaginary part...")

        if normalize:
            # Apply the normalization (USE only to compare 
            print(' Normalizing the correlation function\n')
            normalization = np.arange(self.N, 0, -1)
            C_t_real = np.real(C_t)[:self.N] /normalization
        else:
            print(' NOT normalizing the correlation function\n')
            C_t_real = np.real(C_t)[:self.N]
        
        return C_t_real

    def set_dipole_dipole_correlation_function(self, use_julia = False, python_normalize = False):
        """
        SET THE DIPOLE DIPOLE CORRELATION FUNCTION
        ==========================================

        This method computes the time correlation function of the dipole updating self.correlations and self.error_correlations

        .. math:: C_{pp}(t) = \left\langle p(t) p(0) \right\rangle

        In JULIA we use the windowed average

        .. math:: C_{pp}(m) = \frac{1}{N - m} \sum_{k=1}^{N - m} p_{k} p_{k + m}

        In PYTHON we compute

        .. math:: p(\omega) = \int dt e^{i\omega t} p(t)

        Then we do 

        .. math:: C_{pp}(t) = \int d\omega e^{-i\omega t} \overline{p}(\omega) p(\omega)

        where m is an integer.

        Parameters:
        -----------
            -use_julia: bool, if True we use the windowed average implemented in julia
        """
        print("\n\nWe are computing the dipole dipole time dependent correlation function!")
        print("Are we using Julia? {}".format(use_julia))

        # Get the correlations
        if use_julia and __JULIA__:
            # Get the julia windowed average 
            self.correlations, self.error_correlations = Main.get_time_correlation_vector(self.d, self.N)
        else:
            # Get the python correlation functions using FFT
            self.correlations = self.get_time_correlation_fft(normalize = python_normalize)
            self.error_correlations = np.zeros(len(self.correlations))


    def get_IR_spectra(self, correlations, delta = None):
        """
        GET IR SPECTRA
        ==============

        We comnpute the IR spectra from the expressions

        ..math: I(\omega) = \frac{2 \pi \beta \omega^2}{3 c V} \int dt \exp{i\omega t} \left\langle p(t) p(0) \right\rangle

        Parameters:
        -----------
            -correlations: np.array: the time correlations
            
        Returns:
        --------
            -omega: np.array, the frequenceis in cm-1
            -IR_spectra: np.array, the intensity of the IR spectra
            -delta: the smearing in cm-1
        """
        print("\n\nGET IR | We are computing the IR spectra from the dipole dipole correlation function!")
        # Get the leght of the time correlation
        length = len(correlations)
        
        # The fft has dimension (eAngstrom)^2
        new_correlations = np.zeros(length * 2, dtype = type(correlations[0]))
        new_correlations[:length] = correlations[:]

        new_correlations[0] = 0.5 * new_correlations[0]

        # Picosecond
        t = np.arange(2 * length) * self.dt
        if delta is None:
            print("GET IR | No smearing!\n")
            IR_omega  = scipy.fft.fft(new_correlations) 
        else:
            # Picoseconds
            tau = 1/(delta * THZ_TO_CM**-1)
            print("GET IR | Using a smearing of {:.2f} cm-1 {:.2f} ps\n".format(delta, tau))
            IR_omega  = scipy.fft.fft(new_correlations * np.exp(- 0.5 * t**2 /tau**2)) 
        print()
        
        # The freq are in Thz
        omega     = scipy.fft.fftfreq(2 * length, self.dt)
        # Go in cm-1
        omega *= THZ_TO_CM

        IR_spectra = np.real(IR_omega) * 2 * np.pi/3

        return omega, omega**2 * IR_spectra


    
        


    def plot_results(self, omega_min_max = [0, 5000], delta = None, save_data = False, data_file = "pp_spectra.json"):
        """
        PLOT THE DIPOLE CORRELATION FUNCTION and THE IR SIGNAL
        ======================================================

        In the first plot we report the dipole dipole correlation function
        In the second plot we report the IR spectra (FFT of the dipole correlation function)

        Returns:
        --------
            -omega_min_max: a list of two float, the minimum and maxium freq to plot in cm-1
            -delta: the smearing in cm-1
        """
        # width heigth
        fig = plt.figure(figsize=(10, 5))

        gs = gridspec.GridSpec(1, 2, figure = fig)

        # PLOT THE DIPOLE DIPOLE TIME CORREALTION FUNCTION
        ax = fig.add_subplot(gs[0,0])
    
        # PICOSECOND and (eAngstrom)^2
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
   
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{pp}(t)$ [(eAngstrom)$^{2}$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)

        # PLOT THE FOURIER TRANSFORM = IR SPECTRA
        ax = fig.add_subplot(gs[0,1])

        # cm-1 and (eAngstrom)^2
        omega, IR_spectra = self.get_IR_spectra(self.correlations, delta = delta)
        # select only a small window
        mask = (omega_min_max[0] < omega) & (omega < omega_min_max[1])
        ax.plot(omega[mask], IR_spectra[mask], lw = 3, color = "k")

        ax.set_xlabel('$\\omega$ [cm$^{-1}$]', size = 15)
        ax.set_ylabel('$C_{pp}(\\omega)$ [Angstrom$^2$/ps$^2$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.tight_layout()

        plt.show()

        if save_data:
            data = {"x" : list(omega), "y" : list(IR_spectra), "delta" : delta}
            AtomicSnapshots.save_dict_to_json(data_file, data)



    # DEBUG

    def test_implementation(self):
        """
        A TEST FUNCTION
        ===============

        Here we compare the correlation function obtained with Julia and the one of python using FFT
        """
        
        # Create base plot
        fig, ax1 = plt.subplots()

        self.set_dipole_dipole_correlation_function(use_julia = False, python_normalize = True)
        res1 = np.copy(self.correlations)
        ax1.plot(self.t,self.correlations, lw = 3, color = "k", label = "Julia WINDOWED")
        ax1.legend()

        self.set_dipole_dipole_correlation_function(use_julia = True, python_normalize = True)
        res2 = np.copy(self.correlations)
        ax1.plot(self.t, self.correlations , color = 'red', lw = 2.0, label = "Python FFT")
        ax1.legend(loc = 'upper left')
        plt.show()

        if np.abs(res1[1:-1] -res2[1:-1]).max() > 10:
            raise ValueError("The correlation function is not implemented correctly")

