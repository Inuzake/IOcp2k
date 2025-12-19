import os, sys

import numpy as np

import scipy, scipy.integrate

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
    print("Conductivity module | Sorry no Julia found. In case you want to use it try with python-jl")


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import AtomicSnap
from AtomicSnap import AtomicSnapshots

import scipy, scipy.optimize, scipy.fft

import copy

import time

ANG2_PS_TO_SI = 1e-8
ANG_PS_TO_SI = 1e+2
KELVIN_TO_EV = 8.6173e-5
_e_charge_   = 1.602176634

THZ_TO_CM = 33

matplotlib.use('tkagg')
class Conductivity:
    """
    GET CONDUCTIVITY FROM ASE ATOMS
    ===============================
    """
    
    def __init__(self, **kwargs):
        """
        DEFINE THE ATTRIBUTES OF THE CLASS
        ==================================

        Units are ANGSTROM PICOSECOND and KELVIN
        """
        #ase atoms object, a list of ase atoms objects
        self.atoms = None
        
        # time step in PICOSECOND
        self.dt = None

        # all the volumes
        self.volumes = None

        # get the temperatures in Kelvin
        self.temperatures = None
        
        # atomic types with oxidation charges
        self.types_q = None

        #int the number of snapshots
        self.N = None

        # The time in PS and the integrated conductivity in Si/m
        # The keys are time sigma_t
        self.results = {}

        # The time correlations, i.e. <j(t) j(0)>
        self.correlations = None

        # The errors for the correlations
        self.error_correlations = None

        # All the currents
        self.js = None


        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
        
    def init(self, ase_atoms = None, dt = 0.5, temperatures = None, types_q = None):
        """
        INITIALIZE
        ==========

        Units are ANGSTROM PICOSECOND and KELVIN

        Paramters:
        ----------
            -ase_atoms: list, list of ase atoms objects
            -dt: float, time step in FEMPTOSECOND
            -temperatures: np.array, the istantnaneous temperature in Kelvin
            -types_q: a dictionary with the atomic type and oxidation charge, example {"Na" : +1, "Cl" : -1}
        """
        if ase_atoms is None:
            raise ValueError("Provide ase atoms in input")

        # Set the ase atoms
        self.atoms = ase_atoms

        # CONVERT in PICOSECOND
        self.dt = dt * 1e-3

        # All the temperature in Kelvin
        self.temperatures = temperatures

        # Get atomic types with the ox charges
        self.types_q = types_q

        # The steps of the trajectory
        self.N = len(self.atoms)

        # time in PICOSECONDS
        self.t = np.arange(self.N) * self.dt

        # in ANGSTROM3
        self.volumes = np.zeros(self.N)

        # The correlations with the errors
        self.correlations = np.zeros(self.N, dtype = type(dt))

        self.error_correlations = np.zeros(self.N, dtype = type(dt))


        # Get all the currents in ANGSTROM/PICOSECONDS
        self.js = np.zeros((self.N, 3), dtype = type(self.dt))


    
    def get_Vcom_from_ase_atoms(self, index):
        """
        GET THE CENTER OF MASS VELOCITIES
        =================================
    
        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object
    
        Returns:
        --------
            -V_com: np.array with shape 3, the center of mass velocity in ANGSROM/PICOSECOND
        """
        # Get all the masses in uma
        masses = np.asarray(self.atoms[index].get_masses())
    
        # Get the center of mass velocity in ANGSTROM/PICOSECOND
        V_com = np.einsum('i, ia -> a', masses, self.atoms[index].get_velocities()[:,:]) /np.sum(masses)
    
        return V_com

        
    def get_selected_atoms_qs(self, index, debug = False):
        """
        SELECT ATOMIC INDICES FROM ASE ATOMS
        ====================================

        Select the atoms using the dictioanry self.types_q

        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object

        Returns;
        --------
            -selected_atoms: np.array with the indices of the selected atoms from self.types_q
            -selected_qs: np.array with the ox charges correpsonding to the selected atoms
        """
        # Get the total number of atoms
        N_at_tot = len(self.atoms[index].positions[:,0])
        
        # Get all the chemical symbols as array
        chem_symb = np.asarray(self.atoms[index].get_chemical_symbols()).ravel()
        
        # Select only the atoms I want
        selected_atoms = []
        # with the corresponign ox charges
        selected_qs    = []
        
        for target_atom in self.types_q.keys():
            # Select the atomic indices corresponding to the atoms that I want
            my_sel = np.arange(N_at_tot)[np.isin(chem_symb, [target_atom])]           
            # Select the atoms contributing to the ionic conductivity
            selected_atoms.append(my_sel)
            # Get the corresponding oxidations charges
            selected_qs.append([self.types_q[target_atom]] * len(my_sel))
            if debug:
                print(target_atom)
                print(my_sel)
                print([self.types_q[target_atom]] * len(my_sel))
                print()
            
        # Transform everything in array
        selected_atoms = np.asarray(selected_atoms).ravel()
        selected_qs    = np.asarray(selected_qs, dtype = int).ravel()

        return selected_atoms, selected_qs

    def get_current_from_ase_atoms(self, index, subtract_vcom = True, debug = False):
        """
        GET THE CURRENT
        ===============
    
        Get the total current for the single snapshopts (ANGSTROM/PICOSECOND)/(ANGSTROM^3)
    
        ..math: J(t) = (\sum_{i=1}^{N} q_{i} v_{i}(t)) /Vol
        
        Parameters:
        -----------
            -index: int, the index of the corresponding atoms object
            -debug: bool
            -subtract_vcom: bool, if True we subtract the COM velocity
    
        Returns:
        --------
            -J_all: np.array with shape (3), the total ionic current in 1/(PS ANGSTROM^3)
        """
        # Select the atoms that contribute to the conductivity with the ox charges corresponind to them
        selected_atoms, selected_qs = self.get_selected_atoms_qs(index, debug = debug)
        
        # if debug:
        print('\n\nBuilding the currents step {}'.format(index))
        print('Selected atoms', np.asarray(self.atoms[index].get_chemical_symbols()).ravel()[selected_atoms])
        print('Selected oxchs', selected_qs)
    
        # Get the velocities of the selected atoms ANGSTROM/PICOSECOND
        selected_velocities = self.atoms[index].get_velocities()[selected_atoms,:]
        
        if subtract_vcom:
            # Subtract the COM velocities ANGSTROM/PICOSECOND
            selected_velocities[:,:] -= self.get_Vcom_from_ase_atoms(index)

        # Mutliply the velocitivies with the charges, ANGSTROM/PICOSECOND
        J = np.einsum('i, ia -> ia', selected_qs, selected_velocities)
    
        if debug:
            print('sel atoms', selected_atoms)
            print('q', selected_qs)
            print('J', J)
            print('V', selected_velocities)
            
        # Sum over all the species, ANGSTROM/PICOSECOND
        J_all = np.einsum('ia -> a', J)
        
        # Divide by the volume 1/(PICOSECOND ANGSTROM^2)
        J_all /= self.atoms[index].get_volume()

        print('\nEnd building the currents\n\n')
            
        return J_all

    def write_flux_file(self, filename = "input"):
        """
        CREATE TABLE FILE FOR SPORTRAN CODE
        ===================================

        Write the input file for the SporTran code

        Parameters
        ----------
            -filename: str, the name of the table file
        """
        full_name = filename + ".table"
        
        print("\n Writing table file for Sportran code...")
        print(" The file will contain temperatures volumes and ionic current for each fram in LAMMPS metal units")
        
        file = open(full_name, "w")
        # Some comments
        file.write("# File generated form IOCP2k package\n")
        file.write("# Metal units of LAMMPS used. temp is Kelvin, volumes in Angstrom^3, flux1 in 1/(ps^2 Angstrom^4)\n")
        # The header of the file
        file.write("temp volumes c_flux1[1] c_flux1[2] c_flux1[3]\n")
        for i in range(self.N):
            file.write("{} {} {} {} {}\n".format(self.temperatures[i], self.volumes[i],
                                                 self.js[i,0], self.js[i,1], self.js[i,2])) 

        file.close()

        print(" File {} written in {}\n".format(full_name, os.getcwd()))

        return
    
    def set_correlations(self, use_julia = False, python_normalize = False, debug = False):
        """
        GET THE CURRENT CURRENT AUTOCORRELATION FUNCTION
        ================================================

        Get the total currents for each MD snapshots and compute the time correlation function

        Parameters:
        -----------
            -use_julia: bool, if True we compute the correlation function using the julia code with the windowed average method
            -python_normalize: bool, if True we normalize the FFT so that it coicides with the windowed average
        """
        # Range on all the snapshots to get the currents
        for i in range(self.N):
            # The currents in 1/(PICOSECOND ANGSTROM^2)
            # sum_i q_i v_i /VOL (v_i are the velocties and VOL is the volume)
            self.js[i,:]    = self.get_current_from_ase_atoms(i, debug = debug)
            # Volumes in ANGSTROM3
            self.volumes[i] = self.atoms[i].get_volume()

        # The correlations in 1/(PICOSECOND ANGSTROM^2)^2
        if use_julia and __JULIA__:
            # Get the Julia windowed average (BRUTE FORCE)
            # The correlations in 1/(PICOSECOND ANGSTROM^2)^2
            self.correlations, self.error_correlations = Main.get_time_correlation_vector(self.js, self.N)
        else:
            # Use pyhon
            self.correlations = self.get_time_correlation_fft(normalize = python_normalize) 
            self.error_correlations = np.zeros(self.N)
            

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
        j = np.zeros((2 * self.N, 3), dtype = type(self.js[0,0]))
        
        # Set the inital values and the others will be zeros
        j[:self.N,:] = self.js[:,:]
        
        # The J J correlation in Fourier
        j_omega  = scipy.fft.fft(j[:,:], axis = 0)
        
        # Get the modulus square and perform the dot product
        j2_omega = np.einsum('ia, ia -> i', np.conjugate(j_omega), j_omega)
        
        # Perform the inverse Fourier transform
        C_t = scipy.fft.ifft(j2_omega)

        # Check the imaginary part
        if np.imag(C_t).max() > 1e-5:
            warnings.warn(" WARNING: Discarting imaginary part...")

        if normalize:
            # Apply the normalization, in this way the FFT correlation function coincides with the windowed average
            print(' Normalizing the correlation function\n')
            normalization = np.arange(self.N, 0, -1)
            C_t_real = np.real(C_t)[:self.N] /normalization
        else:
            print(' NOT normalizing the correlation function\n')
            C_t_real = np.real(C_t)[:self.N]
        
        return C_t_real



    def get_spectra(self, correlations, delta = None, pad = False):
        """
        GET THE CORRELATION FUNCTION SPECTRA
        ====================================

        Get the FT of the current current correlation function

        ..math: C(\omega) = \int dt \exp{i \omega t - delta t} \left\langle j(t) j(0) \right\rangle

        Parameters:
        -----------
            -correlations: np.array: the time correlations
            -delta: the smearing in cm-1
            -pad: bool if True we pad the correlations in input
            
        Returns:
        --------
            -omega: np.array, the frequenceis in cm-1
            -sigma_spectra: np.array, the intensity of the sigma spectra
            
        """
        print("\n\nC[OMEGA] | We are computing the dynamical conductivity!")
        # Get the leght of the time correlation
        length = len(correlations)
        
        # The fft has dimension (eAngstrom)^2
        new_correlations = np.zeros(length * 2, dtype = type(correlations[0]))

        if pad:
            print("C[OMEGA] | PAD")
            new_correlations[:length] = correlations[:]
            new_correlations[0] = 0.5 * new_correlations[0]
        else:
            print("C[OMEGA] | NO PAD")
            new_correlations[:length] = correlations[:][::-1]
            new_correlations[length:] = correlations[:]

        # Picosecond
        t = np.arange(2 * length) * self.dt
        if delta is None:
            print("C[OMEGA] | No smearing!\n")
            new_correlations_omega  = scipy.fft.fft(new_correlations) 
        else:
            # Picoseconds
            tau = 1/(delta * THZ_TO_CM**-1)
            print("C[OMEGA] | Using a smearing of {:.2f} cm-1 = {:.2f} ps\n".format(delta, tau))
            new_correlations_omega  = scipy.fft.fft(new_correlations * np.exp(- t/tau)) 
        print()
        
        # The freq are in Thz
        omega     = scipy.fft.fftfreq(2 * length, self.dt)
        # Go in cm-1
        omega *= THZ_TO_CM

        return omega, np.real(new_correlations_omega) 




    def get_correlation_function(self, omega_min_max = [0, 5000], smearing = None, pad = False,
                                 save_data = False, name_json_file = "sigma.json", show_results = False, name_plot = 'sigma.png'):
        """
        GET AND PLOT THE CURRENT CURRENT CORRELATION FUNCTION
        =====================================================

        First, we plot the current current correlation function in time and frequency domanin

        Secondly, we plot the time integral of the current current correlation function to get the conductivity

        Parameters:
        -----------
            -omega_min_max: list of float, the min and max frequency used to plot the FFT of the current-current correlation function
            -smaering: float, the smearing to use when performing the FFT of the current-current correlation function 
            -pad: bool, if True we pad the time series to get the FFT (used in self.set_correlations)
            -save_data: bool, if True we save the evolution of the conductivity in a json file
            -name_json_file: the name of the json file with the time evolution of conductivity 
            -show_results: bool, if True we make a plot
            -name_plot: str
        """
        if np.all(self.correlations == 0):
            self.set_correlations()

        # # Get the cumulative averages
        # # In ANGSTROM3
        # V_average = np.cumsum(self.volumes)      /np.arange(1, self.N + 1)
        # # In KELVIN
        # T_average = np.cumsum(self.temperatures) /np.arange(1, self.N + 1)
        # Just take the average makes more sense after equilibrated MD 
        # In ANGSTROM3
        V_average = np.average(self.volumes) 
        # In KELVIN
        T_average = np.average(self.temperatures)
        
        # Get the cumulative integral of the current current correlations in 1/(ANGSTROM^4 * PISCOSECOND)
        all_sigma = scipy.integrate.cumulative_trapezoid(self.correlations, x = self.t, dx = self.dt,  initial = 0)
        ########### FACTOR to GET THE CONDUCTIVITY IN Si/m ###########
        # The conversion factor
        e_V_3_T = (1e+3 * _e_charge_ * V_average) /(3 * T_average * KELVIN_TO_EV)
        # Convert in Si/m
        all_sigma *= e_V_3_T
        ###############################################################


        
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, figure = fig)
        
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Current-current time correlation {}".format(self.types_q.keys()))
        # PICOSCOND and 1/(ANGSTROM^4 PICOSECOND^2)
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{JJ}(t)$ [Angstrom$^{-4}$ps$^{-2}$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)

        ax = fig.add_subplot(gs[0,1])

        # Get the Fourier tranform of the conductivity
        # CM-1 and 1/(ANGSTROM^4 PICOSECOND)
        omega, sigma = self.get_spectra(self.correlations, delta = smearing, pad = pad)
        if omega_min_max is None:
            ax.plot(omega, sigma, lw = 3, color = "purple")
        else:
            mask = (omega_min_max[0] < omega) & (omega < omega_min_max[1])
            ax.plot(omega[mask], (sigma)[mask], lw = 3, color = "purple")
        
        ax.set_xlabel('$\\omega$ [cm$^{-1}$]', size = 15)
        ax.set_ylabel('$C_{JJ}(\\omega)$ [Angstrom$^{-4}$ps$^{-1}$]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.tight_layout()
        plt.show()
        plt.close()

        
        ###################
        # MAKE A 2nd PLOT #
        ###################
        
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, figure = fig)
        ax = fig.add_subplot(gs[0,0])
        # PICOSCOND and 1/(ANGSTROM^4 PICOSECOND^2)
        ax.errorbar(self.t, self.correlations, yerr = self.error_correlations, lw = 3, color = "k")
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$C_{JJ}(t)$ [Angstrom$^{-4}$ps$^{-2}$]', size = 15)
        ax.tick_params(axis = 'both', labelsize = 12)


        gs = gridspec.GridSpec(1, 2, figure = fig)
        ax = fig.add_subplot(gs[0,1])
        # PICOSCOND and Si/m
        ax.plot(self.t, all_sigma, color = "red", lw = 3)
        
        ax.set_xlabel('Time [ps]', size = 15)
        ax.set_ylabel('$\\sigma(t)$ [S/m]', size = 15)
        ax.tick_params(axis = 'both', labelsize = 12)
        plt.tight_layout()
        if not(name_plot is None):
            plt.savefig(name_plot, dpi = 500)
            
        if show_results:
            plt.show()
            plt.close()
            
        if save_data:
            data = {"time" : list(self.t), "sigma_t" : list(all_sigma), "delta" : smearing, 'e_V_3_kt' : e_V_3_T}
            AtomicSnapshots.save_dict_to_json(name_json_file, data)

        self.results = data


    # def get_factor(self, mask = None):
    #     """
    #     GET THE PREFACTOR TO COMPUTE THE CONDUCTIVITY IN SI UNITS
    #     =========================================================

    #     We compute the current current corerelation function in Angstrom2/picosecond2

    #     then we need to get the conductivity from in Si/m

    #     ..math:  \sigma(\omega) = \frac{1}{3 V k T} \int dt \left\langle j(t) j(0) \right\rangle
    #     """
    #     if not(mask is None):
    #         # Get volume average ANGSTROM3
    #         V_av = np.average(self.volumes[mask])
    #         # Get temperature average KELVIN
    #         T_av = np.average(self.temperatures[mask])
    #     else:
    #         # Get volume average ANGSTROM3
    #         V_av = np.average(self.volumes[:])
    #         # Get temperature average KELVIN
    #         T_av = np.average(self.temperatures[:])
    #     ########### FACTOR to GET THE CONDUCTIVITY IN Si/m ###########
    #     factor =  (1e+3 * _e_charge_) /(V_av * 3 * T_av * KELVIN_TO_EV)
    #     ###############################################################
        
    #     return factor



    
    # DEBUG

    def test_implementation(self):
        """
        A TEST FUNCTION
        ===============

        Here we compare the correlation function obtained with Julia and the one of python using FFT
        """
        
        # Create base plot
        fig, ax1 = plt.subplots()

        self.set_correlations(use_julia = False, python_normalize = True)
        res1 = np.copy(self.correlations)
        ax1.plot(self.t,self.correlations, lw = 3, color = "k", label = "Julia WINDOWED")
        ax1.legend()

        self.set_correlations(use_julia = True, python_normalize = True)
        res2 = np.copy(self.correlations)
        ax1.plot(self.t, self.correlations , color = 'red', lw = 2.0, label = "Python FFT")
        ax1.legend(loc = 'upper left')
        plt.show()

        if np.abs(res1[1:-1] -res2[1:-1]).max() > 10:
            raise ValueError("The correlation function is not implemented correctly")

