from ase import units
import ase, ase.io

import os, sys
import numpy as np
import json

import torch

from mace.calculators import MACECalculator
from deepmd.calculator import DP

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import time
import sklearn

import cellconstructor as CC
import cellconstructor.Methods


class AnalyseMace:
    """
    COMPARE ML RESULTS WITH AI
    ==========================

    """
    def __init__(self, **kwargs):
        """
        INITIALIZE THE CLASS
        ===============================

        Parameters
        ----------
            -**kwargs : any other attribute of the ensemble

        """
        # The ML model path to .model of .pb file
        self.model_path = None

        # The path to .xyz file with energies and forces
        self.dft_xyz_path = None

        # The size of the test configurations
        self.size_test = 10

        # Choose configurations to look at
        # self.configs = None

        # The ase atoms object
        self.ase_atoms = None

        # energies forces and stress in ASE units
        # eV eV/Angstrom ev/Angstrom3
        self.dft = None
        self.ml = None

    def initialize(self, select_random = False):
        """
        INITIALIZATION
        ==============
        """
        print("\n\n============= GET THE ASE ATOMS =============\n")
        self.get_ase_atoms(select_random = select_random)
        print("============= END GET THE ASE ATOMS =============\n")
        
    def get_ase_atoms(self, select_random = False):
        """
        GET THE ASE ATOMS
        =================
        """
        self.ase_atoms = ase.io.read(self.dft_xyz_path, index = ":")
        if not self.size_test is None:
            
            if self.size_test < len(self.ase_atoms):
                if not select_random:
                    print('Getting just the first {} configurations'.format(self.size_test))
                    self.ase_atoms = self.ase_atoms[:self.size_test]
                else:
                    print('Random selection of {} configurations'.format(self.size_test))
                    # mask = np.random.uniform(low = 0, high = len(self.ase_atoms) - 1, N = )
                    mask = np.random.randint(low = 0, high = len(self.ase_atoms) - 1, size = self.size_test)
                    tmp_atoms = []
                    for i in mask:
                        tmp_atoms.append(self.ase_atoms[i])
                    self.ase_atoms = tmp_atoms
        else:
            self.size_test = len(self.ase_atoms)
        print("  Size of the test ensemble {}".format(len(self.ase_atoms)))


    def run(self, dir_comparison, select_random = False):
        """
        COMPARE DFT ML ENERGIES FORCES STRESSES AND COMPARE
        ===================================================
        """
        current_path = os.getcwd()
        
        self.initialize(select_random = select_random)
        
        self.get_energies_forces()

        os.mkdir(dir_comparison)
        os.chdir(dir_comparison)
        
        self.plot_compare(show = True)

        file = open('where_test_xyz.txt' , 'w')
        file.write("MODEL PATH\n")
        file.write(self.model_path)
        file.write("\n")
        file.write("TEST XYZ FILE\n")
        file.write(self.dft_xyz_path)
        file.close()

        for key in self.dft.keys():
            np.savez(key + "_dft", self.dft[key])
            np.savez(key + "_ml", self.ml[key])


        self.plot_compare_atom_type_force(show = True)

        os.chdir(current_path)
            
        return


    def re_plot(self, dir_comparison, dir_with_data):
        """
        REPLOT DFT ML ENERGIES FORCES STRESSES AND COMPARE
        ===================================================
        """
        current_path = os.getcwd()
        
        self.load_ml_and_dft(dir_with_data)

        os.mkdir(dir_comparison)
        os.chdir(dir_comparison)
        
        self.plot_compare(show = True)

        os.chdir(current_path)
            
        return

    def load_ml_and_dft(self, dir_with_data):
        """
        LOAD ML AND DFT RESULTS
        =======================

        Load the dictionaries
            self.dft = {"energy" : E0, "force" : padded_F0, "stress" : S0}
            self.ml  = {"energy" : E1, "force" : padded_F1, "stress" : S1}
        """
        current_dir = os.getcwd()
        os.chdir(dir_with_data)

        self.dft = {}
        self.ml = {}

        for key in ["energy", "force", "stress"]:            
            self.dft[key] = np.load(key + "_dft.npz")
            self.ml[key]  = np.load(key + "_ml.npz")

        os.chidir(current_dir)

    def get_energies_forces(self):
        """
        GET DFT and ML ENERGIES FORCES AND STRESSES
        ===========================================
        """
        print("\n\n============= GET THE ENERGIES FORCES STRESSES =============\n")

        # The number of configurations
        configs = len(self.ase_atoms)
        
        # 0 = DFT
        # 1 = ML
        E0 = np.zeros(configs)
        E1 = np.zeros(configs)
        
        F0 = []
        F1 = []
        
        S0 = np.zeros((configs, 3, 3))
        S1 = np.zeros((configs, 3, 3))

        print("\n  ML DFT | Load the calculator {}\n".format(self.model_path))
        if self.model_path.endswith(".model"):
            print("  ML DFT | MACE MODEL")
            calculator = MACECalculator(model_paths = self.model_path, device = 'cpu')
        elif self.model_path.endswith(".pb"):
            print("  ML DFT | DEEPMD MODEL")
            calculator = DP(model = self.model_path, device = "cpu")
        else:
            raise ValueError("MODEL {} NOT VALID".format(self.model_path))

        print()
        for i in range(configs):
            if i%10 == 0:
                print("  ML DFT | Computing configuration {} out of {} with N atoms {}".format(i, self.size_test, len(self.ase_atoms[i])))
            # DFT
            E0[i]     = self.ase_atoms[i].get_total_energy()
            F0.append(self.ase_atoms[i].get_forces())
            _s0_      = self.ase_atoms[i].get_stress()
            S0[i,:,:] = CC.Methods.transform_voigt(_s0_, voigt_to_mat = True)

            # ML
            init_conf = self.ase_atoms[i].copy()
            init_conf.calc = calculator
            E1[i]     = init_conf.get_total_energy()
            F1.append(init_conf.get_forces())
            _s1_      = init_conf.get_stress()
            S1[i,:,:] = CC.Methods.transform_voigt(_s1_, voigt_to_mat = True)
    
        # Padd the forces
        Nmax_atoms_list = [len(f[:,0]) for f in F0]
        Nmax_atoms = max(len(f[:,0]) for f in F0)
        
        print("\n  ML DFT | The maxium number of atoms is {}".format(Nmax_atoms))
        print("  ML DFT | variety of Natoms ")
        print(np.unique(np.asarray(Nmax_atoms_list)))
        print("  ML DFT | We will padd the forces is NC={} NAT={} cart={}".format(configs, Nmax_atoms, 3))
        
        # Prepare the padded results
        padded_F0 = np.zeros((configs, Nmax_atoms, 3))
        padded_F1 = np.zeros((configs, Nmax_atoms, 3))
    
        for i in range(configs):
            padded_F0[i, :len(F0[i]), :] = F0[i]
            padded_F1[i, :len(F1[i]), :] = F1[i]
        
        self.dft = {"energy" : E0, "force" : padded_F0, "stress" : S0}
        self.ml  = {"energy" : E1, "force" : padded_F1, "stress" : S1}
        
        print("\n============= END GET THE ENERGIES FORCES STRESSES =============\n")


    def plot_compare_atom_type_force(self, show = True):
        """
        PLOT THE FORCE DEVIATION ON EACH ATOMIC TYPE
        =============================================

        We assume to work with data on same concentrations
        """
        print("\n\n============= PLOT FORCES PER ATOM =============\n")
        # for i in range(

        # Look for atomic types
        all_atom_types = []
        
        for atoms in self.ase_atoms:
            uniques_chem_sym = np.unique(np.asarray(atoms.get_chemical_symbols()))
            for chem_sym  in uniques_chem_sym:
                if not chem_sym in all_atom_types:
                    all_atom_types.append(chem_sym)

        print("Found the following atomic types")
        print(all_atom_types)

        
        for atom_type in all_atom_types:
            print("Analyzing {}".format(atom_type))
            
            ml_forces = []
            dft_forces = []
            for i, atoms in enumerate(self.ase_atoms):
                mask = np.where(np.asarray(atoms.get_chemical_symbols()) == atom_type)[0]

                ml_forces.append(  self.ml["force"][i,mask,:])
                dft_forces.append(self.dft["force"][i,mask,:])

            ml_forces   = np.asarray(ml_forces).reshape((len(self.ase_atoms), len(mask), 3))
            dft_forces = np.asarray(dft_forces).reshape((len(self.ase_atoms), len(mask), 3))
            
            ##########
            # FORCES #
            ##########

            bins = len(self.dft["energy"])//2
            
            labels = ["X", "Y", "Z"]
            colors = ["red", "green", "purple"]
            # Width and height
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2,3, figure=fig)
        
            for i in range(3):
                ax = fig.add_subplot(gs[0,i])
                
                ax.set_title('TEST Nc={} {}'.format(self.size_test, atom_type), fontsize = 20)
                # # Mean square error
                # mask = self.dft["force"][:,:,i] == 0.
                # mask = ~mask
                # print(mask)
                ref_rmse_f = sklearn.metrics.root_mean_squared_error(ml_forces[:,:,i], dft_forces[:,:,i])
                    
                ax.plot((dft_forces[:,:,i]).ravel(), (dft_forces[:,:,i]).ravel(), color = 'grey', lw = 1, ls = ':')
                hb = ax.hexbin((dft_forces[:,:,i]).ravel(), (ml_forces[:,:,i]).ravel(),
                               cmap = "rainbow", mincnt = 1, gridsize = self.size_test)
                cb = fig.colorbar(hb, ax = ax)
                cb.set_label("Counts", fontsize = 12)
                
                ax.set_ylabel('F ML  [eV/Angstrom]', size = 12)
                ax.set_xlabel('F DFT [eV/Angstrom]', size = 12)
                ax.tick_params(axis = 'both', labelsize = 12)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            
                ax = fig.add_subplot(gs[1,i])
                
                ax.hist(dft_forces[:,:,i].ravel() - ml_forces[:,:,i].ravel(), color = colors[i],
                        bins = bins, label = "RMSE={:.2e} eV/Ang".format(ref_rmse_f))
                
                ax.set_xlabel('F DFT - F ML [eV/Angstrom]', size = 12)
                ax.set_ylabel('Counts', size = 12)
                ax.tick_params(axis = 'both', labelsize = 12)
                ax.legend(fontsize = 12)
                ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
                
            plt.tight_layout()
            plt.savefig("F_{}_{}.png".format(self.size_test, atom_type), dpi = 500)
            if show:
                plt.show()
            plt.close()
                    
        
    def plot_compare(self, show = True):
        """
        PLOT THE RESULTS OF THE TRAINING
        ================================
        """
        print("\n\n============= PLOT ENERGIES FORCES STRESSES =============\n")

        # if 
        bins = len(self.dft["energy"])//2
        Nconfs = len(self.dft["energy"])
        
        ##########
        # ENERGY #
        ##########
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1,2, figure=fig)
        
        ax = fig.add_subplot(gs[0,0])
    
        ref_rmse_e = sklearn.metrics.root_mean_squared_error(self.dft["energy"], self.ml["energy"])
        
        ax.set_title('TEST Nc={}'.format(self.size_test), fontsize = 20)
        ax.plot(self.dft["energy"], self.dft["energy"],  color = 'grey', lw = 1, ls = ':')
        hb = ax.hexbin(self.dft["energy"], self.ml["energy"], gridsize = 50, cmap = "rainbow", mincnt = 1)
        cb = fig.colorbar(hb, ax = ax)
        cb.set_label("Counts", fontsize = 12)
    
        ax.set_ylabel('E ML [eV/atom]', size = 12)
        ax.set_xlabel('E DFT [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    
        ax = fig.add_subplot(gs[0,1])
        ax.hist((self.dft["energy"] - self.ml["energy"]), bins = bins, label = "RMSE={:.2e} eV/atom".format(ref_rmse_e))
        ax.set_xlabel('E DFT - E ML [eV/atom]', size = 12)
        ax.set_ylabel('Counts', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
        ax.legend(fontsize = 12)
        plt.tight_layout() 
        
        plt.savefig("E_{}.png".format(self.size_test), dpi = 500)
        if show:
            plt.show()
        plt.close()
    

        ##########
        # FORCES #
        ##########
        labels = ["X", "Y", "Z"]
        colors = ["red", "green", "purple"]
        # Width and height
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2,3, figure=fig)
    
        for i in range(3):
            ax = fig.add_subplot(gs[0,i])
            ax.set_title('TEST Nc={}'.format(self.size_test), fontsize = 20)
            # Mean square error
            mask = self.dft["force"][:,:,i] == 0.
            mask = ~mask
            # print(mask)
            ref_rmse_f = sklearn.metrics.root_mean_squared_error(self.dft["force"][:,:,i][mask], self.ml["force"][:,:,i][mask])
                
            ax.plot((self.dft["force"][:,:,i][mask]).ravel(), (self.dft["force"][:,:,i][mask]).ravel(), color = 'grey', lw = 1, ls = ':')
            hb = ax.hexbin((self.dft["force"][:,:,i][mask]).ravel(), (self.ml["force"][:,:,i][mask]).ravel(),
                           cmap = "rainbow", mincnt = 1, gridsize = self.size_test)
            cb = fig.colorbar(hb, ax = ax)
            cb.set_label("Counts", fontsize = 12)
            
            ax.set_ylabel('F ML  [eV/Angstrom]', size = 12)
            ax.set_xlabel('F DFT [eV/Angstrom]', size = 12)
            ax.tick_params(axis = 'both', labelsize = 12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
            ax = fig.add_subplot(gs[1,i])
            ax.hist(self.dft["force"][:,:,i].ravel() - self.ml["force"][:,:,i].ravel(), color = colors[i],
                    bins = bins, label = "RMSE={:.2e} eV/Ang".format(ref_rmse_f))
            ax.set_xlabel('F DFT - F ML [eV/Angstrom]', size = 12)
            ax.set_ylabel('Counts', size = 12)
            ax.tick_params(axis = 'both', labelsize = 12)
            ax.legend(fontsize = 12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
        plt.tight_layout()
        plt.savefig("F_{}.png".format(self.size_test), dpi = 500)
        if show:
            plt.show()
        plt.close()
    
        ##########
        # STRESS #
        ##########
        # Width and height
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2,3, figure=fig)
    
        for i in range(3):
            ax = fig.add_subplot(gs[0,i])
            ax.set_title('TEST Nc={}'.format(self.size_test), fontsize = 20)
            # Mean square error
            ref_rmse_s = sklearn.metrics.root_mean_squared_error(self.dft["stress"][:,i,i], self.ml["stress"][:,i,i])
            
            ax.plot(self.dft["stress"][:,i,i].ravel(), self.dft["stress"][:,i,i].ravel(),  color = 'grey', lw = 1, ls = ':')
            hb = ax.hexbin(self.dft["stress"][:,i,i].ravel(), self.ml["stress"][:,i,i].ravel(),
                           gridsize = self.size_test , cmap = "rainbow", mincnt = 1)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Counts", fontsize = 12)
            
            ax.set_ylabel('$\\sigma$ ML  [eV/Angstrom$^{3}$]', size = 12)
            ax.set_xlabel('$\\sigma$ DFT [eV/Angstrom$^{3}$]', size = 12)
            ax.tick_params(axis = 'both', labelsize = 12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        
            ax = fig.add_subplot(gs[1,i])
            ax.hist(self.dft["stress"][:,i,i].ravel() - self.ml["stress"][:,i,i].ravel(), color = colors[i],
                    bins = bins, label = "RMSE={:.2e} eV/Ang3".format(ref_rmse_s))
            ax.set_xlabel('$\\sigma$ DFT - $\\sigma$ ML [eV/Angstrom$^{3}$]', size = 12)
            ax.set_ylabel('Counts', size = 12)
            ax.tick_params(axis = 'both', labelsize = 12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins = 4))
            ax.legend(fontsize = 12)
            
        plt.tight_layout()
        plt.savefig("S_{}.png".format(self.size_test), dpi = 500)
        if show:
            plt.show()
        plt.close()

        print("\n============= END PLOT ENERGIES FORCES STRESSES =============\n")
            

        
        