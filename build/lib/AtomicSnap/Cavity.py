import numpy as np

import ase
import copy
import os, sys

import subprocess
import json
import scipy



def find_dir(start_dir, target_dir):
    """
    FIND A TARGET  PATH STARTING FROM START_DIR
    ===========================================
    """
    for root, dirs, files in os.walk(start_dir):
        if target_dir in dirs:
            return os.path.join(root, target_dir)
    return None



class CavityAnalysis:
    """
    CAVITY ANALYSIS in THE MD TRAJECTORY
    =====================================

    This simple class calls a python script to perform the cavity analysis.

    HOW TO USE IT:

    1) initialize it with the path to the ase atoms

    2) call run_find_cavities
   
    """
    def __init__(self, path_to_traj = None, path_to_cells = None, calc_type = "NVT", **kwargs):
        """
        INITIALIZE THE CLUSTER CLASS
        ============================

        Parameters:
        -----------
            -path_to_traj: str, the path to the .traj file
            -path_to_cells: str, the path to the .npy file with the cells
            -calc_type: str, choose among NVT and NPT
            -types: list of atom types forming the cluster
        """
        # The path to the ase atoms (use extxyz so we have the info on the cell)
        self.path_to_ase_atoms = path_to_traj

        if not ".traj" in self.path_to_ase_atoms:
            raise ValueError("The file format should be traj")

        self.path_to_cells = path_to_cells

        # Search starting from /home
        _path_ = find_dir("/home", "IOcp2k")

        if _path_ is None:
            raise ValueError("Could not find the IOcp2k directory")

        # The path to find cavities
        self.path_to_cavity_script = os.path.join(_path_, "scripts/cavity.py") 

        # The path to the script for finding the effective volume of molecules
        self.path_to_filled_cavity_script = os.path.join(_path_, "scripts/filled_cavity.py") 

        # atoms in cavity
        self.path_to_atoms_in_cavity_script = os.path.join(_path_, "scripts/atoms_in_cavity.py") 

        if not calc_type in ["NPT", "NVT"]:
            raise ValueError("calc_type should be eitherr NVT or NPT")

        self.calc_type = calc_type

        # The path to the result directory
        self.path_to_result_dir = None
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])



    def run_find_cavities(self, processors = 1, size = 1, debug = False, current_path = None, atoms_types = ['O']):
        """
        RUN THE FIND CAVITIES SCRIPT
        ============================

        Calls an external python script to find at each time step the cavities of a given size (radius).

        A Cavity is condered so only if there are no atoms of types atoms_types around it

        Parameters:
        -----------
            -processors: int, the number of processors on which we run
            -size: float, the radius of the bubbles in ANGTROM
            -debug: bool, if True the python script called will output many informations 
            -atms_types: list of str, a list with the atoms types that we DO NOT want into the cavity
        """
        # Get the current execution path
        if current_path is None:
            current_path = os.getcwd()

        # Check the processors
        if processors < 1:
            raise ValueError("The number of processors must be greater than 1")
            
        # Command as a list of strings
        command = ["mpirun", "-np", "{}".format(processors), "python3", self.path_to_cavity_script,
                   "{}".format(self.path_to_ase_atoms), 
                   # "{}".format(self.path_to_cells),
                   "{}".format(self.calc_type),
                   "{}".format(size), "{}".format(current_path), "{}".format(int(debug))]

        for at in atoms_types:
            command.append(at)
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # real-time output
            print(line, end='')  
            if line.startswith('!'):
                self.path_to_result_dir = (line.split(' ')[-1]).split('\n')[0]
        
        process.wait()


    def run_filled_cavities(self, processors = 1, size = 1, debug = False, current_path = None, atoms_types = ['O']):
        """
        FILLED CAVITIES
        ===============

        Calls an external python script to find at each time step
        how many atoms of type atoms_types are in a cavity of radius size.

        Useful to get the effective volume occupied by a single water molecule

        Parameters:
        -----------
            -processors: int, the number of processors on which we run
            -size: float, the radius of the bubbles in ANGTROM
            -debug: bool, if True the python script called will output many informations
            -atoms_types: list of str, the atoms we are counting in the cavitites
        """
        # Get the current execution path
        if current_path is None:
            current_path = os.getcwd()

        # Check the processors
        if processors < 1:
            raise ValueError("The number of processors must be greater than 1")
            
        # Command as a list of strings
        command = ["mpirun", "-np", "{}".format(processors), "python3", self.path_to_filled_cavity_script,
                   "{}".format(self.path_to_ase_atoms), 
                   # "{}".format(self.path_to_cells),
                   "{}".format(self.calc_type),
                   "{}".format(size), "{}".format(current_path), "{}".format(int(debug))]
        # Add the atoms types
        for item in atoms_types:
            command.append(item)
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # real-time output
            print(line, end='')  
            if line.startswith('!'):
                self.path_to_result_dir = (line.split(' ')[-1]).split('\n')[0]
        
        process.wait()





    def run_atoms_in_cavity(self, processors = 1, size = 1, debug = False, current_path = None,
                            cluster_info_path = None, selected_atoms = ['Na','Cl']):
        """
        LOOK FOR ATOMS INSIDE CAVITIES
        ==============================

        Calls an external python script to find at each time step how many atoms of seleceted_atoms types
        are inside the cluster of cavities (path to pkl file is cluster_info_path).

        Parameters:
        -----------
            -processors: int, the number of processors on which we run
            -size: float, the radius of the bubbles in ANGTROM
            -debug: bool, if True the python script called will output many informations 
            -cluser_info_path: str: path to the pkl file with the info on the cavity clustering
            -selected_atoms: list of str, the atoms types we look for in the cavity clusering
        """
        # Get the current execution path
        if current_path is None:
            current_path = os.getcwd()

        # Check the processors
        if processors < 1:
            raise ValueError("The number of processors must be greater than 1")
            
        # Command as a list of strings
        command = ["mpirun", "-np", "{}".format(processors), "python3", self.path_to_atoms_in_cavity_script,
                   "{}".format(self.path_to_ase_atoms), 
                   "{}".format(cluster_info_path),
                   "{}".format(size), 
                   "{}".format(current_path),
                   "{}".format(int(debug))]

        for item in selected_atoms:
            command.append(item)

        # print(command)
        # print(command)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # real-time output
            print(line, end='')  
            if line.startswith('!'):
                self.path_to_result_dir = (line.split(' ')[-1]).split('\n')[0]
        
        process.wait()