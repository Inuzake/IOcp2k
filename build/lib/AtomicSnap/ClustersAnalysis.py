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



class ClusterAnalysis:
    """
    CLUSTER ANALYSIS in THE MD TRAJECTORY
    =====================================

    This simple class calls a python script to do the cluster analysis.

    It can also read and do some IO on the outputs of the cluster.

    HOW TO USE IT:

    1) initialize it with the path to the ase atoms and the tpyes of atoms we want in the cluster

    2) call run_find_cluster

    3) call run_analysis_cluster
   
    """
    def __init__(self, path_to_ase_atoms = None, types = ["Na", "Cl"], **kwargs):
        """
        INITIALIZE THE CLUSTER CLASS
        ============================

        Parameters:
        -----------
            -path_to_ase_atoms: the path to the extxyz file
            -types: list of atom types forming the cluster
        """
        # The path to the ase atoms (use extxyz so we have the info on the cell)
        self.path_to_ase_atoms = path_to_ase_atoms
        # print('x', os.getcwd())

        # Search starting from /home
        path = find_dir("/home", "IOcp2k")

        if path is None:
            raise ValueError("Could not find the IOcp2k directory")

        # The path to the clustering script (user dependent part)s
        self.path_to_clustering_script = os.path.join(path, "scripts/find_clusters.py") #"/home/antonio/IOcp2k/scripts/find_clusters.py"

        # The path to the clustering analysis script (user dependent part)s
        self.path_to_analysis_script   = os.path.join(path, "scripts/analyze_cluster.py") # "/home/antonio/IOcp2k/scripts/analyze_cluster.py"

        # Setup the types
        self.types = types

        # The path to the result directory
        self.path_to_result_dir = None


        if len(self.types) != 2:
            raise ValueError("The types must be 2 {}".format(self.types))
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])




    def run_find_cluster(self, processors = 1, rcut_min = 0.5, rcut_max = 1, debug = False, current_path = None):
        """
        RUN THE FIND CLUSTERS SCRIPT
        ============================

        Calls an external python script to find at each time step the clusters.

        If the atoms have distance between rcut_min and rcut_max they are considered as clusters

        Parameters:
        -----------
            -processors: int, the number of processors on which we run
            -rcut_min: float, the minimum distance
            -rcut_max: float, the maximum distance, 
            -tpyes: list, the chemical composition of the cluster
            -debug: bool, if True the python script called will output many informations 
        """
        # Get the current execution path
        if current_path is None:
            current_path = os.getcwd()

        # Check the cutoff distances
        if rcut_min > rcut_max:
            raise ValueError("R cut mim {} is larger then R cut max {}".format(rcut_min, rcut_max))

        # Check the processors
        if processors < 1:
            raise ValueError("The number of processors must be greater than 1")
            
        # Command as a list of strings
        command = ["mpirun", "-np", "{}".format(processors), "python3", self.path_to_clustering_script,
                   "{}".format(self.path_to_ase_atoms), "{}".format(rcut_min), "{}".format(rcut_max),
                   "{}".format(int(debug)), "{}".format(self.types[0]) , "{}".format(self.types[1]), "{}".format(current_path)]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # real-time output
            print(line, end='')  
            if line.startswith('!'):
                self.path_to_result_dir = (line.split(' ')[-1]).split('\n')[0]
        
        process.wait()



    def run_analysis_cluster(self, dt = 0.5, debug = False):
        """
        RUN THE CLUSTERING ANALYSIS SCRIPT
        ==================================

        Parameters:
        -----------
            -tpyes: list, the chemical composition of the cluster
            -dt: float, the dt of the MD simulation in FEMPTOSECOND
            -debug: bool, if True the python script called will output many informations 
        """
        # print(
        # Command as a list of strings
        command = [ "python3", self.path_to_analysis_script, "{}".format(self.path_to_result_dir),
                   "{}".format(self.types[0]) , "{}".format(self.types[1]), "{}".format(dt), "{}".format(int(debug))]
        
        # Run the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            # real-time output
            print(line, end='')  
        
        process.wait()

        

def get_path_with_results(output):
    """
    GET THE PATH WITH THE RESULTS
    =============================
    """
    path = None
    for line in output.split('\n'):
        if line.startswith('!'):
            # print(line)
            path = line.split(' ')[-1]
    return path

    