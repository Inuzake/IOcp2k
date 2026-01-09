import numpy as np

import ase
from ase import Atoms

import os, sys

import copy

import json

import subprocess

from datetime import datetime


class AtomicBunches:
    """
    CREATE BUNCHES OF CP2K INPUT FILES TO RUN ON YOUR FAVORITE CLUSTER
    ===================================================================

    """
    def __init__(self, **kwargs):
        """
        INITIALIZE THE ATOMIC BUNCHES
        =============================

        The positions of the atoms are in Angstrom

        We generate batches for MD trajectories using the restarting option of cp2k
        
        Parameters
        ----------
            -**kwargs : any atttribute

        """
        # The number of batches
        self.n_batches = 0
        # The minimum number of batches 
        # the first batch that will be generated
        self.n_batches_min = 1

        # The structure file to initialize everything use .xyz data in angstrom units
        self.structure_file = None

        # int 0=False 1=True
        self.restart = False
        self.restart_file = None
        
        self.local_path = None
        
        ###########
        # CLUSTER #
        ###########
        # Create the dictionary
        self.cluster_dict = { # Cluster name
                             "cluster_name" : "sicilana@irene-fr.ccc.cea.fr",
                              # The pwd
                             "pwd" : "30Luglio1996!",
                             # The cluser directory where all the calculations will be done
                              "cluster_scratch" : None,
                              # THE PART FOR THE JOB FILE
                              # job name
                              'job_name' : 'brines',
                              # partition name
                              'partition_name' : 'rome',
                             # The number of nodes
                              'nnodes' : 1,
                             # The number of cpus
                              'ncpus' : 1, 
                              # No requeue
                              'no_requeue' : True,
                              # The qos
                              'qos' : 'normal',
                               # The execution time in SECONDS or in HH:MM:SS depending on the cluster
                              'time' : 10,
                               # The account
                              'account' : 'gen2309',
                              # set the -x op MSUB
                              'minus_x' : True, 
                              # The module load part before running the calculation
                              'module_load' : """
                              
module switch feature/openmpi/net/ib/ucx-noxpmem\n
module load gnu/11 mpi/openmpi/4 dbcsr/2.7.0\n
module load cp2k/2024.3\n
                              """,
                              # the mpirun command
                              'mpirun' : "ccc_mprun ",
                              # the submission command fro job file (run.sh)
                              'run_job' : "ccc_msub ",
                              # the executable
                              'exe' : "cp2k.psmp "}

        ########
        # CP2k #
        ########
        # Create the dictionary
        self.cp2k_dict = {# TYPE OF RUN
                          "_RTYPE_" : "MD",
                          # TYPE OF ENSEMBLE
                          "_CALCTYPE_" : "NVT", 
                          # SYSTEM NAME IN CP2k
                          "_SYSTEM_" : "brines",
                          # IF YOU WANT TO RESTART THE CALCULATION SET RESTART TO 1
                          "_REST_" : 0, "_RES_FILE_" : "initial.restart", "_RES_WFN_FILE_" : "initial.wfn",
                          # THE FULL RESTART FILE PATH TO KEEP TRACK OF WHAT WE DO
                          "_full_RES_FILE_" : None,
                          # Where to find the basis set and basis and pseudo file
                          "_BASIS_POT_PATH_" :  "/ccc/work/cont003/gen2309/sicilana/DATA_CP2K", "_BASIS_FILE_" : "GTH_BASIS_SETS",
                          "_POT_FILE_" : "GTH_POTENTIALS",
                          # VDW INTERACTION, VDW FUNCTIONAL and WHICH ATOM TO EXCLUDE FROM VDW
                          "_VDW_" : 1, "_VDW3_FUNCTIONAL_" : None, "_VdW_EXCLU_" : 0, "_VDW3_EXCLUDE_ATOM_" : 3,
                          "_RCvdw_" : 12, 
                          # THE SMOOTHING OF THE DENSITY
                          "_XC_SMOOTH_RHO_" : "NONE", "_XC_DERIV_" :  "PW",
                          # CUTOFF in RYDBERG AND NUMBER OF GRIDS
                          "_CUTOFF_" : 600, "_REL_CTOFF_" : 60, "_NGRIDS_" : 4, 
                          # The functional
                          "_CP2K_XC_FUNCTIONAL_" : None,
                          # ADMM (for hybrid functional)
                          "_ADMM_" : 0, "_BASIS_AUX_FILE_" : "BASIS_ADMM_UZH",
                          # Extrapolation strategy for the wavefunction
                          "_EXTRAPOLATION_" : "ASPC", "_EXTRA_ORDER_" : "3",
                          # GPAW
                          "_USE_GAPW_" : 1,
                          # OT PARAMETERS
                          "_OT_PRECONDITIONER_" : "FULL_SINGLE_INVERSE",
                          "_OT_MINIMIZER_" : "DIIS",
                          # THE PARAMETERS OF THE STRUCTURE in ANGSTROM (IF RESTART IS 1 then this is override)
                          "_A_" : 0.0, "_B_" : 0.0, "_C_" : 0.0,
                          # If THERE NO RESTART FILE IS GIVEN THE SIMULATION WILL START FROM THIS STRUCTURE FILE
                           "_COORD_FILE_FORMAT_" : "xyz", "_COORD_FILE_NAME_" : "structure.xyz",
                          # KINDS ATOMS AND BASIS SET
                           "_KINDS_BASIS_SET_" : None,
                          # MD TIMESTEPS dt in FEMPTOSECONDS and NUMBER OF STEPS
                          "_TIMESTEP_" : 0.5, "_STEPS_" : 1,
                          # THERMOSTAT T in KELVIN and TIMECONSTANT IN FEMPTOSECOND
                          "_TEMPERATURE_" : None, "_TIMECONCSVR_" : None,
                          # BAROSTAT PRESSURE in BAR TIME OCNSTANT in FEMPTOSECOND: NPT_I 
                          "_PRESSURE_" : None, "_TIMECONCONPRESS_" : None,
                          # POLARIZATION
                          "_USE_BERRY_" : 0,
                          # COMPUTE HOMO LUMO GAPS EVERY STEPS
                          "_N_HL_GAL_PRINT_" : 200}

        # tHIS IS SET TO TRUE AFTER CALLING initialize
        self.initialized = False
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])



    def initialize(self, where_dict_cluster, where_dict_cp2k):
        """
        INITALIZE ALL THE QUANTITIES NEEDED FOR THE CALCULATIONS
        ========================================================

        Parameters:
        ----------
            -where_dict_cluster: the path to the json file with the dictionary used to create the submission file
            -where_dict_cp2k: the path to the json file with  the cp2k dictionary
        """
        # Load the dictionaries for the cluster and the cp2k calculations
        cluster_dict = load_dict_from_json(where_dict_cluster)

        # Load the dictionaries for the cluster and the cp2k calculations
        cp2k_dict    = load_dict_from_json(where_dict_cp2k)

        # Check for compatibility
        if len(cp2k_dict)    != len(self.cp2k_dict):
            raise ValueError('The two cp2k dictionaries do not match')

        if len(cluster_dict) != len(self.cluster_dict):
            raise ValueError('The two cluster dictionaries do not match')

        # Check consistency between the keys
        for key in self.cp2k_dict.keys():
            if not(key in cp2k_dict):
                raise ValueError('The cp2k dictionary has {} missing'.format(key))

        # Check consistency between the keys
        for key in self.cluster_dict.keys():
            if not(key in cluster_dict):
                raise ValueError('The cluster dictionary has {} missing'.format(key))

        # Updates the dictionaries
        self.cp2k_dict    = copy.deepcopy(cp2k_dict)

        self.cluster_dict = copy.deepcopy(cluster_dict)

        if "NPT" in self.cp2k_dict["_CALCTYPE_"] and bool(self.cp2k_dict["_XC_SMOOTH_RHO_"]!="NONE"):
            raise ValueError("The smoothing procedure to get the pressure should be tested")

        if "NVT" in self.cp2k_dict["_CALCTYPE_"] and not bool(self.cp2k_dict["_XC_SMOOTH_RHO_"]!="NONE"):
            #raise ValueWarning("In NVT considering the smoothing procedure")
            print("WARNING: consider using the smoothing procedure in NVT")
        # Check that the CP2k functional and the parameterization of the VDW are consistent
        # Otherwis an error wil be raised
        dft_functional_used  = self.cp2k_dict["_CP2K_XC_FUNCTIONAL_"].split()[1]
        # Check for a specific parametrization of the XC functional
        if "PARAMETRIZATION" in self.cp2k_dict["_CP2K_XC_FUNCTIONAL_"].split():
            index = self.cp2k_dict["_CP2K_XC_FUNCTIONAL_"].split().index("PARAMETRIZATION")
            dft_functional_used = self.cp2k_dict["_CP2K_XC_FUNCTIONAL_"].split()[index + 1]
        if "SCALE_C" in self.cp2k_dict["_CP2K_XC_FUNCTIONAL_"].split():
            dft_functional_used = "PBE0"
        if self.cp2k_dict["_VDW3_FUNCTIONAL_"] != dft_functional_used:
            raise ValueError("The DFT xc is {} whereas the D3 parametrization is {}".format(dft_functional_used, self.cp2k_dict["_VDW3_FUNCTIONAL_"]))

        if not ".xyz" in self.structure_file:
            raise ValueError("Please use a xyz file structure in ANGSTROM")

        self.initialized = True

        return 




    def create_inp_cp2k(self, dictionary):
        """
        CREATE THE INPUT FOR NPT SIMULATIONS in CP2K
        """

        input_text = """
@SET RESTART        _REST_
@SET RTYPE          _RTYPE_
@SET BASIS_POT_PATH _BASIS_POT_PATH_
@SET BASIS_FILE     _BASIS_FILE_
@SET BASIS_AUX_FILE _BASIS_AUX_FILE_ 
@SET POT_FILE       _POT_FILE_
@SET SYSTEM         _SYSTEM_
@SET VDW            _VDW_
@SET VDW_EXCLU      _VdW_EXCLU_
@SET USE_GAPW       _USE_GAPW_
@SET PRINT_P_BERRY  _USE_BERRY_
@SET PRINT_HL_GAP   _N_HL_GAL_PRINT_
@SET ADMM           _ADMM_
        
&GLOBAL
  PROJECT     ${SYSTEM}
  RUN_TYPE    ${RTYPE}
  PRINT_LEVEL LOW
  FLUSH_SHOULD_FLUSH 
&END GLOBAL

&FORCE_EVAL

  METHOD QuickStep

  @IF ( ${RTYPE} /= ENERGY_FORCE)
  STRESS_TENSOR ANALYTICAL
  @ENDIF

  &DFT
    BASIS_SET_FILE_NAME ${BASIS_POT_PATH}/${BASIS_FILE}
    @IF ADMM
    BASIS_SET_FILE_NAME ${BASIS_POT_PATH}/${BASIS_AUX_FILE}
    @ENDIF
    POTENTIAL_FILE_NAME ${BASIS_POT_PATH}/${POT_FILE}

    @IF ${RESTART}
    WFN_RESTART_FILE_NAME _RES_WFN_FILE_
    @ENDIF

    &MGRID
      CUTOFF [Ry]       _CUTOFF_
      NGRIDS            _NGRIDS_
      REL_CUTOFF [Ry]   _REL_CTOFF_
    &END MGRID

    &QS
      EPS_DEFAULT 1.0E-14    # def=1.0E-10
      EXTRAPOLATION _EXTRAPOLATION_    #Extrapolation strategy for the wavefunction, ASPC recommended for MD, PS for SPE
      EXTRAPOLATION_ORDER _EXTRA_ORDER_   #Default is 3
      @IF ${USE_GAPW}
          METHOD GAPW          # Gaussian Augumented Plane Waves
          QUADRATURE   GC_LOG  # Algorithm to construct the atomic radial grid for GAPW
          EPSFIT       1.E-6   # Precision to give the extension of a hard gaussian
          EPSISO       1.0E-12 # Precision to determine an isolated projector
          EPSRHO0      1.E-8   # Precision to determine the range of V(rho0-rho0soft)
          # LMAXN0       4
          # LMAXN1       6
          # ALPHA0_H     10 # Exponent for hard compensation charge
      @ENDIF
    &END QS

    &SCF
      @IF ${RESTART}
      SCF_GUESS RESTART
      @ENDIF
      EPS_SCF 1.0E-7 # def=1.0E-5 the exponent should be half of EPS_DEFAULT
      MAX_SCF 50   # def=50
      &OUTER_SCF
        EPS_SCF 1.0E-7 # def=1.0E-5
        MAX_SCF 50
      &END OUTER_SCF
      &OT
        PRECONDITIONER _OT_PRECONDITIONER_ # Example FULL_SINGLE_INVERSE, FULL_KINETIC
        MINIMIZER      _OT_MINIMIZER_      # Example DIIS
      &END OT
    &END SCF

    &XC

      _CP2K_XC_FUNCTIONAL_
      
      &XC_GRID
         XC_SMOOTH_RHO  _XC_SMOOTH_RHO_
         XC_DERIV       _XC_DERIV_
      &END XC_GRID

      @IF ${VDW}
      &vdW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
#          CALCULATE_C9_TERM .TRUE. # Include the 3-body term
#          REFERENCE_C9_TERM .TRUE.
          TYPE DFTD3
          LONG_RANGE_CORRECTION .TRUE.
          PARAMETER_FILE_NAME ${BASIS_POT_PATH}/dftd3.dat
          VERBOSE_OUTPUT .TRUE.
          REFERENCE_FUNCTIONAL _VDW3_FUNCTIONAL_
          R_CUTOFF [angstrom] _RCvdw_ # def=10 angstrom
          EPS_CN 1.0E-6  # def=1.0E-6 dp cutoff value for coordination number function
          @IF ${VDW_EXCLU}
            D3_EXCLUDE_KIND _VDW3_EXCLUDE_ATOM_ # Exclude the Na atom type 3
          @ENDIF
        &END PAIR_POTENTIAL
      &END vdW_POTENTIAL
      @ENDIF
    &END XC

    @IF ADMM
    &AUXILIARY_DENSITY_MATRIX_METHOD
      ADMM_TYPE ADMM2
      EXCH_CORRECTION_FUNC PBEX
    &END AUXILIARY_DENSITY_MATRIX_METHOD
    @ENDIF

    &PRINT
        &MO_CUBES
            &EACH
              @IF (${RTYPE} = ENERGY_FORCE)
              JUST_ENERGY  ${PRINT_HL_GAP}
              @ENDIF
              @IF (${RTYPE} = MD)
              MD ${PRINT_HL_GAP}
              @ENDIF
            &END EACH
            NHOMO        2
            NLUMO       10
            WRITE_CUBE   FALSE
        &END MO_CUBES

        @IF ${PRINT_P_BERRY}
        &MOMENTS ON
            COMMON_ITERATION_LEVELS 20000
            FILENAME =${SYSTEM}-1.dipoles
            ADD_LAST NUMERIC
            REFERENCE COM
            &EACH
              @IF (${RTYPE} = ENERGY_FORCE)
              JUST_ENERGY 1
              @ENDIF
              @IF (${RTYPE} = MD)
              MD 1
              @ENDIF
            &END EACH
        &END MOMENTS
        @ENDIF
        
    &END PRINT

  &END DFT

  @IF (${RTYPE} = ENERGY_FORCE )
  &PRINT
    &FORCES
      &EACH JUST_ENERGY
      &END EACH
      ADD_LAST NUMERIC
    &END FORCES
  &END PRINT
  @ENDIF

  &SUBSYS

    &CELL
      ABC [angstrom]     _A_ _B_ _C_
    &END CELL

    &TOPOLOGY
      CONNECTIVITY OFF
      COORD_FILE_FORMAT _COORD_FILE_FORMAT_
      COORD_FILE_NAME   ./_COORD_FILE_NAME_
    &END TOPOLOGY

    _KINDS_BASIS_SET_

  &END SUBSYS

&END FORCE_EVAL

@IF ( ${RTYPE} /= ENERGY_FORCE)
&MOTION
  &MD
    ENSEMBLE      _CALCTYPE_
    STEPS             _STEPS_
    TIMESTEP [fs]     _TIMESTEP_
    TEMPERATURE [K]   _TEMPERATURE_
    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON [fs]  _TIMECONCSVR_
      &END CSVR
    &END THERMOSTAT
    &BAROSTAT
	   PRESSURE [bar]  _PRESSURE_
     TIMECON  [fs]   _TIMECONCONPRESS_
    &END BAROSTAT
  &END MD

  &PRINT
    &TRAJECTORY  SILENT
      FILENAME =${SYSTEM}-1.xyz
      &EACH
        MD 1
      &END EACH
    &END TRAJECTORY

    &FORCES  SILENT
      FILENAME =${SYSTEM}-1.force
      &EACH
        MD 1
      &END EACH
    &END FORCES

    &VELOCITIES SILENT
    	FILENAME =${SYSTEM}-1.vel
        &EACH
            MD 1
        &END EACH
    &END VELOCITIES

    &CELL  SILENT
      FILENAME =${SYSTEM}-1.cell_xyz
      &EACH
        MD 1
      &END EACH
    &END CELL

    &STRESS SILENT
    	FILENAME =${SYSTEM}-1.stress
        &EACH
            MD 1
        &END EACH
    &END STRESS
    
    &RESTART
      FILENAME =${SYSTEM}-1.restart
      &EACH
        MD 1
      &END EACH
    &END RESTART
    
    &RESTART_HISTORY SILENT
      &EACH
        MD 50
      &END EACH
    &END RESTART_HISTORY
  &END PRINT

&END MOTION
@ENDIF

@if ${RESTART}
&EXT_RESTART
  RESTART_FILE_NAME _RES_FILE_
  RESTART_COUNTERS    T
  RESTART_AVERAGES    T
  RESTART_POS         T
  RESTART_VEL         T
  RESTART_THERMOSTAT  T
  RESTART_BAROSTAT    T
&END EXT_RESTART
@endif
            """

        def is_all_upper(s):
            return s.isupper() and len(s) > 0
        print("\nCREATING INPUT")
        for key, value in dictionary.items():
            # if is_all_upper(key):
            print(f"{key} => {value}")
            pre_input_text = copy.deepcopy(input_text)
            input_text = input_text.replace(key, "{}".format(value))
            # print(input_text == pre_input_text)
            # print(input_text)
            if input_text == pre_input_text and not(key in ["_full_RES_FILE_"]):
                raise ValueError("KEY {} NOT FOUND, please check the text of the cp2k calculation".format(key))

        
        file = open("input.inp", "w")
        file.write(input_text)
        file.close()



   
    def create_sh_copy(self, dirs_to_copy):
        """
        CREATE AN SH FILE TO COPY EVERYTHING TO THE CLUSTER
        """
        if os.path.isfile("copy.sh"):
            raise ValueError("The file already exists")
            
        new_file = open("copy.sh", "w")
        
        new_file.write("#!/bin/bash \n")
        for item in dirs_to_copy:
           if self.cluster_dict["pwd"] is None:
               line = "scp -r {} {}:{} \n wait\n".format(item, self.cluster_dict["cluster_name"], self.cluster_dict["cluster_scratch"] )
           else:
               line = "sshpass -p {} scp -r {} {}:{} \n wait\n".format(self.cluster_dict["pwd"], item, self.cluster_dict["cluster_name"], self.cluster_dict["cluster_scratch"] )
           new_file.write(line)
        
        new_file.close()
    
        return


    
    def create_sh_retrive(self, dirs_to_copy):
        """
        CREATE AN SH FILE TO RETRIVE EVERYTHING FROM THE CLUSTER
        """
        if os.path.isfile("retrive.sh"):
            raise ValueError("The file already exists")
            
        new_file = open("retrive.sh", "w")
        
        new_file.write("#!/bin/bash \n")
        for item in dirs_to_copy:
           if self.cluster_dict["pwd"] is None:
               line = "scp -r {}:{}/{} ./ \n wait\n".format(self.cluster_dict["cluster_name"], self.cluster_dict["cluster_scratch"], item)
           else:
               line = "sshpass -p {} scp -r {}:{}/{} ./ \n wait\n".format(self.cluster_dict["pwd"], self.cluster_dict["cluster_name"], self.cluster_dict["cluster_scratch"], item)
           new_file.write(line)
        
        new_file.close()
    
        return




    def make_batches(self, custom_cluster_function = "IRENE"):
        """
        CREATE THE BATCHES 
        ===================

        This function creates batche from self.n_batches_min to self.n_batches + self.n_batches_min

        Each batch will contain an MD simulation to run on a supercomputer

        After each batch has finisched the restart file is used to start the nest simulation


        Parameters:
        -----------
            -custom_cluster_function: a function that takes as input the self.cluster_dict and create the cluster bash job
        """
        if not self.initialized:
            raise ValueError('Before creating the batche run the initialize method')
            
        if custom_cluster_function == "IRENE":
            print("\nRUNNING ON IRENE\n")
            custom_cluster_function = self.create_run_file_irene
        elif custom_cluster_function == "PARACELSUS":
            print("\nRUNNING ON PARACELSUS\n")
            custom_cluster_function = self.create_run_file_paracelsus

        # A list with all the execution dir containing a run.sh file
        execution_dir_list = []

        # Add the +1 to have exactly n_batches directories
        for batch in range(self.n_batches_min, (self.n_batches + self.n_batches_min) ):
        
            # The execution directory
            if self.cp2k_dict["_RTYPE_"] == "MD":
              execution_dir =  "BATCH_{:d}_MD_T_{:d}_steps_{:d}_dt_{:.1f}_".format(batch, self.cp2k_dict["_TEMPERATURE_"],
                                                                                    self.cp2k_dict["_STEPS_"],
                                                                                    self.cp2k_dict["_TIMESTEP_"])
              if self.cp2k_dict["_CALCTYPE_"] == "NPT_I":
                  execution_dir =  "BATCH_{:d}_MD_T_{:d}_P_{:d}_steps_{:d}_dt_{:.1f}_".format(batch, self.cp2k_dict["_TEMPERATURE_"],
                                                                                              self.cp2k_dict["_PRESSURE_"],
                                                                                              self.cp2k_dict["_STEPS_"],
                                                                                              self.cp2k_dict["_TIMESTEP_"])
            elif self.cp2k_dict["_RTYPE_"] == "ENERGY_FORCE":
                execution_dir = f"BATCH_{batch:d}_SPE_conf_{self.cp2k_dict['_STEPS_']:d}_"
            else:
                print(f"Run type {self.cp2k_dict["_RTYPE"]} not implemented")
                raise NotImplementedError("Please, choose between MD or ENERGY_FORCE")
                
            # Add the structure file without the extension
            execution_dir += os.path.basename(self.structure_file)[:-4] 
            execution_dir_list.append(execution_dir)
            print("\n\n=> The execution dir will be {}\n".format(execution_dir))

            # If it is the first batch check if the calculation needs to be restarted from somewhere else
            # self.restart and self.restart file must be changed by the user
            if batch == self.n_batches_min:
                # Update this variable of cp2k dictionary
                print('Hello! This is batch {} we will start the calculation from\nFILE={}'.format(self.n_batches_min, self.restart_file))
                SURE = input('Are you ok with this decision? YES or NO ')
                if SURE == "NO":
                    raise ValueError("The restart file is wrong according to you! The creation of the batches has been killed!")
                
                # Check if the scratch directory is ok
                print('The scratch directory will be\n{}'.format(self.cluster_dict["cluster_scratch"]))
                SURE_scratch = input('Are you ok with this decision? YES or NO ')
                if SURE_scratch == "NO":
                    raise ValueError("The scratch dir is not correct according to you! The creation of the batches has been killed!")
                self.cp2k_dict["_REST_"]       = int(np.copy(self.restart))
                self.cp2k_dict["_full_RES_FILE_"] = copy.deepcopy(self.restart_file)
 
            # If it is not the first batch we should force the restart from the previous batch
            else:
                # Update this variable of cp2k dictionary
                self.cp2k_dict["_REST_"]       = 1
                # self.cp2k_dict["_full_RES_FILE_"] = os.path.join(self.cluster_dict["cluster_scratch"], execution_dir_list[-2])
                self.cp2k_dict["_full_RES_FILE_"] = os.path.join(self.local_path, execution_dir_list[-2])
           
            
            # Create the execution dir
            os.mkdir(execution_dir) 
            os.chdir(execution_dir)
            # Check where I am
            print("\n==> Right now I am in {}".format(os.getcwd()))
        
            # Copy the structure file previously built .xyz (obtained from a structural relaxation)
            subprocess.run(["cp", self.structure_file, os.path.join("./", self.cp2k_dict["_COORD_FILE_NAME_"])], check = True)

            # Copy the restart file of a previous batch if this is the first batch to submit
            if batch == self.n_batches_min:
                if self.restart:
                    #Copy the restart file previously built
                    subprocess.run(["cp", self.restart_file, "./{}".format(self.cp2k_dict["_RES_FILE_"])], check=True)  
             

            # Create the cp2k input
            if   self.cp2k_dict["_CALCTYPE_"] == "NVT":
                self.create_inp_cp2k(self.cp2k_dict)
            elif self.cp2k_dict["_CALCTYPE_"] == "NPT_I":
                self.create_inp_cp2k(self.cp2k_dict)
            else:
                raise NotImplementedError("NPT_I and NVT are the only implemented")
        

            # Save the dictionary with all the input data
            save_dict_to_json("param_cp2k.json"   , self.cp2k_dict)
            save_dict_to_json("param_cluster.json", self.cluster_dict)
            save_dict_to_json("when_created.json" , {"when_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            
            # Create the .sh file to run all files in one shot
            if batch == self.n_batches + self.n_batches_min - 1:
                # If this is the last batch we set batch_index to -1 so at the end it does not copy the restart file for the next job
                custom_cluster_function(-1,    execution_dir)
            else:
                custom_cluster_function(batch, execution_dir)
                        
            os.chdir("../")
        
    
        # Create the copy.sh file to easility copy all the new dirs
        self.create_sh_copy(execution_dir_list)
        self.create_sh_retrive(execution_dir_list)
        subprocess.run(["chmod", "+x", "./copy.sh"], check=True)    
        subprocess.run(["chmod", "+x", "./retrive.sh"], check=True)    


    
#     def create_run_file_generic(self, batch_index, execution_dir):
#         """
#         CREATES THE RUN.SH FILE FOR SLURM SCHEDULER
#         ===========================================

#         Parameters:
#         -----------
#             -batch_index: int used as label for the job name and for submitting the next batch calculation
#             -execution_dir: the dir containing the run.sh file and all the inputs needed by cp2k
#         """
    
#         avail_partitions = ["QC", "MD"]
#         # Chek if the partition is ok
#         if not (self.cluster_dict["partition_name"] in avail_partitions):
#             raise ValueError("Partition name not valid")
    
#         # myQOS = "normal"
#         # if self.cluster_dict["time"] > 60 * 60 * 24:
#         #     myQOS = "long"
    
#         # Check if the numer of NODES are correct
#         if self.cluster_dict["partition_name"] == avail_partitions[0]:
#             if self.cluster_dict["ncpus"] % 32 != 0:
#                 exp_nodes = 1 + self.cluster_dict["ncpus"] // 32 
#             else:
#                 exp_nodes = self.cluster_dict["ncpus"] // 32
            
        
#         if exp_nodes != self.cluster_dict["nnodes"]:
#             print("GENERIC| The expected number of nodes is {} but you choose {} for ncpus {}", exp_nodes,  self.cluster_dict["nnodes"],  self.cluster_dict["ncpus"])
#             raise ValueError("GENERIC| The number of nodes is not correct, the job will crash")
    
        
#         file = open("run.sh", "w")
        
#         file.write("""
# #!/bin/bash
# #SBATCH -J {}_{:d}
# #SBATCH -p {}   
# #SBATCH -N {:d}
# #SBATCH -n {:d}
# #SBATCH --time={}  # HH:MM:SS
        
# {}
        
# """.format(self.cluster_dict["job_name"], batch_index,
#                self.cluster_dict["partition_name"],
#                self.cluster_dict["nnodes"],
#                self.cluster_dict["ncpus"], 
#                self.cluster_dict["time"],
#                self.cluster_dict["module_load"]))
        
        
#         allfiles = os.listdir("./")
#         for myfile in allfiles:
#             if myfile.endswith(".inp"):
#                 print("GENERIC| You are running {} \n".format(myfile))
#                 file.write("""
# cd {}
# {} {} -i {} -o output.out
    
# """.format(os.path.join(self.cluster_dict["cluster_scratch"], execution_dir),
#                self.cluster_dict["mpirun"],  self.cluster_dict["exe"], myfile))
    
#         if batch_index > 0:
#             final_dir  = execution_dir.replace("BATCH_{}".format(batch_index), "BATCH_{}".format(batch_index + 1))
#             final_path = os.path.join(self.cluster_dict["cluster_scratch"], final_dir)
#             file.write("cp ./{}-1.restart {}\n\n".format(self.cp2k_dict["_SYSTEM_"], os.path.join(final_path, self.cp2k_dict["_RES_FILE_"])))
        
#             # Go in the next directory and run the new job
#             file.write("cd {}\n".format(final_path))
#             # Chagne the restart
#             file.write("chmod g+s ./*\n")
#             file.write("{} run.sh\n".format(self.cluster_dict['run_job']))
        
        
#         file.close()


    def create_run_file_paracelsus(self, batch_index, execution_dir):
        """
        CREATES THE RUN.SH FILE FOR PARACELSUS (ENS CLUSTER)
        =================================

        Parameters:
        -----------
            -batch_index: int used as label for the job name and for submitting the next batch calculation, -1 if it is the last batch to submit
            -execution_dir: the dir containing the run.sh file and all the inputs needed by cp2k
        """
    
        avail_partitions = ["qAVX0", "qAVX1", "qAVX2", "qEPYC"]
        # Chek if the partition is ok
        if not (self.cluster_dict["partition_name"] in avail_partitions):
            raise ValueError("Partition name not valid")

    
        # Check if the number of NODES are correct
        if self.cluster_dict["partition_name"] == avail_partitions[0]:
            if self.cluster_dict["ncpus"] % 12 != 0:
                exp_nodes = 1 + self.cluster_dict["ncpus"] // 12  
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 12
        elif self.cluster_dict["partition_name"] == avail_partitions[1]:
            if self.cluster_dict["ncpus"] % 16 != 0:
                exp_nodes = 1 + self.cluster_dict["ncpus"] // 16  
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 16
        elif self.cluster_dict["partition_name"] == avail_partitions[2]:
            if self.cluster_dict["ncpus"] % 20 != 0:
                exp_nodes = 1 + self.cluster_dict["ncpus"] // 20  
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 20
        else:
            if self.cluster_dict["ncpus"] % 64 != 0:
                exp_nodes =  1 + self.cluster_dict["ncpus"] // 64
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 64
            
        
        if exp_nodes != self.cluster_dict["nnodes"]:
            print("PARACELSUS| The expected number of nodes is {} but you choose {} for {} cpus".format(exp_nodes,self.cluster_dict["nnodes"],self.cluster_dict["ncpus"]))
            raise ValueError("PARACELSUS| The number of nodes is not correct, the job will crash")
    
        
        file = open("run.sh", "w")
        
        file.write(f"""#!/bin/bash
#SBATCH --job-name={self.cluster_dict["job_name"]}_{batch_index:d}
#SBATCH --export=ALL,MODULEPATH='',MODULEHOME=''  # Compulsory
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aurelien.zavadil@ens.psl.eu
#SBATCH --time={self.cluster_dict["time"]}
#SBATCH --nodes={self.cluster_dict["nnodes"]}
#SBATCH --ntasks={self.cluster_dict["ncpus"]}
#-SBATCH --ntasks-per-node=1    # of MPI tasks/node: @qAVX0 max 8|12, @qAVX1 12|16, @qAVX2 max 20, @EPYC 64
#SBATCH --cpus-per-task=1       # of OMP threads/task (default =  1)
#SBATCH --ntasks-per-core=1     # HT (default = 1, HyperThreads = 2)
#SBATCH --partition={self.cluster_dict["partition_name"]} # qAVX0 | qAVX1 | qAVX2 | qEPYC | qGPU

{self.cluster_dict["module_load"]}
        
""")

        allfiles = os.listdir("./")
        for myfile in allfiles:
            if myfile.endswith(".inp"):
                print("PARACELSUS| You are running {} \n".format(myfile))
                file.write("""
cd {}
{} {} -i {} -o output.out
    
""".format(os.path.join(self.cluster_dict["cluster_scratch"], execution_dir),
               self.cluster_dict["mpirun"],  self.cluster_dict["exe"], myfile))

        if batch_index > 0:
            final_dir  = execution_dir.replace("BATCH_{}".format(batch_index), "BATCH_{}".format(batch_index + 1))
            final_path = os.path.join(self.cluster_dict["cluster_scratch"], final_dir)
            #check on the convergence of the SCF caclculation with if
            # If converged, Go in the next directory and run the new job
            # Change the restart
            file.write(f"""
if grep -q "SCF run NOT" output.out || grep -q "ABORT" output.out; then
  echo 'WARNING : run not converged or job aborted
    Next job cancelled'
else
  cp ./{self.cp2k_dict["_SYSTEM_"]}-1.restart {os.path.join(final_path, self.cp2k_dict["_RES_FILE_"])}
  cp ./{self.cp2k_dict["_SYSTEM_"]}-RESTART.wfn {os.path.join(final_path, self.cp2k_dict["_RES_WFN_FILE_"])}
                       
  cd {final_path}
  chmod g+s ./*
  {self.cluster_dict['run_job']} run.sh
fi
""")        
        
        file.close()
    

    def create_run_file_irene(self, batch_index, execution_dir):
        """
        CREATES THE RUN.SH FILE FOR IRENE
        =================================

        Parameters:
        -----------
            -batch_index: int used as label for the job name and for submitting the next batch calculation, -1 if it is the last batch to submit
            -execution_dir: the dir containing the run.sh file and all the inputs needed by cp2k
        """
    
        avail_partitions = ["rome", "skylake"]
        # Chek if the partition is ok
        if not (self.cluster_dict["partition_name"] in avail_partitions):
            raise ValueError("Partition name not valid")
    
        myQOS = "normal"
        myQOS = self.cluster_dict["qos"]
        if self.cluster_dict["time"] > 60 * 60 * 24:
            myQOS = "long"
    
        # Check if the number of NODES are correct
        if self.cluster_dict["partition_name"] == avail_partitions[0]:
            if self.cluster_dict["ncpus"] % 128 != 0:
                exp_nodes = 1 + self.cluster_dict["ncpus"] // 128  
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 128
        else:
            if self.cluster_dict["ncpus"] % 48 != 0:
                exp_nodes =  1 + self.cluster_dict["ncpus"] // 48
            else:
                exp_nodes = self.cluster_dict["ncpus"] // 48
            
        
        if exp_nodes != self.cluster_dict["nnodes"]:
            print("IRENE| The expected number of nodes is {} but you choose {} for ncpus {}", exp_nodes,  self.cluster_dict["nnodes"],  self.cluster_dict["ncpus"])
            raise ValueError("IRENE| The number of nodes is not correct, the job will crash")
    
        
        file = open("run.sh", "w")
        
        file.write("""
#!/bin/bash
#MSUB -r {}_{:d}
#MSUB -q {}   #rome has 128 prc per node, skylake has 48
#MSUB -N {:d}
#MSUB -n {:d}
#MSUB -m scratch
#MSUB -x
#MSUB -E '--no-requeue'
#MSUB -Q {}    #normal or long
#MSUB -T {:d}   
#MSUB -A {}
set -x
        
{}
        
""".format(self.cluster_dict["job_name"], batch_index,
                   self.cluster_dict["partition_name"],
                   self.cluster_dict["nnodes"],
                   self.cluster_dict["ncpus"], 
                   myQOS,
                   self.cluster_dict["time"],
                   self.cluster_dict["account"],
                   self.cluster_dict["module_load"]))
        
        # TODO ADD the possibility of not using minus x and no requeue
        if not self.cluster_dict["minus_x"]:
            raise ValueError("IRENE| The minus x option should be true")

        if not self.cluster_dict["no_requeue"]:
            raise ValueError("IRENE| The no requeue option should be true")
        
        allfiles = os.listdir("./")
        for myfile in allfiles:
            if myfile.endswith(".inp"):
                print("IRENE| You are running {} \n".format(myfile))
                file.write("""
cd {}
{} {} -i {} -o output.out
    
""".format(os.path.join(self.cluster_dict["cluster_scratch"], execution_dir),
               self.cluster_dict["mpirun"],  self.cluster_dict["exe"], myfile))

        if batch_index > 0:
            final_dir  = execution_dir.replace("BATCH_{}".format(batch_index), "BATCH_{}".format(batch_index + 1))
            final_path = os.path.join(self.cluster_dict["cluster_scratch"], final_dir)
            #check on the convergence of the SCF caclculation with if
            # If converged, Go in the next directory and run the new job
            # Change the restart
            file.write(f"""
if grep -q "SCF run NOT" output.out || grep -q "ABORT" output.out; then
  echo 'WARNING : run not converged or job aborted
    Next job cancelled'
else
  cp ./{self.cp2k_dict["_SYSTEM_"]}-1.restart {os.path.join(final_path, self.cp2k_dict["_RES_FILE_"])}
  cp ./{self.cp2k_dict["_SYSTEM_"]}-RESTART.wfn {os.path.join(final_path, self.cp2k_dict["_RES_WFN_FILE_"])}
                       
  cd {final_path}
  chmod g+s ./*
  {self.cluster_dict['run_job']} run.sh
fi
""")        
        
        file.close()
    



def save_dict_to_json(json_file_name, my_dict):
    """
    SAVE A DICTIONARY TO JSON FILE
    ==============================
    """

    # Save dictionary to a JSON file
    with open(json_file_name, "w") as file:
         # 'indent=4' makes the JSON human-readable
        json.dump(my_dict, file, indent = 4) 
        

def load_dict_from_json(json_file_name):
    """
    LOAD A DICTIONARY FROM A JSON FILE
    ==================================
    """
    # Load dictionary from a JSON file
    with open(json_file_name, "r") as file:
        # Converts JSON into a Python dictionary
        dictionary = json.load(file)   
    
    return dictionary


        
