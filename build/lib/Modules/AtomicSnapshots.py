import numpy as np

import ase
from ase import Atoms
import ase.geometry
import ase.neighborlist
import ase.calculators.singlepoint
import ase.data

import psutil

import copy
import os, sys, gzip

import subprocess
import json
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.ticker import MaxNLocator

import AtomicSnap
from AtomicSnap import AtomicSnapshots
from AtomicSnap import Conductivity
from AtomicSnap import Vibrational


__MDANALYSIS__ = False
try:
    import MDAnalysis
    import MDAnalysis.analysis
    import MDAnalysis.analysis.rdf
    import MDAnalysis.analysis.msd
    __MDANALYSIS__ = True
except:
    __MDANALYSIS__ = False
    print("No MD Analysis found\n")



__JULIA__ = False
try:
    from julia import Julia
    # Avoid precompile issues
    jl = Julia(compiled_modules=False)  
    
    # Import a Julia module
    from julia import Main
    
    # Main.include("/home/antonio/IOcp2k/Modules/time_correlation.jl")
    __JULIA__ = True
except:
    __JULIA__ = False
    print("It seems that Julia is not available. Try to run with python-jl\n")

try:
    print("Test if Julia works...\n")
    julia.Main.eval('println("Hello from Julia!")')
    __JULIA__ = True
except:
    print("No Julia found!\n")
    __JULIA__ = False

# Conversions
BOHR_TO_ANGSTROM = 0.529177249 
ANGSTROM_TO_BOHR = 1/BOHR_TO_ANGSTROM
HA_TO_EV = 27.2114079527
HA_TO_KELVIN = 315775.326864009
HA_BOHR_TO_EV_ANGSTROM   = HA_TO_EV / BOHR_TO_ANGSTROM
HA_BOHR3_TO_EV_ANGSTROM3 = HA_TO_EV / BOHR_TO_ANGSTROM**3
BAR_TO_GPA = 0.0001
HABOHR3_TO_GPA = 29421.015697000003  #7355.256621097397
AU_TIME_TO_PICOSEC = 2.4188843265864003e-05

BAR_TO_HA_BOHR3 = BAR_TO_GPA * HABOHR3_TO_GPA**-1  

# Velocities
ANG_FEMPTOSEC_TO_HA = 0.04571028907825843

# Dipoles
DEBEYE_TO_eANG = 0.20819433622621247
DEBEYE_TO_AU = DEBEYE_TO_eANG * ANGSTROM_TO_BOHR

matplotlib.use('tkagg')

class AtomicSnapshots:
    """
    GET ATOMIC SNAPSHOTS OF CP2K 
    ============================


   
    """
    def __init__(self, **kwargs):
        """
        INITIALIZE THE ATOMIC SNAPSHOTS
        ================================

        Get atomic snapshots from cp2k to do some io
        
        Positions are in ANGSTROM

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR3 (the output of cp2k are BAR)
        
        Velocities are in BOHR/AU TIME

        Time step in FEMPTOSECOND

        It works only for CUBIC BOX!

        Parameters
        ----------
            -**kwargs : any other attribute of the ensemble

        """
        # Atomic position ANGSTROM
        # unwrapped positions
        self.positions = None
        # Atomic forces HARTREE/BOHR
        self.forces = None
        # Energies HARTREE
        self.energies = None
        # stress HARTREE/BOHR3
        self.stresses = None
        # Atomic types
        self.types = None
        # The number of atoms
        self.N_atoms = -1
        # The number of snapshots
        self.snapshots = 0
        # The cell NVT ANGSTROM
        self.unit_cell = np.eye(3) * 10
        # The unit cell NPT
        self.unit_cells = None
        # Potential and kientic energy HARTREE
        self.kinetic_energies = None
        self.potential_energies = None
        self.temperatures = None
        # Pressures and density
        # HARTREE/BOHR3
        self.pressures = None 
        # g/cm3
        self.densities = None 
        # HARTREE
        self.cons_quant = None
        # BOHR/AU_TIME
        self.velocities = None 
        # Dipole moments with berry phase in ATOMIC UNITS so BOHR
        self.dipoles = None
        self.dipoles_quantum = None
        self.dipoles_origin = None
        # the time step in FEMPTOSEC
        self.dt = -1
        # The type of the calculation
        self.calc_type = None

        # The ASE atom object
        self.ase_atoms_obj = None
        
        
        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        # This must be the last attribute to be setted
        self.fixed_attributes = True 

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])





    def init_from_lammps(self, file_dump_lammps, file_log_lammps, dt = 0.5, atomic_types = ["A", "B"],
                         read_forces = True,
                         create_ase_atoms = False, wrap_ase_pos = False,
                         subtract_com_ase = False, subtract_com_vel_ase = False, ase_pbc = [True, True, True],
                         max_snapshots = None,
                         type_def = np.float32,
                         calc_type = "NPT", verbose = True, debug = False):
        """
        INIT FROM CUSTOM LAMMPS FILE
        ============================

        Build to read the files from Lammps (using METAL units).

        The first file we read comes from
                
                thermo_style custom step time etotal ke pe temp press vol density cella cellb cellc cpuremain
                thermo_modify flush yes
                # Output the thermodynamic information every N steps
                thermo     NP

        and contains non-atomic resolved quantities.

        The second file we read comes from 

                dump      1 all custom NP output.dump id type xu yu zu vx vy vz fx fy fz


        NB make sure that the printing frquencies of the two files coincides with NP.

        NB male sure to print the unwrapped coordiantes.

        All the attributes of the class have atomic units exept for positions, unit_cells (ANGSTROM)

        Parameters:
        -----------
            -file_dump_lammps: str, the name of the file to read. Please use a gz file.
            -file_log_lammps: str, path to a LAMMPS log file
            -dt: float, the time step between the snapshots
            -atomic_types: list, the atomic species included in the simulation
            -read_forces: bool, set it to False to optimize the memory consumption
            -create_ase_atoms: float, if True we create also the ase atoms object
            -wrap_ase_pos:
            -ase_pbc: list of bool
            -max_snapshots: int, the maximum number of snapshots that we want to read.
            -calc_type: str, the type of clalculation NPT, NVT
            -verbose: bool
            -debug: bool
            
        """
        if not max_snapshots is None:
            # The maximum number of snapshots
            max_snapshots = int(max_snapshots)
        
        # Create a dictionary
        atomic_types = sorted(atomic_types)
        types_to_atoms = {}
        for i, item in enumerate(atomic_types):
            types_to_atoms.update({int(i+1) : item})

        if verbose:
            print('\n====> LAMMPS SNAPSHOTS <====')
            print("Reading custom LAMMPS trjectory from {} and {}".format(file_log_lammps, file_dump_lammps))
            print("The atoms are {}\n".format(types_to_atoms))
            
        # Initialize everything

        # The time step in FEMPTOSECONDS
        self.dt = np.copy(dt)
        # Positions ANGSTROM 
        self.positions = [] # np.zeros((self.snapshots, self.N_atoms, 3))
        # Atomic types
        self.types = []
        # Forces HARTREE/BOHR
        self.forces = [] # np.zeros((self.snapshots, self.N_atoms, 3))
        # Unit cells ANGSTROM  each row has a unit cell vector
        self.unit_cells = [] # np.zeros((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = [] # np.zeros((self.snapshots, self.N_atoms, 3))
       
        # Conserved quantity in HARTREE
        self.cons_quant = [] # np.zeros(self.snapshots)
        # Energies HARTREE
        self.energies = [] # np.zeros(self.snapshots)
        # Kinetic and potential energy in HARTREE
        self.kinetic_energies   = [] #np.zeros(self.snapshots)
        self.potential_energies = [] #np.zeros(self.snapshots)
        # Pressure in HA/BOHR3
        self.pressures = []
        # Density in g/cm3
        self.densities = []
        # Stresses HARTREE/BOHR3
        self.stresses = [] # np.zeros((self.snapshots, 3, 3))
        # Temperature in KELVIN
        self.temperatures = [] # np.zeros(self.snapshots)

        # Dipoles are in ATOMIC UNITS units
        self.dipoles    = np.zeros((self.snapshots, 3), dtype = type_def)
        self.dipoles_quantum    = np.zeros((self.snapshots, 3, 3), dtype = type_def)
        # Dipole orgin in ANGSTROM
        self.dipoles_origin = np.zeros((self.snapshots, 3), dtype = type_def)

        # The calculation executed
        self.calc_type = calc_type


        
        ###################   
        # READ LOG LAMMPS #
        ###################
        if verbose:
            print("=========== Read log LAMMPS file (energies, pressures, temperatures, denities) ===========")
            print("The file is {} is {}".format(file_log_lammps, os.getcwd()))
             
        if not file_log_lammps is None:
            file = open(file_log_lammps, 'r')
            lines = file.readlines()

            # Read the log file of lammps
            start_collecting = False
            
            for index, line in enumerate(lines):
                if line.startswith('   Step'):
                    start_collecting = True
                #print(line,start_collecting)
                if start_collecting:
                    try:
                        if type(float(line.split()[0])) is float:
                            # print(
                            # Get all the qunatities in METAL units Kelvin, eV, bar, g/cm3
                            self.kinetic_energies.append(type_def(line.split()[3]))
                            self.potential_energies.append(type_def(line.split()[4]))
                            self.temperatures.append(type_def(line.split()[5]))
                            self.pressures.append(type_def(line.split()[6]))
                            self.densities.append(type_def(line.split()[8]))
                    except:
                        # print(line)
                        pass

        self.snapshots = len(self.temperatures)
        # If the number of snapshots is greater than what expexcted 
        if not max_snapshots is None:
            if self.snapshots > max_snapshots:
                if verbose:
                    print("The snapshots found are {} = {:.1f} ps".format(self.snapshots, self.snapshots * dt * 1e-3))
                    print("reducing to {} snaphots = {:.1f} ps\n".format(max_snapshots, max_snapshots * dt * 1e-3))
                self.snapshots = max_snapshots
            else:
                print("The number of snapshots {} red is smaller than {}".format(self.snapshots, max_snapshots))
        else:
            if verbose:
                print("The snapshots found are {} = {:.1f} ps\n".format(self.snapshots, self.snapshots * dt * 1e-3))
        
        self.kinetic_energies   = np.asarray(self.kinetic_energies[:self.snapshots]) * HA_TO_EV**-1
        self.potential_energies = np.asarray(self.potential_energies[:self.snapshots]) * HA_TO_EV**-1
        self.energies           = self.kinetic_energies + self.potential_energies
        self.temperatures       = np.asarray(self.temperatures[:self.snapshots]) 
        self.pressures          = np.asarray(self.pressures[:self.snapshots]) * BAR_TO_HA_BOHR3
        self.densities          = np.asarray(self.densities[:self.snapshots])
        self.cons_quant = np.zeros(self.snapshots, dtype = type_def)

        if verbose:
            print("=========== End read log LAMMPS file... ===========\n")

        #######################   
        # END READ LOG LAMMPS #
        #######################

        
        
        ########################################   
        # READ POSITIONS  VELOCTIES AND FORCES #
        ########################################
        
        # All the ase atoms objects
        all_atoms_objects = []
    
        # Snapshot id       
        id_snap = 0
        # The numner of atoms
        Nat = 0
        # The cell and the corresponding shift ANGSTROM
        cell_d, shift = 0., 0.

        if verbose:
            print("=========== Reading custom LAMMPS file (positions, velocities, forces) in metal units... ===========")
            print("The file is {} in {}".format(file_dump_lammps, os.getcwd()))
            print("Are we reading the forces? {}".format(read_forces))
            print("ASE ATOMS | Are we creating the atoms? {} Wrapping positions? {} PBC {}".format(create_ase_atoms, wrap_ase_pos, ase_pbc))
            print("We will read {} snapshopts for a total time of {:.2f} ps".format(self.snapshots, self.snapshots * self.dt * 1e-3))
            print()
            
        # To spare some meory read line by line
        with gzip.open(file_dump_lammps, 'rt') as f:
        
            for index, line in enumerate(f):

                # Get the step id
                if line.startswith('ITEM: TIMESTEP'):
                    id_snap += 1

                # Get the number of atoms
                if line.startswith('ITEM: NUMBER OF ATOMS'):
                    Nat = int(next(f).split()[0])

                # Get the box bounds and shape
                if line.startswith('ITEM: BOX BOUNDS'):
                    # Get the cell in ANGSTROM
                    #    ITEM: BOX BOUNDS xx yy zz
                    #    xlo xhi
                    #    ylo yhi
                    #    zlo zhi
                    cell_d = np.zeros(3, dtype = type_def)
                    for i in range(3):
                        # Get the cell
                        _line_ = next(f)
                        shift     = type_def(_line_.split()[0]) 
                        cell_d[i] = type_def(_line_.split()[1]) - shift
                    
                if line.startswith('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz'):
                    # Get all the other atomic properties
                    arrays = np.zeros((Nat, 2 + 3 + 3 + 3), dtype = type_def)
                    # NB remember that the coordinates are unwrapped
                    for at in range(Nat):
                        # Each line has : id type xu yu zu vx vy vz fx fy fz
                        arrays[at,:] = np.asarray(next(f).split())

                    # Set the cell Angstrom
                    cell = np.eye(3, dtype = type_def) * cell_d
                    
                    # Print with the same order used in the ouptufile of LAMMPS
                    if debug and id_snap%10 == 0:
                        print("\nSTEP {} of LAMMPS".format(id_snap))
                        print("IDs")
                        print(arrays[0, 0])
                        print("POSITIONS [ANG]")
                        print(arrays[0, 2:5])
                        print("VELOCITIES [ANG/PS]")
                        print(arrays[0, 5:8])
                        print("FORCES [eV/ANG]")
                        print(arrays[0, 8:11])
                        print("CELL [ANG]")
                        print(cell)
                        print()

                    if verbose and id_snap%500 == 0:
                        # Get virtual memory details
                        mem = psutil.virtual_memory()
                        print("Processing LAMMPS snapshot {} with {} atoms.".format(id_snap, Nat))
                        print("RAM avail {:.2f} Gb".format(mem.available/1024**3))

                    
                    # Order according to the ids (they are not ordered in general)
                    arrays = arrays[arrays[:, 0].argsort()]
                    
                    # The ids of the atoms from LAMMPS 
                    ids = arrays[:,0]
                    # Get the atomic types as a list
                    types = [types_to_atoms[item] for item in arrays[:,1]]
                    # Get positions, velocities and forces
                    positions   = arrays[:, 2:5] - shift   # Angstrom
                    velocities  = arrays[:, 5:8]           # Angstrom/ps
                    if read_forces:
                        forces = arrays[:, 8:11]           # eV/Angstrom
                    else:
                        forces = None
    
                    # Put everything in the correct shape
                    # Positions ANGSTROM
                    self.positions.append(positions)
                    # Unit cells ANGSTROM
                    self.unit_cells.append(cell)

                  
                    
                    # BOHR/AU TIME
                    self.velocities.append(velocities * ANGSTROM_TO_BOHR /AU_TIME_TO_PICOSEC**-1)
                    # FORCES in HA/BOHR
                    if read_forces:
                        self.forces.append(forces * HA_TO_EV**-1 /ANGSTROM_TO_BOHR)
                    
                    # Check the type connsistency
                    if id_snap > 1:
                        if types != self.types:
                            raise ValueError("The types have changed {}".format(types))
                    self.types = types
    

                    if create_ase_atoms:
                        # Create the ase atoms uisng eV, Angstrom and ps
                        structure = ase.atoms.Atoms(types, positions, cell = cell, pbc = ase_pbc)
                        structure.set_cell(cell)
                        # ANGSTROM/PICOSEC
                        structure.set_velocities(velocities)
                        
                        if subtract_com_ase:
                            masses = [ase.data.atomic_masses[ase.data.atomic_numbers[s]] for s in types]
                            masses = np.asarray(masses)
                            structure.positions -= np.einsum('a, ab -> b', masses, structure.positions) /np.sum(masses)

                        if wrap_ase_pos:
                            structure.wrap()

                        # if subtract_com_vel_ase:
                        #     structure.velocities -= np.einsum('a, ab -> b', masses, structure.get_velocities()) /np.sum(masses)
                        # Now set the calculator so that we have energies and forces
                        calculator = ase.calculators.singlepoint.SinglePointCalculator(structure,
                                                                                       # watch out for the -1 convention EV
                                                                                       energy = self.energies[id_snap-1] * HA_TO_EV, 
                                                                                       # EV/ANGSTROM
                                                                                       forces = forces,
                                                                                       stress = np.zeros(6))
                        # Attach the calculator
                        structure.calc = calculator
                        
                        # Store all the ase atoms objects
                        all_atoms_objects.append(structure)

                # The id_snap starts from zero
                if id_snap > self.snapshots:
                    print("Snapshots red {}. Max number of snaphsots {} reached".format(id_snap, self.snapshots))
                    break

        if verbose:            
            print("=========== End reading custom LAMMPS file... ===========\n")

        
        # if len(self.kinetic_energies) != len(self.forces[:,0,0]):
        #     raise ValueError("The length does not coincide")
            
        # Store the ase atoms object
        if create_ase_atoms:
            if verbose:
                print("Creating the ase atoms object from LAMMPS data in metal units")
            self.ase_atoms_obj = all_atoms_objects
        else:
            self.ase_atoms_obj = None
            
        # Get the number of snapshots
        self.snapshots = len(self.energies)
        # Get the number of atoms
        self.N_atoms = Nat
        
        # Positions ANGSTROM 
        self.positions  = np.asarray(self.positions).reshape((self.snapshots, self.N_atoms, 3))
        if read_forces:
            # Forces HARTREE/BOHR
            self.forces     = np.asarray(self.forces).reshape((self.snapshots, self.N_atoms, 3)) 
        else:
            self.forces     = None
        # Unit cells ANGSTROM  each row has a unit cell vector
        self.unit_cells = np.asarray(self.unit_cells).reshape((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = np.asarray(self.velocities).reshape((self.snapshots, self.N_atoms, 3)) 
        # Init the stresses HA/BOHR3
        self.stresses   = np.zeros((self.snapshots, 3, 3))


        # Check consistency
        if self.kinetic_energies.shape != self.positions[:,0,0].shape:
            raise ValueError("The shape of the datas does not match")
            

        if verbose:
            print("Finalizing...")
            print("The snapshots are {}".format(self.snapshots))
            print("Total time is {} ps".format(self.snapshots * dt * 1e-3))
            print("The number of atoms is {}".format(Nat))
            
        return 
        

    def init(self, file_name_head, unit_cell, debug = False, verbose = True, calc_type = 'GEO_OPT',
                   ext_pos = None, ext_force = None,
                   ext_stress = None, ext_vel = None,
                   ext_ener = None, ext_cell = None, ext_dipoles = None):
        """
        READ FILES AS CREATED BY CP2K
        =============================

        Positions-Unit cells are in ANGSTROM

        Energies are in HARTREE

        Forces are in HARTREE/BOHR
        
        Stresses are in HARTREE/BOHR^3
        
        Velocities are in BOHR/AU-TIME
        
        Paramters:
        ----------
            -file_name_head: the path to the extension of the file
            -unit_cell: np.array with the cell shape in ANGSTROM, lattice vectors are the rows.
            -debug: bool,
            -verbose: bool,
            -calc_type: strg, it is needed to understand which calculation was done
            -ext_pos: strg, the extension of the position file
            -ext_force: strg, the extension of the force file
            -ext_stress: strg, the extension of the stress file
            -ext_velocities: strg, the extension of the velocities file
            -ext_ener: strg, the extension of the energ file
            -ext_cell: strg, the extension of the cell file
            -ext_dipoles: strg, the extension of the dipoles file
        """ 
        # Set up the unit cell in Angstrom
        self.unit_cell = np.copy(unit_cell)
        
        avail_calc_type = ['GEO_OPT', 'MD']
        if not(calc_type in avail_calc_type):
            raise ValueError('Choose calc_type among {}'.format(avail_calc_type))
        self.calc_type = calc_type

        ####### READ THE POSITIONS FILE ########
        if ext_pos is None:
            raise ValueError('Initalize the class with file.ext_pos')
        # Open the target file
        # Note that at least the postion file should exists otherwise we can not initialize the class
        file_xyz = open(file_name_head + ext_pos, 'r')
        lines = file_xyz.readlines()

        # Get the number of atoms
        self.N_atoms = int(lines[0])
        # Get the number of MD, GEO_OPT steps
        for index, line in enumerate(lines):
            if line.startswith(' i ='):
                self.snapshots += 1
            if calc_type == 'MD':
                if line.startswith(' i =        1'):
                    self.dt = float(lines[index].split()[5][:-1])
                
        if verbose:
            print('\n====> CP2K SNAPSHOTS <====')
            print('CP2k reading from files beginning with {}'.format(file_name_head))               
            print('N_atoms   = {}'.format(self.N_atoms)) 
            print('Snapshots = {}'.format(self.snapshots))
        
        # Initialize everything
        # Positions ANGSTROM 
        self.positions = np.zeros((self.snapshots, self.N_atoms, 3))
        # Atomic types
        self.types = [''] * self.N_atoms
        # Forces HARTREE/BOHR
        self.forces = np.zeros((self.snapshots, self.N_atoms, 3))
        # Unit cells ANGSTROM  each row has a unit cell vector
        self.unit_cells = np.zeros((self.snapshots, 3, 3))
        # Conserved quantity in HARTREE
        self.cons_quant = np.zeros(self.snapshots)
        # Energies HARTREE
        self.energies = np.zeros(self.snapshots)
        # Stresses HARTREE/BOHR3
        self.stresses = np.zeros((self.snapshots, 3, 3))
        # Velocities BOHR/AU TIME
        self.velocities = np.zeros((self.snapshots, self.N_atoms, 3))
        # Pressures and densities are NONE
        self.pressures = None #np.zeros(self.snapshots)
        self.densities = None #np.zeros(self.snapshots)
        
        # Kinetic and potential energy in HARTREE
        self.kinetic_energies   = np.zeros(self.snapshots)
        self.potential_energies = np.zeros(self.snapshots)
        # Temperature in KELVIN
        self.temperatures = np.zeros(self.snapshots)

        # Dipoles are in ATOMIC UNITS units
        self.dipoles    = np.zeros((self.snapshots, 3))
        self.dipoles_quantum    = np.zeros((self.snapshots, 3, 3))
        # Dipole orgin in ANGSTROM
        self.dipoles_origin = np.zeros((self.snapshots, 3))
        
        
        # Read the atomic positions of each snapshots
        for isnap in range(self.snapshots):
            # The index where the atomic positions start
            file_index = isnap * (self.N_atoms + 2) + 2
            
            # This is the potential energy
            self.energies[isnap] = float(lines[file_index - 1].split()[-1])
            for iatom in range(self.N_atoms):
                coords = np.asarray(lines[file_index + iatom].split()[1:])
                self.positions[isnap, iatom, :] = coords
                # Get the types
                self.types[iatom] = lines[file_index + iatom].split()[0].capitalize()
                
            if debug:
                print('COORDS SNAP={}'.format(isnap))
                print(self.positions[isnap,:,:])
                print(self.types)
                print()

        # close the file
        file_xyz.close()
        
        if verbose:
            if debug:
                print('Types of atoms {}'.format(self.types))
            print('Unique Types of atoms {}'.format(set(self.types)))
        ####### END READ THE POSITIONS FILE ########
        

        ####### READ THE FORCES FILE ########
        if not(ext_force is None):
            if not os.path.exists(file_name_head + ext_force):
                raise ValueError('I do not find the force file {}'.format(file_name_head + ext_force))
            # Open the target file
            file_force = open(file_name_head + ext_force, 'r')
            lines = file_force.readlines()

            # Read the atomic positions of each snapshots
            for isnap in range(self.snapshots):
                # The index where the forces start
                file_index = isnap * (self.N_atoms + 2) + 2

                # Read the force for the atoms
                for iatom in range(self.N_atoms):
                    atom_force = np.asarray(lines[file_index + iatom].split()[1:])
                    self.forces[isnap, iatom, :] = atom_force

                if debug:
                    print('FORCES SNAP={}'.format(isnap))
                    print(self.forces[isnap,:,:])
                    print()
            # Close the file
            file_force.close()
        
        ####### END READ THE FORCES FILE ########
        
        
        
        
        ####### READ THE STRESS FILE ########
        units_bar = False
        if not(ext_stress is None):
            if not os.path.exists(file_name_head + ext_stress):
                raise ValueError('I do not find the stress file {}'.format(file_name_head + ext_stress))
            file_stress = open(file_name_head + ext_stress, 'r')
            lines = file_stress.readlines()
            
            if '[bar]' in lines[0].split():
                units_bar = True
            else:
                raise ValueError('I do not know the units for pressure')
                
            # Read the atomic positions of each snapshots
            for isnap in range(self.snapshots):
                #print(np.asarray(lines[isnap + 1].split()[2:], dtype = float))
                #xx = float(lines[isnap + 1].split()[2])
                #xy = float(lines[isnap + 1].split()[3])
                #xz = float(lines[isnap + 1].split()[4])
                #yx = float(lines[isnap + 1].split()[5])
                #yy = float(lines[isnap + 1].split()[6])
                #yz = float(lines[isnap + 1].split()[7])
                #zx = float(lines[isnap + 1].split()[8])
                #zy = float(lines[isnap + 1].split()[9])
                #zz = float(lines[isnap + 1].split()[10])
                #self.stresses[isnap,:,:] = np.asarray([[xx, xy, xz],
                #                                       [yx, yy, yz],
                #                                       [zx, zy, zz]])
                self.stresses[isnap,:,:] = np.asarray(lines[isnap + 1].split()[2:], dtype = float).reshape((3,3))
                #print(self.stresses[isnap,:,:] - np.asarray(lines[isnap + 1].split()[2:], dtype = float).reshape((3,3)))
                
            if units_bar:
                print('Stress tensor in HA BOHR3')
                # print(BAR_TO_HA_BOHR3)
                self.stresses[:, :, :] *= BAR_TO_HA_BOHR3
                
            # Close the file
            file_stress.close()
        ####### END READ THE STRESS FILE ########
        
        
        
        
        ####### READ THE VELOCITY FILE ########
        if not(ext_vel is None):
            if not os.path.exists(file_name_head + ext_vel):
                raise ValueError('I do not find the velocity file {}'.format(file_name_head + ext_vel))
            # Open the target file
            file_vel = open(file_name_head + ext_vel, 'r')
            lines = file_vel.readlines()

            # Read the atomic velocties of each snapshots
            for isnap in range(self.snapshots):
                file_index = isnap * (self.N_atoms + 2) + 2

                for iatom in range(self.N_atoms):
                    coords = np.asarray(lines[file_index + iatom].split()[1:])
                    self.velocities[isnap, iatom, :] = coords
                if debug:
                    print('VEL SNAP={}'.format(isnap))
                    print(self.velocities[isnap,:,:])
                    print()

            # close the file
            file_vel.close()
        ####### END READ THE VELOCITY FILE ########
        
        
        
        ####### READ THE ENERG FILE ########
        if not(ext_ener is None):
            
            if not os.path.exists(file_name_head + ext_ener):
                raise ValueError('I do not find the energ file {}'.format(file_name_head + ext_ener))
            # Open the target file
            file_ener = open(file_name_head + ext_ener, 'r')
            lines = file_ener.readlines()

            for isnap in range(self.snapshots):
                self.kinetic_energies[isnap]   = float(lines[isnap + 1].split()[2])
                self.temperatures[isnap]       = float(lines[isnap + 1].split()[3])
                self.potential_energies[isnap] = float(lines[isnap + 1].split()[4])
                self.cons_quant[isnap]         = float(lines[isnap + 1].split()[5])
            if self.calc_type == 'MD':
                try:
                    self.dt = float(lines[2].split()[1]) - float(lines[1].split()[1])
                except:
                    print("Set anually the time step. The file is too short. Probably only one step was done.")
                    self.dt = float(input("Please enter the dt: "))
                print('Update the dt to {} fs'.format(self.dt))
                self.dt = float(self.dt)
            # close the file
            file_ener.close()
        ####### END READ THE ENER FILE ########


        ####### READ THE DIPOLE FILE ########
        if not(ext_dipoles is None):
            
            if not os.path.exists(file_name_head + ext_dipoles):
                raise ValueError('I do not find the dipoles file {}'.format(file_name_head + ext_dipoles))
            # Open the target file
            file_dipoles = open(file_name_head + ext_dipoles, 'r')
            lines = file_dipoles.readlines()

            for isnap in range(self.snapshots):
                index = (isnap + 0) * 10 + 9
                # print(isnap, lines[index].split())
                # DEBYE TO ATOMIC UNITS
                self.dipoles[isnap, :] = np.array([float(lines[index].split()[1]),
                                                   float(lines[index].split()[3]),
                                                   float(lines[index].split()[5])]) * DEBEYE_TO_AU
                # IN ANGSTROM as all the positions
                self.dipoles_origin[isnap, :] = np.array([float(lines[index - 9].split()[-3]),
                                                          float(lines[index - 9].split()[-2]),
                                                          float(lines[index - 9].split()[-1])]) * BOHR_TO_ANGSTROM

                # DEBYE TO ATOMIC UNITS x, y, z along eahc row
                self.dipoles_quantum[isnap, :, :] = np.array([[float(lines[index - 4].split()[2]), float(lines[index - 4].split()[3]), float(lines[index - 4].split()[4])],
                                                              [float(lines[index - 3].split()[1]), float(lines[index - 3].split()[2]), float(lines[index - 3].split()[3])],
                                                              [float(lines[index - 2].split()[2]), float(lines[index - 2].split()[3]), float(lines[index - 2].split()[4])]]) * DEBEYE_TO_AU
                                                
                
            # close the file
            file_dipoles.close()

            # self.refold_all_dipoles(Nmax = 10, tol = 1.0, debug = False, debug_visualize = False)

                    
        ####### END READ THE ENER FILE ########



        ####### READ THE CELL FILE ########
        if not(ext_cell is None):
            
            if not os.path.exists(file_name_head + ext_cell):
                raise ValueError('I do not find the cell file {}'.format(file_name_head + ext_cell))
            # Open the target file
            file_cells = open(file_name_head + ext_cell, 'r')
            lines = file_cells.readlines()

            for isnap in range(self.snapshots):
                for i in range(3):
                    self.unit_cells[isnap, 0, i]   = float(lines[isnap + 1].split()[2 + i])

                for i in range(3):
                    self.unit_cells[isnap, 1, i]   = float(lines[isnap + 1].split()[5 + i])

                for i in range(3):
                    self.unit_cells[isnap, 2, i]   = float(lines[isnap + 1].split()[8 + i])
                

            # close the file
            file_cells.close()
        ####### END READ THE CELL FILE ########
        else:
            print("No unit cells files was used\nwe are copying")
            print(self.unit_cell)
            print('\nin self.unit_cells')
            self.unit_cells[:,:,:] = np.copy(self.unit_cell)

        return 





    def create_ase_snapshots(self, wrap_positions, subtract_com, pbc = True, write_forces = True, write_velocities = True):
        """
        CREATE ASE SNAPSHOTS 
        ====================
        
        Rember that ase use EV, ANGSTROM, EV/ANGSTROM3 ANGSTROM/PICOSECOND

        It correctly prints energy forces stresses and position

        In cp2k the coordinates are saved not wrapped so we wrap them back in the cell

        In addition, the center of mass can drift. So, after wrapping  the positions
        
        Paramters:
        ----------
            -wrap_positions: bool, use False to compute Mean Square Displacement
            -subtract_com: bool, if True we subract the position of the center of mass (useful for MSD analysis)
            -pbc: bool, the PBC conditions along x y z (def True)
        Returns:
        --------
            -all_atoms: a list of ase objects
        """
        # Set up all the ase atoms
        all_atoms = []

        print("\nCREATING ASE ATOMS SNAPSHOTS...")
        print("1) Subtracting the center of mass positions? {}".format(subtract_com))
        print("2) Wrapping the positions? {}".format(wrap_positions))
        print("3) Setting PBC? {}\n".format(pbc))

        if subtract_com:
            # Get the center of mass positions (self.snaphsots, 3)
            R_com = self.get_com_positions()
            
        # Range in the snapshots ans use ASE units
        # ev and angstrom, ev/angstrom3, angstrom/picosecond
        for isnap in range(self.snapshots):

            energy     = self.potential_energies[isnap] * HA_TO_EV
            if energy == 0. or energy is None:
                raise ValueError("The energies are zero or not setted")

            if not self.forces is None:
                forces     = self.forces[isnap,:,:] * HA_BOHR_TO_EV_ANGSTROM
                if np.all(forces == np.zeros((self.N_atoms,3))):
                    raise ValueError("The forces are zero")
            else:
                forces = None
            
            stress     = -transform_voigt(self.stresses[isnap,:,:]) * HA_BOHR3_TO_EV_ANGSTROM3
            velocities = self.velocities[isnap,:,:] * BOHR_TO_ANGSTROM /AU_TIME_TO_PICOSEC

            if subtract_com:
                positions = np.copy(self.positions[isnap,:,:] - R_com[isnap,:])
            else:
                positions = np.copy(self.positions[isnap,:,:])

            if wrap_positions:
                positions = ase.geometry.wrap_positions(positions[:,:], self.unit_cells[isnap,:,:], pbc = pbc)
            
            
            # Create the ase atoms object
            if np.linalg.det(self.unit_cells[isnap,:,:]) <= 1e-6: 
                raise ValueError("No cell found")
                
            structure = Atoms(self.types, positions,
                              cell = self.unit_cells[isnap,:,:],
                              pbc = [True, True, True])
            structure.set_cell(self.unit_cells[isnap,:,:])
            structure.pbc[:] = pbc
            if write_velocities:
                # Set the velocties
                structure.set_velocities(velocities)

            if write_forces:
                # Now set the calculato so that we have energies and forces
                calculator = ase.calculators.singlepoint.SinglePointCalculator(structure, energy = energy, forces = forces, stress = stress)
            else:
                calculator = ase.calculators.singlepoint.SinglePointCalculator(structure, energy = energy)
                
            # Attach the calculator
            structure.calc = calculator

            # Append to the list
            all_atoms.append(structure)
            
        return all_atoms

    def copy_snapshots(self):
        """
        RETURN A COPY OF THE CURRENT CLASS
        ===================================
        """
        snapcopy = AtomicSnapshots()

        # Iterate over instance attributes
        for attr in self.__dict__:  
            # Get the attribute's value
            value = getattr(self, attr) 
            
            # Use np.copy() for numpy arrays, deep copy for others
            if isinstance(value, np.ndarray):
                setattr(snapcopy, attr, np.copy(value))
            else:
                # General deep copy
                setattr(snapcopy, attr, copy.deepcopy(value))  
            
        return snapcopy  

    def merge_multiple_snapshots(self, list_of_snapshots):
        """
        MERGE MULTIPLE SNAPSHOTS
        ========================

        Parameters:
        -----------
            -atomic_snapshot: a list of AtomicSnapshots() objects to merge with them selves

        Retrurns:
        ---------
            -snap_merge: the final AtomicSnapshots() object
        """
        snap_merge = AtomicSnapshots()
        # snap_merge

        print("\n=========== MERGING ===========")
        print("We are merging {} objects".format(len(list_of_snapshots)))
        
        for i, snap in enumerate(list_of_snapshots):
            if i == 0:
                snap_merge = snap.copy_snapshots()
            else:
                snap_merge = snap_merge.merge_snapshots(snap)

        print("The merge is finished")
        print("The total time we have is {:.1f} fs {:.2f} ps".format(snap_merge.dt * snap_merge.snapshots, snap_merge.dt * snap_merge.snapshots * 1e-3))
        print("=========== END MERGING ===========")
        print()

        return snap_merge
        
    def merge_snapshots(self, atomic_snapshots):
        """
        MERGE TWO ATOMIC SNAPSHOTS CLASS
        ================================

        We check that the two are compatible

        Parameters:
        -----------
            -atomic_snapshot: the AtomicSnapshots() object to merge with self
        
        Returns:
        --------
            -snap_merge: the merged AtomicSnapshots() object
        """
        
        if self.N_atoms != atomic_snapshots.N_atoms:
            raise ValueError("The number of atoms should be the same")

        if self.types != atomic_snapshots.types:
            print(self.types, atomic_snapshots.types)
            raise ValueError("The atomic types are different")

        if self.dt != atomic_snapshots.dt:
            raise ValueError("The time step dt should be the same")

        if np.max(np.abs(self.unit_cell - atomic_snapshots.unit_cell)) > 1e-10:
            raise ValueError("The unit cell should be the same")

        if self.calc_type != atomic_snapshots.calc_type:
            raise ValueError("The type of calculation should be the same")

        snap_merge = AtomicSnapshots()

        # Iterate over instance attributes
        for attr in self.__dict__:  
            if not(attr in ['unit_cell', 'types', 'calc_type', 'snapshots', 'dt']):
                # Get the attribute's value
                value1 =             getattr(self, attr) 
                value2 = getattr(atomic_snapshots, attr)
                
                # Use np.copy() for np arrays, copy.deepcopy for others
                if isinstance(value1, np.ndarray):
                    setattr(snap_merge, attr, np.concatenate((value1, value2), axis=0))
                # Use deepcopy for lists, string, dictionary etc
                else:
                    setattr(snap_merge, attr, copy.deepcopy(value1))  

        snap_merge.snapshots = self.snapshots + atomic_snapshots.snapshots
        
        snap_merge.unit_cell = np.copy(self.unit_cell)

        snap_merge.calc_type = copy.deepcopy(self.calc_type)

        snap_merge.types = copy.deepcopy(self.types)

        snap_merge.dt = copy.deepcopy(self.dt)
        
        return snap_merge  
    
    def plot_energy_force(self, img_name = None):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS 
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::  \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -uc: a np.array with the cell BOHR
            -img_name: imag name if you want to save
        """
        # matplotlib.use('tkagg')

        energies = self.energies * HA_TO_EV /self.N_atoms
        forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms
        
        x = np.arange(self.snapshots, dtype = int)
        
        # Width and height
        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(2, 1, figure=fig)
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, energies, 's', color = 'k')
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, forces, 'd', color = 'red')
        ax.set_xlabel('Steps', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()
        
        return


    def plot_temperature_evolution(self, average_window = [0,-1], img_name = None, show = True):
        """
        PLOT TEMPERATURE EVOLUTION FROM AND MD
        ======================================

        The average temperature is 

        .. math:: T_{av} = \frac{1}{N} \sum_{n=1}^{N} T_{n}

        and for the error we use the standard error of scipy.stats.sem
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.temperatures[average_window[0]:average_window[1]]
            -img_name: imag name if you want to save
        """
        # matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        # In femptosceond
        x = np.arange(self.snapshots) * self.dt

        # Get the average and the standard error
        T_av = np.average(self.temperatures[average_window[0]: average_window[1]])
        T_err = scipy.stats.sem(self.temperatures[average_window[0]: average_window[1]])
        
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3, label = 'T = {:.1f} {:.3f} K from {:.2f} to {:.2f} ps'.format(T_av, T_err,
                                                                                                                              xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x, self.temperatures, np.min(self.temperatures), where=(x >= xmin) & (x <= xmax), color='purple', alpha=0.3)
        ax.axhline(y = T_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)

        if show:
            plt.show()


    def plot_pressure_evolution(self, average_window = [0,-1], img_name = None, show = True):
        """
        PLOT PRESSURE EVOLUTION FROM AND MD
        ======================================

        The average temperature is 

        .. math:: P_{av} = \frac{1}{N} \sum_{n=1}^{N} P_{n}
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.temperatures[average_window[0]:average_window[1]]
            -img_name: imag name if you want to save
        """
        # matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        x = np.arange(self.snapshots) * self.dt

        # The pressure is the trace of the stress tensor
        if self.pressures is None:
            pressures = np.einsum("iaa -> i", self.stresses) * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA /3 
        else:
            pressures = self.pressures * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA

        # Get the average with the standard error GPa
        P_av  = np.average(pressures[average_window[0]: average_window[1]])
        P_err = scipy.stats.sem(pressures[average_window[0]: average_window[1]])
        
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, pressures,  color = 'green', lw = 3,
                label = 'P={:.3f} {:.3f} GPa from {:.2f} to {:.2f} ps'.format(P_av, P_err, xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x, pressures, np.min(pressures), where=(x >= xmin) & (x <= xmax), color='green', alpha=0.3)
        ax.axhline(y = P_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Pressure [GPa]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        if show:
            plt.show()


    def plot_unit_cell_evolution(self, average_window = [0,-1], img_name = None, show = True):
        """
        PLOT ISOTROPIC UNIT CELL and VOLUME EVOLUTION FROM MD
        =====================================================

        The unit cell is plotted in ANGSTROM and the volume in ANGSTROM^3
        
        Paramters:
        ----------
            -average_window: a list of two numbers, so we will average self.unit_cell[average_window[0]:average_window[1],:,:]
            -img_name: imag name if you want to save
        """
        # matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1,2, figure=fig)

        x = np.arange(self.snapshots) * self.dt
        # Get the number of samples
        N_samples = len(self.unit_cells[average_window[0]: average_window[1], 0, 0])
        
        # Get the average and the standard error in ANGSTROM
        UC_av = np.einsum('iab -> ab', self.unit_cells[average_window[0]: average_window[1],:,:]) /N_samples
        UC_err = np.sqrt(np.einsum('iab -> ab', (self.unit_cells[average_window[0]: average_window[1],:,:] - UC_av[:,:])**2))/N_samples

        # Get the volumes in ANGSTROM^3
        volumes = np.zeros(self.snapshots, dtype = float)
        for i in range(self.snapshots):
            volumes[i] = np.linalg.det(self.unit_cells[i,:,:]) 
            
        # Get the average and the standard error in ANGSTROM^3
        V_average = np.average(volumes[average_window[0]: average_window[1]])
        V_err = scipy.stats.sem(volumes[average_window[0]: average_window[1]])

        for i in range(2):
                ax = fig.add_subplot(gs[0,i])
                xmin, xmax = np.sort(x[average_window])
                if i == 0:
                    # Plot only the x component
                    ax.plot(x, self.unit_cells[:,i,i],  color = 'green', lw = 3,
                            label = 'L={:.2f} {:.2f} Ang\nfrom {:.2f} to {:.2f} ps'.format(UC_av[i,i], UC_err[i,i], xmin * 1e-3, xmax * 1e-3))
                    ax.fill_between(x,  self.unit_cells[:,i,i], np.min(self.unit_cells[:,i,i]),
                                    where=(x >= xmin) & (x <= xmax), color='green', alpha=0.3)
                    ax.axhline(y = UC_av[i,i], color = 'k', linestyle=':')
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(axis = 'both', labelsize = 12)
                    ax.set_ylabel('Cell [Angstrom]', size = 12)
                else:
                    ax.plot(x, volumes,  color = 'blue', lw = 3,
                            label = 'V={:.2f} {:.2f} Ang3\nfrom {:.2f} to {:.2f} ps'.format(V_average, V_err, xmin * 1e-3, xmax * 1e-3))
                    ax.fill_between(x,  np.asarray(volumes), np.min(volumes),
                                    where=(x >= xmin) & (x <= xmax), color='blue', alpha=0.3)
                    ax.axhline(y = V_average, color = 'k', linestyle=':')
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.tick_params(axis = 'both', labelsize = 12)
                    ax.set_ylabel('Volume [Angstrom$^{3}$]', size = 12)
                ax.set_xlabel('Time [fs]', size = 15)
                plt.legend(fontsize = 12)
                plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)

        if show:
            plt.show()


    def plot_density_evolution(self, average_window = [0,-1], img_name = None, save_rho_json = False,
                               json_file_result = "time_rho.json",
                               return_values = False, show = True):
        """
        PLOT DENISTY EVOLUTION FROM MD
        ==============================

        The density in g/cm3. Use ANGSTROM for length and g/mol for the masses
        
        Paramters:
        ----------
            -average_window: a list of two int numbers, so we will average self.unit_cell[average_window[0]:average_window[1],:,:]
            -img_name: image name if you want to save
        """
        # matplotlib.use('tkagg')
        # Width and height
        fig = plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1,1, figure=fig)

        # Time steps in FEMPTOSECOND
        x = np.arange(self.snapshots) * self.dt

        if self.densities is None:
            # Get the total molar mass in UMA 
            total_mass = np.sum(self.get_masses_from_types())
    
            # Get the volumes in ANGSTROM^3
            volumes = np.zeros(self.snapshots, dtype = float)
            for i in range(self.snapshots):
                volumes[i] = np.linalg.det(self.unit_cells[i,:,:]) 
            
            # Get the density from UMA/ANG3 in g/cm^3
            # rho = total_mass  /(0.602214076 * volumes)
            rho = total_mass * 1.660538 /volumes
        else:
            rho = self.densities

        if save_rho_json:
            # Save the results to a json file
            times_rho = {"t" : list(x), "rho" : list(rho)}
            save_dict_to_json(json_file_result, times_rho)
        
        rho_av  = np.average(rho[average_window[0]: average_window[1]]) 
        rho_err = np.std(rho[average_window[0]: average_window[1]])

        # Plot everything
        ax = fig.add_subplot(gs[0,0])
        xmin, xmax = np.sort(x[average_window])
        ax.plot(x, rho,  color = 'purple', lw = 3,
                label = '$\\rho$' + '={:.4f} {:.4f} g/cm3\nfrom {:.2f} to {:.2f} ps'.format(rho_av, rho_err, xmin * 1e-3, xmax * 1e-3))
        ax.fill_between(x,  rho, np.min(rho), where=(x >= xmin) & (x <= xmax), color='purple', alpha=0.3)
        ax.axhline(y = rho_av, color = 'k', linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis = 'both', labelsize = 12)
        ax.set_ylabel('Density [g/cm$^{3}$]', size = 12)
        ax.set_xlabel('Time [fs]', size = 15)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        if show:
            plt.show()

        if return_values:
            return rho_av, rho_err
        

    def plot_md_nvt(self, img_name = None, show = True):
        """
        PLOT ENERGY FORCE KINETIC ENERGY TEMERATURE FROM ASE SNAPSHOTS FOR NVT SIMULATION
        ==================================================================================
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::   \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -img_name: imag name if you want to save
        """
        matplotlib.use('tkagg')
        
        energies = self.energies * HA_TO_EV /self.N_atoms

        if not self.forces is None:
            forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms
        else:
            print("plot_md_nvt | No forces were setted.")
            forces = np.zeros(self.snapshots)
        
        x = np.arange(self.snapshots, dtype = int) * self.dt
        
        # Width and height
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(3,2, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(x, energies,  color = 'k', lw = 3)
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        


        
        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, self.kinetic_energies * HA_TO_EV/self.N_atoms,  color = 'darkorange', lw = 3)
        ax.set_ylabel('Kin energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax = fig.add_subplot(gs[1,1])
        ax.plot(x, forces,  color = 'red', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        

        ax = fig.add_subplot(gs[2,0])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax = fig.add_subplot(gs[2,1])
        ax.plot(x, self.cons_quant * HA_TO_EV * 1000/self.N_atoms,  color = 'darkblue', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Cons qunt [meV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        if show:
            plt.show()
        
        return

    def plot_md_npt(self, img_name = None, show = True):
        """
        PLOT ENERGY FORCE FROM ASE SNAPSHOTS FOR NVT SIMULATION
        
        Rember that ase use EV, ANGSTROM
        
        To plot the force at each time step 
        
        .. math::   \sum_{I=1}^{N_at} F_I \cdot F_I /N_at
        
         Paramters:
        ----------
            -img_name: imag name if you want to save
        """
        # matplotlib.use('tkagg')

        energies = self.energies * HA_TO_EV /self.N_atoms
        if not self.forces is None:
            forces = np.einsum('iab, iab -> i', self.forces * HA_BOHR_TO_EV_ANGSTROM, self.forces * HA_BOHR_TO_EV_ANGSTROM) /self.N_atoms
        else:
            print("plot_md_npt | No forces were setted.")
            forces = np.zeros(self.snapshots)

        
        x = np.arange(self.snapshots, dtype = int) * self.dt
        
        # Width and height
        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(4,2, figure=fig)
        
        ax = fig.add_subplot(gs[0,:])
        ax.plot(x, energies,  color = 'k', lw = 3)
        ax.set_ylabel('Energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        

        ax = fig.add_subplot(gs[1,0])
        ax.plot(x, self.kinetic_energies * HA_TO_EV/self.N_atoms,  color = 'darkorange', lw = 3)
        ax.set_ylabel('Kin energy [eV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[1,1])
        ax.plot(x, forces,  color = 'red', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Force [eV/Ang/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        
        ax = fig.add_subplot(gs[2,0])
        ax.plot(x, self.temperatures,  color = 'purple', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Temperature [Kelvin]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[2,1])
        ax.plot(x, self.cons_quant * HA_TO_EV * 1000/self.N_atoms,  color = 'darkblue', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('Cons qunt [meV/atom]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


        ax = fig.add_subplot(gs[3,:])
        if self.pressures is None:
            pressure = np.einsum("iaa -> i", self.stresses) * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA /3
        else:
            pressure = self.pressures * BAR_TO_HA_BOHR3**-1 * BAR_TO_GPA
        ax.plot(x, pressure,  color = 'green', lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('P [GPa]', size = 12)
        ax.tick_params(axis = 'both', labelsize = 12)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        
        plt.tight_layout()

        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        if show:
            plt.show()
        
        return

    def get_masses_from_types(self):
        """
        GET THE MASSES FOR ALL THE ATOMS IN THE SNAPSHOTS
        =================================================

        Use units of ASE so UMA
        """
        # Get the masses
        masses_array = np.zeros(self.N_atoms)
        for i, at_type in enumerate(self.types):
            masses_array[i] = ase.data.atomic_masses[ase.data.atomic_numbers[at_type]]

        return masses_array
        
    def get_com_positions(self):
        """
        GET THE CENTER OF MASS POSITION
        ===============================

        In Angstrom

        Returns:
        --------
            -R_com: np.array with shape (snapshots, 3), the center of mass positions
        """
        masses_array = self.get_masses_from_types()

        # Get the center of mass position
        R_com = np.einsum('a, iab -> ib', masses_array, self.positions) /np.sum(masses_array)

        return R_com


    def get_com_velocities(self):
        """
        GET THE CENTER OF MASS VELOCITY
        ===============================

        In BOHR/AU_TIME

        Returns:
        --------
            -V_com: np.array with shape (snapshots, 3), the center of mass velocities
        """
        # Get the masses
        masses_array = np.zeros(self.N_atoms)
        for i, at_type in enumerate(self.types):
            masses_array[i] = ase.data.atomic_masses[ase.data.atomic_numbers[at_type]]

        # Get the center of mass velocities
        V_com = np.einsum('a, iab -> ib', masses_array, self.velocities) /np.sum(masses_array)

        return V_com



    def show_com_motion(self, img_name = None):
        """
        PLOT THE MOTION OF THE CENTER OF MASS
        =====================================

        Checks the drift of the com

        Use units of AMU for the mass from ase
        """
        matplotlib.use('tkagg')
       
        # Get the center of mass positions (self.snaphsots, 3)
        R_com = self.get_com_positions()

        # Time in femptosecond
        x = np.arange(self.snapshots, dtype = int) * self.dt

        
        # Width and height
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(2,3, figure=fig)
        colors = ["red", "green", "purple"]
        for i in range(3):
            ax = fig.add_subplot(gs[0,i])
            ax.plot(x, R_com[:,i],  color = colors[i], lw = 3)
            ax.set_xlabel('Time [fs]', size = 15)
            ax.set_ylabel('R$_{com}$ [Angstrom]', size = 12)
        for i in range(3):
            ax = fig.add_subplot(gs[1,i])
            ax.plot(x, np.gradient(R_com[:,i], x[1] - x[0]) * ANG_FEMPTOSEC_TO_HA,  color = colors[i], lw = 3)
            ax.set_xlabel('Time [fs]', size = 15)
            ax.set_ylabel('V$_{com}$ [Bohr/t$_{au}$]', size = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig(img_name, dpi = 500)
        plt.show()


        # Width and height
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(1,1, figure=fig)
        colors = ["red", "green", "purple"]
        ax = fig.add_subplot(gs[0,0])
        ax.plot(x, np.einsum('abc -> a',self.forces) * HA_BOHR_TO_EV_ANGSTROM,  color = colors[i], lw = 3)
        ax.set_xlabel('Time [fs]', size = 15)
        ax.set_ylabel('F$_{tot}$ [eV/Angstrom]', size = 12)
        plt.tight_layout()
        if not(img_name is None):
            plt.savefig('forces_' + img_name , dpi = 500)
        plt.show()
        
        return


    def get_pair_correlation_functions(self, selected_atoms, custom_ase_atoms = None,
                                       t_range = None, my_r_range = (0.01, 6.0), bins = 500,
                                       ase_atoms_file = "atoms_gr.xyz", save_ase_atoms_file = False, wrap_positions = True, use_pbc = True,
                                       json_file_result = "pair_corr_function.json" , show_results = True, save_plot = False):
        """
        GET THE PAIR CORRELATION FUNCTION
        =================================

        Note MDAnalysis always uses PBC irrespectively of wheater the ase atoms used have PBC.
        Note that to have correct PBC in MDAnalysis we must initialize the dimensions.

        Parameters:
        -----------
            -selected_atoms: list, list of atomic types for which we compute the pair correlation function

            -custom_ase_atoms: ase atoms object
            -dt: the time step in FEMPTOSECOND
            
            -t_range: list of float, the time window in PICOSECOND. In this widow compute the averages. 
                      If not specified we consider just the last half of the trajectory
            
            -my_r_range: tuple, the minimum and maximum value for r in ANGSTROM
            -bins: int, the number of r values
            
            -ase_atoms_file: the name of the xyz file containing all the snapshots info
            -save_ase_atoms_file: bool, if True we do not delete the xyz file ase_atoms_file
            
            -wrap_positions: bool, if True the positions are wrapped in the ase atoms objects
            -use_pbc: bool, if True PBC are imposed in the ase atoms objects
            
            -json_file_result: the name of the json file containing the dictionary with the results
            
            -show_results: bool, if True we print the diffusion constants and the MSD
            -save_plot: bool, if True we save the plot

        Returns:
        --------
            -g_results: a dictionary containing as items the atomic pair types with r and g(r)
        """
        # matplotlib.use('tkagg')
        # Check if there is MD analysis
        if not __MDANALYSIS__:
            raise NotImplementedError("We need MDAnalysis to run the pair correlation function calculations")
        
        # If there are no selected atoms we compute the MSD and D for all the atomic types
        print("\n\n========PAIR CORRELATION FUNCTION with MDanalysis========")
        
        # Get the atomic types for which we want the diffusion constant
        if selected_atoms is None:
            raise ValueError("Provide a list of atom pairs for which you want to compute the g(r)")

        # if my_r_range is None:
        #     my_r_range = (0.0, 6.0)

        # Prepare the ase atoms to be read
        if custom_ase_atoms is None:
            print("Create the ase atoms")
            ase_atoms = self.create_ase_snapshots(wrap_positions = wrap_positions, subtract_com = False, pbc = use_pbc)
        else:
            print("We already have a custom ase atoms")
            ase_atoms = custom_ase_atoms
        
        if t_range is None:
            index_ini = int(self.snapshots //2)
            index_fin = int(self.snapshots)
            # Get the range of times in PICOSECONDS
            t_range = np.asarray([index_ini, index_fin]) * self.dt * 1e-3
        else:
            t_range = np.sort(t_range)
            index_ini = int(t_range[0] * 1e+3/self.dt)
            index_fin = int(t_range[1] * 1e+3/self.dt)

        # Temporary wrtie the ase atoms objects
        ase.io.write(ase_atoms_file, ase_atoms[index_ini:index_fin], format = "xyz")
        print("xxx", len(ase_atoms[index_ini:index_fin]))
        # Get the correct indices (usefule to set the corret shape of the box in MDanalysis)
        my_indices = np.arange(index_ini, index_fin, dtype = int)

        
        # A dictionary with atom types and the corresponding diffusion constant and error
        g_results = {}
        # Read the xyz file
        MD_atoms = MDAnalysis.Universe(ase_atoms_file)

        # Prepare a plot
        if show_results:
            # Width and height
            fig = plt.figure(figsize=(8, 8))
            if len(selected_atoms) == 1:
                gs = gridspec.GridSpec(1, 1, figure = fig)
            else:
                gs = gridspec.GridSpec(len(selected_atoms)//2, 2, figure = fig)

        print("Setting the cell dimension and the time step (this might take a while)...")
        for snapshot, MD_atoms_snapshot in enumerate(MD_atoms.trajectory):
            # Manually set the unit cell dimensions in ANGSTROM
            ind_sn = my_indices[snapshot]
            MD_atoms_snapshot.dimensions = [self.unit_cells[ind_sn,0,0], self.unit_cells[ind_sn,1,1], self.unit_cells[ind_sn,2,2], 90.0, 90.0, 90.0] 
            # MD_atoms_snapshot.dimensions = [ase_atoms[snapshot].cell[0,0], ase_atoms[snapshot].cell[1,1], ase_atoms[snapshot].cell[2,2], 90.0, 90.0, 90.0] 
            # Apply the time step to all frames in PICOSECONDS
            MD_atoms_snapshot.dt = self.dt * 1e-3

        for index, atomic_pair in enumerate(selected_atoms):
            print('\nRDF for {} {} from {:.1f} to {:.1f} ps'.format(atomic_pair[0], atomic_pair[1], t_range[0], t_range[1]))
            # Select only the atomic type we want
            MD_atoms_selected1 = MD_atoms.select_atoms('name {}'.format(atomic_pair[0]))
            # Select only the atomic type we want
            MD_atoms_selected2 = MD_atoms.select_atoms('name {}'.format(atomic_pair[1]))
            # #  Wrap again the positions
            # MD_atoms_selected1.wrap()
            # MD_atoms_selected2.wrap()

            # Compute RDF
            rdf_calc = MDAnalysis.analysis.rdf.InterRDF(MD_atoms_selected1, MD_atoms_selected2,
                                                        range = my_r_range, nbins = bins, norm = 'rdf')
            rdf_calc.run(verbose = True)

            r, gr = rdf_calc.results.bins, rdf_calc.results.rdf
            # Save the results
            g_results.update({"{}{}".format(atomic_pair[0], atomic_pair[1]) : [list(r), list(gr)]})
            
            if show_results:
                ax = fig.add_subplot(gs[index // 2, index %2])
                ax.plot(r, gr, lw = 3, color = "purple", label = "g(r) for {} {}".format(atomic_pair[0], atomic_pair[1]))
                if gr.max() > 10 * gr[-1]:
                    ax.set_ylim(0, 2 * gr[-1])
                ax.set_xlabel('r [Angstrom]', size = 15)
                ax.set_ylabel('g(r)', size = 12)
                ax.tick_params(axis = 'both', labelsize = 12)
                plt.legend(fontsize = 15)
                
        if show_results:
            plt.tight_layout()
            if save_plot:
                plt.savefig("gr.png", dpi = 500)
            plt.show()

        save_dict_to_json(json_file_result, g_results)
        
        if not save_ase_atoms_file:
            subprocess.run("rm {}".format(ase_atoms_file), shell = True)

        return g_results


        


    def get_diffusion_constant(self, t_range = None, time_windows = None, custom_ase_atoms = None, subtract_com = True, selected_atoms = None,
                               ase_atoms_file   = "atoms_msd.xyz", save_ase_atoms_file = False,
                               json_file_result = "self_diffusion.json" , show_results = True, save_plot = False):
        """
        GET THE SELF-DIFFUSION CONSTANT FROM THE FIT OF THE MEAN SQUARE DISPLACEMENT
        ============================================================================

        Remeber that for the MSD the coordinates should not be wrapped otherwise all the distances are fucked up

        Also remember to subtract the COM position during the MD

        Parameters:
        -----------
            -t_range: list of float, the time window  in PIDCOSECOND. In this window compute the averages.

            -time_windows: list of float, the time window in PICOSECOND for the fit of the MSD

            -custom_ase_atoms: ase atoms object
            
            -subtract_com: bool, if true we subtract the center of mass positions to the all atomic positions in the snapshots
            -selected_atoms: list, list of atomic types for which we compute the self diffusion constant
            
            -ase_atoms_file: the name of the xyz file containing all the snapshots info
            -save_ase_atoms_file: bool, if True we do not delete the xyz file ase_atoms_file
            
            -json_file_result: the name of the json file containing the dictionary with the results
            
            -show_results: bool, if True we print the diffusion constants and the MSD
            -save_plot: bool, if True we save the plot

        Returns:
        --------
            -D_results: a dictionary containing as items the atomic types with diffusion constants and its error
            -MSD_results: a dictionary containing as items the atomic types with lagtimes and meas square displacement
        """
        matplotlib.use('tkagg')
        # Check if there is MD analysis
        if not __MDANALYSIS__:
            raise NotImplementedError("We need MDAnalysis to run the self-diffusion calculations")
        # If there are no selected atoms we compute the MSD and D for all the atomic types
        print("\n\n========MSD ANALYSIS========")
        
        # Get the atomic types for which we want the diffusion constant
        if selected_atoms is None:
            selected_atoms = set(self.types)
            print("MSD will be computed for ", selected_atoms)
        else:
            if np.sum(np.isin(selected_atoms, self.types), dtype = int) != len(selected_atoms):
                raise ValueError("{} not found in the snapshots!")

        if custom_ase_atoms is None:
            # Prepare the ase atoms to be read
            ase_atoms = self.create_ase_snapshots(wrap_positions = False, subtract_com = subtract_com, pbc = True)
        else:
            ase_atoms = custom_ase_atoms
            
        if t_range is None:
            index_ini = int(self.snapshots //2)
            index_fin = int(self.snapshots)
            # Get the range of times in PICOSECONDS
            t_range = np.asarray([index_ini, index_fin]) * self.dt * 1e-3
        else:
            # Sort
            t_range = np.sort(t_range)
            index_ini = int(t_range[0] * 1e+3/self.dt)
            index_fin = int(t_range[1] * 1e+3/self.dt)
        ase.io.write(ase_atoms_file, ase_atoms[index_ini:index_fin], format = "extxyz")
        

        # Get the correct indices 
        my_indices = np.arange(index_ini, index_fin, dtype = int)

        # A dictionary with atom types and the corresponding diffusion constant and error
        D_results = {}
        # A dictionary with atom types and the corresponding diffusion constant and error
        MSD_results = {}
        # Read the xyz file
        MD_atoms = MDAnalysis.Universe(ase_atoms_file)
        
        # Prepare a plot
        if show_results:
            # Width and height
            fig = plt.figure(figsize=(8, 8))
            if len(selected_atoms) == 1:
                 gs = gridspec.GridSpec(1, 1, figure = fig)
            else:
                gs = gridspec.GridSpec(len(selected_atoms)//2, 2, figure = fig)

        print("Setting the cell dimension and the time step...")
        for snapshot, MD_atoms_snapshot in enumerate(MD_atoms.trajectory):
            # Manually set the unit cell dimensions in ANGSTROM
            # Get the correct id of the snapshot
            ind_snap = my_indices[snapshot]
            MD_atoms_snapshot.dimensions = [self.unit_cells[ind_snap,0,0], self.unit_cells[ind_snap,1,1], self.unit_cells[ind_snap,2,2], 90.0, 90.0, 90.0]
            # Apply the time step to all frames in PICOSECONDS
            MD_atoms_snapshot.dt = self.dt* 1e-3
            # if snapshot % 100 == 0:
            #     print(f"Frame {MD_atoms_snapshot.frame}: Unit cell = {MD_atoms_snapshot.dimensions}")
        

        for index, atomic_type in enumerate(selected_atoms):
            # Select only the atomic type we want
            MD_atoms_selected = MD_atoms.select_atoms('name {}'.format(atomic_type))
            # Prepare the MSD analysis
            MSD_tool = MDAnalysis.analysis.msd.EinsteinMSD(MD_atoms_selected, select = 'all', msd_type = 'xyz',  fft = True)
            print("\nGetting the Mean Square Displacement for {} from {:.2f} to {:.2f} ps".format(atomic_type, t_range[0], t_range[1]))
            # Run the calculation
            MSD_tool.run()
            print("The number of atoms is {} and the number of frames {}".format(MSD_tool.n_particles, MSD_tool.n_frames))
            # PICOSCOND and ANGSTROM^2
            

            # print(MSD.results.msds_by_particle.shape )
            my_msd =  MSD_tool.results.timeseries
            # prepare the lagtimes in PICOSECOND
            timestep = MD_atoms.trajectory[0].dt
            lagtimes = np.arange(len(MD_atoms.trajectory)) * timestep

            if time_windows is None:
                start_time,  end_time  = lagtimes[len(lagtimes)//2] - 0.3 * lagtimes[-1], lagtimes[len(lagtimes)//2] + 0.3 * lagtimes[-1]
            else:
                start_time,  end_time  = time_windows
            print("The fit of MSD is done from {:.2f} to {:.2f} ps with a total time {:.2f} ps".format(start_time, end_time, lagtimes[-1]))
            mask = (start_time < lagtimes) & (lagtimes < end_time)
            
            linear_model = scipy.stats.linregress(lagtimes[mask], my_msd[mask])
            slope, slope_error, intercept = linear_model.slope, linear_model.stderr, linear_model.intercept
            # dim_fac is 3 as we computed a 3D msd with 'xyz', ANGSTROM2/PICOSECOND
            D, D_error = slope * 1/(2 * MSD_tool.dim_fac), slope_error * 1/(2 * MSD_tool.dim_fac)
            # # Now in METER^2/SECOND
            # D *= 1e-8
            # D_error *= 1e-8

            # Now in ANGSTROM2/PICOSECOND and ANGSTROM2
            D_results.update({atomic_type : [D, D_error, intercept]})
            # Now in PICOSECOND, ANGSTROM^2
            MSD_results.update({atomic_type : [list(lagtimes), list(my_msd)]})
            print("=>Diffusion constant for atom {}".format(atomic_type))
            print("==>D {:.3e} +- {:.3e} ".format(D, D_error) + "Ang$^2$/ps")
            # plt.plot(lagtimes, res)
        
            if show_results:
                ax = fig.add_subplot(gs[index // 2, index %2])
                ax.set_title("MSD for {}".format(atomic_type))
                # PICOSCOND and ANGSTROM^2
                ax.plot(lagtimes, my_msd, lw = 3, color = "k")
                ax.plot(lagtimes[mask], lagtimes[mask] * slope + linear_model.intercept, lw = 3, ls = ":", color = "red",
                        label = 'Fit D={:.2e} '.format(D) + "Ang$^2$/ps")
                ax.set_xlabel('Time [ps]', size = 15)
                ax.set_ylabel('MSD [Angstrom$^2$]', size = 12)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(axis = 'both', labelsize = 12)
                plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig("diffusion.png", dpi = 500)
        if show_results:
            plt.show()

        # Save the results to a json file
        save_dict_to_json(json_file_result, D_results)
        save_dict_to_json("raw_" + json_file_result, MSD_results)
        
        # Remove the ase atoms object
        if not save_ase_atoms_file:
            subprocess.run("rm {}".format(ase_atoms_file), shell = True)

        return D_results, MSD_results


    def get_conductivity(self, types_q = {"Na" : +1, "Cl" : -1}, custom_ase_atoms = None,
                         time_window = None,
                         use_julia = False, python_normalize = True,
                         pad = True, omega_ir = [0, 5000], smearing = None,
                         save_sigma = False, name_sigma_file = "sigma.json",
                         test = True, show_results = True, name_plot = 'sigma_plot.png',
                         write_table_file = False,
                         name_table_file = 'cond',
                         ase_atoms_file = 'ase_cond.xyz'):
        """
        GET THE CONDUCTIVITY
        =====================

        First we  compute the current current correlation function both in time and frequency

        then we compute the integral, i.e. the DC conductivity from

        ..math:: \sigma = \frac{1}{3 V k T} \int dt \left\langle J(t) J(0) \right\rangle

        This should coincide with

        ..math:: \sigma = \sigma(\omega=0)


        Parameters:
        -----------
            -types_q: dict with the atomic types and the ox charges
            
            -time_window: list, the initial and final time for sampling in PICOSECONDS
            
            -use_julia: bool, if True the dipole dipole correlation fucntion is computed using the Windowed average in JULIA
            -python_normalize: bool, if True the FFT is normalized so it coicides with the windowed average of Julia

            -pad: bool, if True we pad the sigmal before doing FFT
            -omega_ir: list, the range to plot in cm-1
            -smearing: float the smaering in cm-1 for plotting the IR spectra
            
            -save_sigma: bool, if True we save the conductivity evolution as a json file
            -name_sigma_file: str, the name of the json file with the integrated copn
            -test: bool: if True we compare the julia and python correlation functions
            -show_results: bool

            -write_table_file: bool, if True we write the file for the cepstral anaylsis in sportran
            -name_table_file: str, the name of the table file

            
        """
        if time_window is None:
            index1 = 0
            index2 = self.snapshots + 1
            t_min = 0
            t_max = self.snapshots * self.dt * 1e-3
        else:
            t_min, t_max = np.sort(np.asarray(time_window))
            # Convert in FEMPTOSCEON ONLY TO GET THE INDICES
            index1 = int(t_min * 1e+3/self.dt)
            index2 = int(t_max * 1e+3/self.dt)
            if index2 > self.snapshots:
                index2 = self.snapshots

        print("\n\n================= CONDUCTIVITY ANALYSIS from {:.2f} to {:.2f} ps =================".format(t_min, t_max))
            
        # Prepare the ase atoms to be read
        # ase_atoms = self.create_ase_snapshots(wrap_positions = False, subtract_com = False, pbc = False)
        # Prepare the ase atoms to be read
        if custom_ase_atoms is None:
            print("Create the ase atoms")
            ase_atoms = self.create_ase_snapshots(wrap_positions = False, subtract_com = False, pbc = False)
        else:
            print("We already have a custom ase atoms")
            ase_atoms = custom_ase_atoms

        # Set up the class
        cond = Conductivity.Conductivity()
        
        # Give the dipoles and the time step
        cond.init(ase_atoms = ase_atoms[index1: index2], dt = self.dt,
                  temperatures = self.temperatures[index1: index2],
                  types_q = types_q)

        # Set the time correlations
        cond.set_correlations(use_julia = use_julia, python_normalize = python_normalize)

        # Compute the time integral of the correlation function and eventually plot everyhting
        cond.get_correlation_function(omega_min_max = omega_ir, smearing = smearing, pad = pad,
                                      save_data = save_sigma, name_json_file = name_sigma_file,
                                      show_results = show_results, name_plot = name_plot)
        # Input file for SportTran code
        if write_table_file:
            cond.write_flux_file(filename = name_table_file)

        if test:
            cond.test_implementation()

        return cond.results

        

        

    

    def get_ir_spectra(self, time_window = None,
                       Nmax = 5, tol_refold = 1.0, debug_refold = False,
                       use_julia = False, python_normalize = False,
                       omega_ir = [0, 5000], smearing = None, save_ir = False, plot_results = True,
                       test = False):
        """
        GET THE IR SPECTRA FROM DIPOLE-DIPOLE CORRELATION FUNCTION
        ==========================================================

        Parameters:
        -----------
            -time_window: list, the initial and final time for sampling in PICOSECONDS

            -Nmax: int, the max integer to add/subtract to the dipoles to refold them
            -tol_refold: float, the tolerance to consider the dipoles snapshots continuous
            -debug_refold: bool, if True we print the refolding details
            
            -use_julia: bool, if True the dipole dipole correlation function is computed using the Windowed average in JULIA
            -python_normalize: bool, if True the FFT is normalized so it coincides with the windowed average of Julia

            -omega_ir: list, the range to plot in cm-1
            -smearing: float the smaering in cm-1 for plotting the IR spectra
            -save_ir: bool, if True we save the IR spectra as a json file

            -test: bool: if True we compare the julia and python correlation functions
        """
        # For the time we use PICOSECONDS
        if time_window is None:
            index1 = 0
            index2 = self.snapshots + 1
            t_min = 0
            t_max = self.snapshots * self.dt * 1e-3
        else:
            t_min, t_max = np.sort(np.asarray(time_window))
            # Convert in fs only to get the indices
            index1 = int(t_min * 1e+3/self.dt)
            index2 = int(t_max * 1e+3/self.dt)

        
        print("\n\n================= VIBRATIONAL ANALYSIS from {:.2f} to {:.2f} ps =================".format(t_min, t_max))
        
        # First we refold the dipoles
        self.refold_all_dipoles(Nmax = Nmax, tol = tol_refold, debug = debug_refold)

        # Set up the class
        vibrations = Vibrational.Vibrational()
        
        # Give the dipoles and the time step
        vibrations.init(self.dipoles[index1:index2], self.dt)
        
        # Get the dipole-dipole time correlation function
        vibrations.set_dipole_dipole_correlation_function(use_julia = use_julia, python_normalize = python_normalize)

        if plot_results:
            # Plot the results
            vibrations.plot_results(omega_min_max = omega_ir, delta = smearing, save_data = save_ir)

        if test:
            vibrations.test_implementation()
        

    
    def refold_all_dipoles(self, Nmax = 10, tol = 1.0, debug = False):
        """
        REFOLD ALL THE DIPOLES USING MULTIPLES OF BERRY QUANTUM
        =======================================================

        If the component i of the dipoles is discontinous we add (or subtract) a multiple integer of the trivial phase

        Parameters:
        -----------
            -Nmax: int, we try to refold the dipoles
            -tol: float, the tolerance need to find discontinuties in the dipole moments. If |P[i] - P[i+1]| > tol we try to make it continous.
            -debug: bool, to debug and test
        """
        cmps = ["X", "Y", "Z"]
        # Refold the dipoles 
        for cart in range(3):
            print("\n\n\n   ========== REFOLDING {} ==========".format(cmps[cart]))
            new_P = refold_dipole(self.dipoles[:,cart], self.dipoles_quantum[:,cart,cart],
                                  Nmax = Nmax, tol = tol, debug = debug)
            self.dipoles[:,cart] = np.copy(new_P)
        
        





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


def refold_dipole(P, cell, Nmax = 1, tol = 1.0, debug = True):
    """
    REFOLD THE DIPOLES USING INTEGER MULTIPLES OF THE BERRY QUANTUM OF POLARIZATION
    ===============================================================================

    For every P[i], very simply we check if the difference |P[i] - P[i-1]| is to large. 
    If so, we subtract/add multiple integers of the quantum of polarization to P[i] 
    
    Parameters:
    -----------
        -P:    np.array with shape N, the dipoles along a given axis x, y, z for a trajectory of size N
        -cell: np.array with shape N, the Berry quantum along the given axis for a trajectory of size N
        -Nmax: int, the integer number we multiply the quantum of polarization to have a continuous function
        -tol: float, the tolerance which we use to tell if the dipoles are continuous or not
        -debug: bool

    Returns:
    --------
        -Pini: np.array with shape N, the CONTINUOUS dipoles along a given axis x, y, z for a trajectory of size N
        
    """
    # The dipoles we manipulate
    Pini = np.copy(P)
    # Get the size of the trajectory
    trajectory_size = len(P)
    
    all_failure = []
    for i in range(1, trajectory_size):
        
        delta = np.abs(Pini[i] - Pini[i-1])
        if delta > tol:
            all_failure.append(i)
            Pold = np.copy(Pini[i])

            for integer in range(-Nmax, +Nmax + 1):
                Pnew = Pold + integer * cell[i]
                new_delta = np.abs(Pnew - Pini[i-1])
                if new_delta < tol:
                    if debug:
                        print("     INDEX {} delta {:.1f} solved with delta {:.1f} @ step {}".format(i, delta, new_delta, integer))
                        print(Pini[i], Pnew)
                    Pini[i] = np.copy(Pnew)
                    # Get out from this loop
                    break
                    
    if debug:
        plt.plot(P, label = "BEFORE")
        plt.plot(Pini, ls = ":", label = "AFTER")
        plt.legend()
        plt.show()

    return Pini
                


def print_units():
    """
    PRINT THE UNITS USED IN THE CLASS AtomiSnapshots
    ================================================
    """
    print()
    print("=========== UNITS USED IN THE CODE ===========")
    print("========== UNITS OF THE ATTRIBUTES ===========")
    print("POSITIONS in ANGSTROM")
    print("VELOCITIES in BOHR/AUTIME")
    print("ENERGIES in HARTREE")
    print("FORCES in HARTREE/BOHR")
    print("STRESS in HARTREE/BOHR3")
    print("DIPOLES in AU")
    print("")
    print()


def transform_voigt(tensor, voigt_to_mat = False):
    """
    TRANSFORM TO VOIGT NOTATION
    ===========================
    
    Copied from Cellconstructor useful to convert everything in ase format.

    The voigt notation transforms a symmetric tensor

    ..math: \sigma_{00} \sigma_{01} \sigma_{02}
            \sigma_{10} \sigma_{11} \sigma_{12}
            \sigma_{20} \sigma_{21} \sigma_{22}

    to a 6 compoentn vector

    ..math: \sigma_{00} \sigma_{11} \sigma_{22} \sigma_{12} \sigma_{02} \sigma_{01}
     
    Parameters:
    -----------
        -voit_to_mat. bool default is False, Otherwise it is assumed as a 3x3 symmetric matrix (upper triangle will be read).
                      If True the tensor is assumed to be in voigt format.
    Returns:
    --------
        -new_tensor: np.array
    """

    if voigt_to_mat:
        assert len(tensor) == 6
        new_tensor = np.zeros((3,3), dtype = type(tensor[0]))
        for i in range(3):
            new_tensor[i,i] = tensor[i]

        new_tensor[1,2] = new_tensor[2,1] = tensor[3]
        new_tensor[0,2] = new_tensor[2,0] = tensor[4]
        new_tensor[1,0] = new_tensor[0,1] = tensor[5]
    else:
        assert tensor.shape == (3,3)
        new_tensor = np.zeros(6, dtype =  type(tensor[0,0]))
        # First 
        for i in range(3):
            new_tensor[i] = tensor[i,i]
        new_tensor[3] = tensor[1,2]
        new_tensor[4] = tensor[0,2]
        new_tensor[5] = tensor[0,1]
    
    return new_tensor
