from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce

# load PDB file into PDBFile object
pdb = PDBFile('alanine-dipeptide-implicit.pdb')

# load forcefield into into ForceField object
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# combine force field with molecular topology from PDB file to create a complete 
# mathematical description of the system (as a System object)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=1*nanometer, constraints=HBonds)

# script = """
# d: DISTANCE ATOMS=1,10
# PRINT FILE=COLVAR STRIDE=10"""
script = """
# set up two variables for Phi and Psi dihedral angles 
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
#
# Activate metadynamics in phi and psi
# depositing a Gaussian every 500 time steps,
# with height equal to 1.2 kJ/mol,
# and width 0.35 rad for both CVs. 
#
metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE=HILLS 

# monitor the two variables and the metadynamics bias potential
PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=COLVAR
"""

system.addForce(PlumedForce(script))

# creates the integrator to use for advancing the equations of motion 
# (in this case the LangevinMiddleIntegrator for Langevin dynamics)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

# combine the molecular topology, system, and integrator to begin a new simulation
simulation = Simulation(pdb.topology, system, integrator)

# set initial atom positions
simulation.context.setPositions(pdb.positions)

# tells OpenMM to perform a local energy minimization
simulation.minimizeEnergy()

# create reporter and append to list of reporters
# the output is reported to the reporter during the simulation,
# this reporter writes the output to a PDB file 
# (in this case every 1000 time steps)
simulation.reporters.append(PDBReporter('output.pdb', 1000))

# add another reporter to print out some basic information every 1000 time steps:
# the current step index, the potential energy of the system, and the temperature
# output file = stdout -> write the results to the console
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))

# run the simulation, integrating the equations of motion for 10,000 time steps. 
simulation.step(10000)
