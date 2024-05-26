from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing


# Check if COLVAR_compare exists and delete it if it does
if os.path.exists('COLVAR_compare'):
    os.remove('COLVAR_compare')
if os.path.exists('HILLS_compare'):
    os.remove('HILLS_compare')


def plot_cvs(colvar_path, cvs):
    """
    plot cvs against time
    """

    colvar_data = np.loadtxt(colvar_path)

    # time = colvar_data[:, 0]

    # for i, cv_label in enumerate(cvs):
    #     cv = colvar_data[:, i+1]
    #     plt.scatter(time, cv, s=2, label=cv_label, marker='x')

    phi = colvar_data[:, 1]
    psi = colvar_data[:, 2]

    plt.scatter(phi, psi, s=1, marker='x')


    # Adding labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Collective Variables over Time")
    # plt.legend()

    # Show plot
    plt.grid(True)

    plt.savefig("test_phi_psi", bbox_inches='tight')

    # plt.show()


def plot_cvs_time(colvar_path, cvs):
    """
    plot cvs against time
    """

    colvar_data = np.loadtxt(colvar_path)

    time = colvar_data[:, 0]

    for i, cv_label in enumerate(cvs):
        cv = colvar_data[:, i+1]
        plt.scatter(time, cv, s=1, label=cv_label, marker='x')


    # Adding labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Collective Variables over Time")
    # plt.legend()

    # Show plot
    plt.grid(True)

    plt.savefig("test_cvs_time", bbox_inches='tight')

    # plt.show()

# def evaluate(colvar_path, begin_index, end_index):
#     """
#     evaluate the prob_hist based on the kl divergence, and return its absolute value
#     """
#     colvar_data = np.loadtxt(colvar_path)

#     # get the data from the resulting COLVAR file
#     phi = colvar_data[:, 1][begin_index : end_index]
#     psi = colvar_data[:, 2][begin_index : end_index]
#     # bias = molsim.colvar_data[:, -1]

#     # print("phi", phi)
#     # print("psi", psi)
#     # print("bias", bias)

#     # use this data to make a probability histogram
#     hist = np.histogram2d(phi, psi, bins=10, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=None)
#     # print("hist", hist)

#     prob_hist = hist[0]

#     hist_no_zeros = prob_hist + 1

#     normalized_hist = preprocessing.normalize(hist_no_zeros)

#     print("normalized_hist", normalized_hist)
    
#     # calculate the kl divergence based on the provided probability histogram
#     div_kl = np.sum(normalized_hist * np.log2(normalized_hist))

#     # print("div_kl", div_kl)

#     return abs(div_kl)

def get_prob_hist_from_colvar(colvar_path, begin_index, end_index):

    colvar_data = np.loadtxt(colvar_path)

    # get the data from the resulting COLVAR file
    phi = colvar_data[:, 1][begin_index : end_index]
    psi = colvar_data[:, 2][begin_index : end_index]
    # bias = molsim.colvar_data[:, -1]

    # print("phi", phi)
    # print("psi", psi)
    # print("bias", bias)

    # use this data to make a probability histogram
    hist = np.histogram2d(phi, psi, bins=10, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=None)
    # print("hist", hist)

    prob_hist = hist[0]

    return prob_hist


def evaluate(prob_hist):
    """
    evaluate the prob_hist based on the kl divergence, and return its absolute value
    """

    hist_no_zeros = prob_hist + 1

    normalized_hist = preprocessing.normalize(hist_no_zeros)

    # print("normalized_hist", normalized_hist)
    
    # calculate the kl divergence based on the provided probability histogram
    div_kl = np.sum(normalized_hist * np.log2(normalized_hist))

    # print("div_kl", div_kl)

    return abs(div_kl)


def evaluate2(prob_hist):
    """
    evaluate the prob_hist by finding the mean squared distance between it and a uniform histogram
    """
    # find number of bins
    shape = prob_hist.shape

    # The number of bins in each dimension is equal to the size of that dimension
    num_bins = np.prod(shape)

    # goal value
    goal_value = 1/num_bins

    # get normalized histogram
    normalized_hist = preprocessing.normalize(prob_hist)


    # calculate mean squared error
    mse = np.square(np.subtract(normalized_hist,goal_value)).mean()

    return mse



# Load PDB file into PDBFile object
pdb = PDBFile('../alanine-dipeptide-implicit.pdb')

# Load forcefield into ForceField object
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# Combine force field with molecular topology from PDB file to create a complete 
# mathematical description of the system (as a System object)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,
                                 nonbondedCutoff=1*nanometer, constraints=HBonds)

script = """
# set up two variables for Phi and Psi dihedral angles 
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
#
# Activate metadynamics in phi and psi
# depositing a Gaussian every 500 time steps,
# with height equal to 1.2 kJ/mol,
# and width 0.35 rad for both CVs. 

metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE=HILLS_compare2 

# monitor the two variables and the metadynamics bias potential
PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=COLVAR_compare2
"""

system.addForce(PlumedForce(script))

# Read MDP file and set parameters
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)
integrator.setConstraintTolerance(0.00001)  # Set constraint tolerance based on MDP file
# integrator.setAngularMomentum(True)  # Set angular momentum removal based on MDP file

nsteps = 2500000  # Set the number of steps based on MDP file
dt = 0.002  # Set the time step based on MDP file

# Combine the molecular topology, system, and integrator to begin a new simulation
simulation = Simulation(pdb.topology, system, integrator)

# Set initial atom positions
simulation.context.setPositions(pdb.positions)

# Perform a local energy minimization
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

# Run the simulation
simulation.step(nsteps)

plot_cvs("COLVAR_compare2", ["phi", "psi"])
plot_cvs_time("COLVAR_compare2", ["phi", "psi"])

begin_hist = get_prob_hist_from_colvar("COLVAR_compare2", 1, 50000)
middle_hist = get_prob_hist_from_colvar("COLVAR_compare2", 100000, 500000)
end_hist = get_prob_hist_from_colvar("COLVAR_compare2", 2000000, 25000000)

print("eval: 1-50000", evaluate(begin_hist))
print("eval2: 1-50000", evaluate2(begin_hist))
print("eval: 100000-500000", evaluate(middle_hist))
print("eval2: 100000-500000", evaluate2(middle_hist))
print("eval: 2000000-2500000", evaluate(end_hist))
print("eval2: 2000000-2500000", evaluate2(end_hist))


