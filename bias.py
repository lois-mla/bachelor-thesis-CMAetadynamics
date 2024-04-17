from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce
import numpy as np
import matplotlib.pyplot as plt


class MolSim:
    def __init__(self, pdb_filename, forcefield):
        # load PDB file into PDBFile object
        self.pdb = PDBFile(pdb_filename)

        # set bias to be False
        self.bias_script = False

        # ? what are the amber files? do we want to be able to change those
        self.forcefield = forcefield

    # have to generalize this later on
    def add_bias(self):
        self.bias_script = """
        RESTART
        # set up two variables for Phi and Psi dihedral angles 
        # = the collective variables
        phi: TORSION ATOMS=5,7,9,15
        psi: TORSION ATOMS=7,9,15,17
        #
        # Activate metadynamics in phi and psi
        # depositing a Gaussian every 500 time steps,
        # with height equal to 1.2 kJ/mol,
        # and width 0.35 rad for both CVs. 
        #
        metad: METAD ARG=phi,psi PACE=500000000 HEIGHT=0 SIGMA=0.15,0.15 FILE=HILLS

        # monitor the two variables and the metadynamics bias potential
        PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=COLVAR
        """

    # ! generalize later on
    def run_sim(self):
        # combine force field with molecular topology from PDB file to create a complete
        # mathematical description of the system (as a System object)
        system = self.forcefield.createSystem(
            self.pdb.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer, constraints=HBonds
        )

        if self.bias_script:
            system.addForce(PlumedForce(self.bias_script))

        # creates the integrator to use for advancing the equations of motion
        # (in this case the LangevinMiddleIntegrator for Langevin dynamics)
        integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)

        # combine the molecular topology, system, and integrator to begin a new simulation
        simulation = Simulation(self.pdb.topology, system, integrator)

        # set initial atom positions
        simulation.context.setPositions(self.pdb.positions)

        # tells OpenMM to perform a local energy minimization
        simulation.minimizeEnergy()

        # create reporter and append to list of reporters
        # the output is reported to the reporter during the simulation,
        # this reporter writes the output to a PDB file
        # (in this case every 1000 time steps)
        simulation.reporters.append(PDBReporter("output.pdb", 1000))

        # add another reporter to print out some basic information every 1000 time steps:
        # the current step index, the potential energy of the system, and the temperature
        # output file = stdout -> write the results to the console
        simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))

        # run the simulation, integrating the equations of motion for 10,000 time steps.
        simulation.step(10000)


    def read_colvar(self, file_path):
        # check if it exists

        data = []
        with open(file_path, "r") as file:
            # Skip header lines starting with "#!"
            for line in file:
                if not line.startswith("#!"):
                    break

            # Read data lines
            for line in file:
                line = line.strip()

                if line:
                    parts = line.split()

                    if len(parts) >= 4:  # Ensure at least four columns are present
                        time = float(parts[0])
                        phi = float(parts[1])
                        psi = float(parts[2])
                        metad_bias = float(parts[3])
                        data.append((time, phi, psi, metad_bias))
                    else:
                        print("Warning: Incomplete data line:", line)

        return data

        # Example usage
        file_path = "your_covar_file.covar"
        covar_data = read_covar_file(file_path)
        print(covar_data)

        # time = []
        # variable1 = []
        # variable2 = []
        # with open(filename, 'r') as file:
        #         for line in file:
        #         if line.strip():  # Skip empty lines
        #                 data = line.split()  # Assuming space-separated data
        #                 time.append(float(data[0]))
        #                 variable1.append(float(data[1]))
        #                 variable2.append(float(data[2]))
        # return time, variable1, variable2

    def plot_cvs(self):
        
        colvar_data = np.loadtxt('COLVAR')

        time = colvar_data[:, 0]
        phi = colvar_data[:, 1]
        psi = colvar_data[:, 2]
        
        # Plotting
        plt.scatter(time, phi, label="phi", marker='x')
        plt.scatter(time, psi, label="psi", marker='x')

        # Adding labels and legend
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Collective Variables over Time")
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    molsim = MolSim("alanine-dipeptide-implicit.pdb", forcefield)
    molsim.add_bias()
    molsim.run_sim()
#     print(molsim.read_colvar("COLVAR"))
#     print(np.loadtxt('COLVAR'))
    molsim.plot_cvs()




    # # load PDB file into PDBFile object
    # pdb = PDBFile('alanine-dipeptide-implicit.pdb')

    # # load forcefield into into ForceField object
    # forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # # combine force field with molecular topology from PDB file to create a complete
    # # mathematical description of the system (as a System object)
    # system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic,
    #         nonbondedCutoff=1*nanometer, constraints=HBonds)

    # # script = """
    # # d: DISTANCE ATOMS=1,10
    # # PRINT FILE=COLVAR STRIDE=10"""
    # script = """
    # RESTART
    # # set up two variables for Phi and Psi dihedral angles
    # # = the collective variables
    # phi: TORSION ATOMS=5,7,9,15
    # psi: TORSION ATOMS=7,9,15,17
    # #
    # # Activate metadynamics in phi and psi
    # # depositing a Gaussian every 500 time steps,
    # # with height equal to 1.2 kJ/mol,
    # # and width 0.35 rad for both CVs.
    # #
    # metad: METAD ARG=phi,psi PACE=500000000 HEIGHT=0 SIGMA=0.15,0.15 FILE=HILLS

    # # monitor the two variables and the metadynamics bias potential
    # PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=COLVAR
    # """

    # system.addForce(PlumedForce(script))

    # # creates the integrator to use for advancing the equations of motion
    # # (in this case the LangevinMiddleIntegrator for Langevin dynamics)
    # integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    # # combine the molecular topology, system, and integrator to begin a new simulation
    # simulation = Simulation(pdb.topology, system, integrator)

    # # set initial atom positions
    # simulation.context.setPositions(pdb.positions)

    # # tells OpenMM to perform a local energy minimization
    # simulation.minimizeEnergy()

    # # create reporter and append to list of reporters
    # # the output is reported to the reporter during the simulation,
    # # this reporter writes the output to a PDB file
    # # (in this case every 1000 time steps)
    # simulation.reporters.append(PDBReporter('output.pdb', 1000))

    # # add another reporter to print out some basic information every 1000 time steps:
    # # the current step index, the potential energy of the system, and the temperature
    # # output file = stdout -> write the results to the console
    # simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
    #         potentialEnergy=True, temperature=True))

    # # run the simulation, integrating the equations of motion for 10,000 time steps.
    # simulation.step(10000)
