from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce
import numpy as np
import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt
import os
import re
from bias import MolSim

class MolSimCMA:
    def __init__(self, default_sigma, resolution, template_hills_file):
        self.default_sigma = default_sigma
        self.resolution = resolution
        self.template_hills_file = template_hills_file

        # create template hills file
        self.create_template_hills()
    
    def create_template_hills(self):

        if os.path.exists(self.template_hills_file):
            os.remove(self.template_hills_file)

        with open (self.template_hills_file, "x") as hills:
            hills.write("#! FIELDS time phi psi sigma_phi sigma_psi height biasf\n\
                        #! SET multivariate false\n\
                        #! SET kerneltype stretched-gaussian\n\
                        #! SET min_phi -pi\n\
                        #! SET max_phi pi\n\
                        #! SET min_psi -pi\n\
                        #! SET max_psi pi\n")

            # need to generalize this later on!
            height_index = 0
            step_size = 2*np.pi / self.resolution
            for phi in np.arange(-np.pi, np.pi, step_size):
                for psi in np.arange(-np.pi, np.pi, step_size):
                    hills.write(f"0 {phi} {psi} {self.default_sigma} {self.default_sigma} h{height_index} {1}\n")

                    height_index += 1
                
    def update_hills(self, x):
        """
        This function opens the template_hills_file,
        replaces the template heights by corresponding vector values in x,
        then writes this to the HILLS file
        """
        # open template file and read contents
        with open(self.template_hills_file, "r") as hills:
            content = hills.read()
            # print(content)

        # replace h0,...,h100 by corresponding vector values
        for i in range(100):
            pattern = rf'\bh{i}\b'
            content = re.sub(pattern, str(x[i]), content)

        # write to output_hills_file
        with open("HILLS", "w") as hills:
            hills.write(content)

    def run_simulation(self, x):
        """
        
        """
        # write x to HILLS file 
        self.update_hills()

        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        molsim = MolSim("alanine-dipeptide-implicit.pdb", forcefield)
        molsim.add_bias()
        molsim.run_sim()

        prob = -1*np.mean(cvs[-5:-1])/1056
        return prob

        # use HILLS file to "hardcode bias" and run simulation

        # find P

        return P


    def evaluate(self, P):
        # run metadynamics simulation on these values & find the probability distribution
        # P = ...
        return np.sum(P * np.log(P))
    
    def CMA(self):
        optimizer = CMA(mean=np.zeros(100), sigma=0.2)

        generations = 8
        for generation in range(generations):
            solutions = []
            
            # pick ideal population size!
            for _ in range(optimizer.population_size):
                x = optimizer.ask()

                value = self.evaluate(x)

                # append solutions by the point and its value according to 
                # the evaluate function
                solutions.append((x, value))
                
                print(f"#{generation} {value} (x={x})")

            optimizer.tell(solutions)


if __name__ == "__main__":


    test = MolSimCMA(0.15, 10, "TEMPLATE_HILLS")
    test.create_template_hills()
    x = np.zeros(100)
    test.update_hills(x)