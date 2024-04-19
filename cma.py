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
from sklearn import preprocessing

class MolSimCMA:
    def __init__(self, default_sigma, resolution, template_hills_file):
        self.default_sigma = default_sigma
        self.resolution = resolution
        self.template_hills_file = template_hills_file

        # create template hills file
        self.create_template_hills()
    
    # ! generalize later
    def create_template_hills(self):
        """
        Creates a template hills file containing the default values for the time, the cvs, their 
        sigmas and the biasf. For heights h0,...,hN are added, which can be overwritten by values 
        from an input vector.
        """

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

            height_index = 0
            step_size = 2*np.pi / self.resolution
            for phi in np.arange(-np.pi, np.pi, step_size):
                for psi in np.arange(-np.pi, np.pi, step_size):
                    hills.write(f"0 {phi} {psi} {self.default_sigma} {self.default_sigma} h{height_index} 1\n")

                    height_index += 1
                
    def update_hills(self, x, gen, sample):
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

        # write to output_hills_file, specifying the generation & sample
        with open(f"gen{gen}-sample{sample}-HILLS", "w") as hills:
            hills.write(content)


    def run_simulation(self, x, gen, sample):
        """
        
        """
        # write x to HILLS file 
        self.update_hills(x, gen, sample)

        # ! move this somewhere else later on
        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        bias_script = f"""
        RESTART
        # set up two variables for Phi and Psi dihedral angles 
        # = the collective variables
        phi: TORSION ATOMS=5,7,9,15
        psi: TORSION ATOMS=7,9,15,17
        #
        # Activate metadynamics in phi and psi
        # with height equal to 1.2 kJ/mol,
        # and width 0.35 rad for both CVs. 
        #
        metad: METAD ARG=phi,psi PACE=500000000 HEIGHT=0 SIGMA=0.15,0.15 FILE=gen{gen}-sample{sample}-HILLS

        # monitor the two variables and the metadynamics bias potential
        PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=gen{gen}-sample{sample}-COLVAR
        """
        cvs = ["phi", "psi"]

        # make MolSim object
        molsim = MolSim("alanine-dipeptide-implicit.pdb", forcefield, cvs, bias_script, gen, sample)

        # run simulation
        molsim.run_sim()

        # get the data from the resulting COLVAR file
        phi = molsim.colvar_data[:, 1]
        psi = molsim.colvar_data[:, 2]
        bias = molsim.colvar_data[:, -1]

        print("phi", phi)
        print("psi", psi)
        print("bias", bias)

        # use this data to make a probability histogram
        hist = np.histogram2d(phi, psi, bins=10, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=None)

        return hist[0]


    def evaluate(self, prob_hist):
 
        hist_no_zeros = prob_hist + 1

        normalized_hist = preprocessing.normalize(hist_no_zeros)

        print("normalized_hist", normalized_hist)
        
        # calculate the kl divergence based on the provided probability histogram
        div_kl = np.sum(normalized_hist * np.log2(normalized_hist))

        print("div_kl", div_kl)

        return abs(div_kl)
    

    def evaluate2(self, prob_hist):

        # for a uniform distribution, 

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

    
    
    def CMA(self):

        # initialize optimizer
        optimizer = CMA(mean=np.zeros(100), sigma=0.2)

        generations = 8
        for generation in range(generations):
            solutions = []
            
            for sample in range(optimizer.population_size):

                # ask optimizer for a sample
                x = optimizer.ask()

                # run the simulation on this sample and get the corresponding probability histogram
                prob_hist = self.run_simulation(x, generation, sample)

                # evaluate the prob_hist with kl divergence
                value = self.evaluate2(prob_hist)

                # append solutions by both x and its kl div value
                solutions.append((x, value))
                
                print(f"#{generation} {value} (x={x})")

            plot_cvs_per_generation(generation, ["phi", "psi"], solutions)

            # tell the optimizer the solutions
            optimizer.tell(solutions)


def plot_cvs(colvar_path, cvs):

    colvar_data = np.loadtxt(colvar_path)

    time = colvar_data[:, 0]

    for i, cv_label in enumerate(cvs):
        cv = colvar_data[:, i+1]
        plt.scatter(time, cv, label=cv_label, marker='x')

    # Adding labels and legend
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Collective Variables over Time")
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

def plot_cvs_per_generation(gen, cvs, solutions=None):
    
    f, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12), tight_layout=True)

    for sample in range(16):

        colvar_data = np.loadtxt(f"gen{gen}-sample{sample}-COLVAR")

        time = colvar_data[:, 0]

        ax = axes.flatten()[sample]
        
        for i, cv_label in enumerate(cvs):
            cv = colvar_data[:, i+1]
            ax.scatter(time, cv, label=cv_label, marker='x')

        # Adding labels and legend
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        
        if solutions:
            ax.set_title(f"sample {sample}, kl_div = {solutions[sample][1]}")
        else:
            ax.set_title(f"sample {sample}")
    

    f.suptitle(f"generation {gen}")
    
    # f.legend()
    # Show plot
    # f.grid(True)
    plt.savefig(f'generation{gen}_mse.png', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":


    test = MolSimCMA(0.15, 10, "TEMPLATE_HILLS")
    test.CMA()

    # print(test.evaluate2(np.array([[2, 3], [3, 4]])))


    # plot_cvs("gen7-sample16-COLVAR", ["phi", "psi"])

    # for gen in range(8):
    #     plot_cvs_per_generation(gen, ["phi", "psi"])


    # test.create_template_hills()
    # x = np.zeros(100)
    # # test.update_hills(x)
    # print(test.run_simulation(x))
