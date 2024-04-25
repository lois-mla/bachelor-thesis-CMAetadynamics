from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce
import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt
import os
import re
from bias import MolSim
from sklearn import preprocessing
import matplotlib.cm as cm
# import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

class MolSimCMA:
    def __init__(self, default_sigma, resolution, template_hills_file):
        self.default_sigma = default_sigma
        self.resolution = resolution
        self.template_hills_file = template_hills_file

        # create template hills file
        self.create_template_hills()
    

    # ! generalize later on for other cvs
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
        for i in range(self.resolution**2):
            pattern = rf'\bh{i}\b'
            content = re.sub(pattern, str(x[i]), content)

        # write to output_hills_file, specifying the generation & sample
        with open(f"gen{gen}-sample{sample}-HILLS", "w") as hills:
            hills.write(content)


    # ! generalize later on for other cvs
    def run_simulation(self, x, gen, sample):
        """
        run the MD simulation, using a bias consisiting of SoG of the Gaussians described in the
        HILLS file with heights of vector x. Return a histogram containing the measured values 
        os phi and psi
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
        # bias = molsim.colvar_data[:, -1]

        print("phi", phi)
        print("psi", psi)
        # print("bias", bias)

        # use this data to make a probability histogram
        hist = np.histogram2d(phi, psi, bins=self.resolution, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=None)
        print("hist", hist)

        return hist[0]


    def evaluate(self, prob_hist):
        """
        evaluate the prob_hist based on the kl divergence, and return its absolute value
        """
 
        hist_no_zeros = prob_hist + 1

        normalized_hist = preprocessing.normalize(hist_no_zeros)

        print("normalized_hist", normalized_hist)
        
        # calculate the kl divergence based on the provided probability histogram
        div_kl = np.sum(normalized_hist * np.log2(normalized_hist))

        print("div_kl", div_kl)

        return abs(div_kl)
    

    def evaluate2(self, prob_hist):
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

    
    def CMA(self, folder, max_bound, sigma):
        """
        execute CMA-ES
        """
        # initialize optimizer
        optimizer = CMA(mean=np.zeros(self.resolution**2), sigma=sigma, bounds=np.array([(0, max_bound)] * self.resolution**2))

        generations = 50
        for generation in range(generations):
            solutions = []
            
            for sample in range(optimizer.population_size):
            # for sample in range(50):

                # Check if files already exist and delete them if they do
                if os.path.exists(f'gen{generation}-sample{sample}-COLVAR'):
                    os.remove(f'gen{generation}-sample{sample}-COLVAR')
                if os.path.exists(f'gen{generation}-sample{sample}-HILLS'):
                    os.remove(f'gen{generation}-sample{sample}-HILLS')

                # ask optimizer for a sample
                x = optimizer.ask()

                # run the simulation on this sample and get the corresponding probability histogram
                prob_hist = self.run_simulation(x, generation, sample)

                # evaluate the prob_hist with kl divergence
                value = self.evaluate2(prob_hist)

                # append solutions by both x and its kl div value
                solutions.append((x, value))
                
                print(f"#{generation} {value} (x={x})")

            plot_cvs_and_heights(generation, ["phi", "psi"], optimizer.population_size, solutions, folder)

            # plot_cvs_per_generation(generation, ["phi", "psi"], 50, solutions)

            # tell the optimizer the solutions
            optimizer.tell(solutions)
        
        contourplot_animation(0, generations, optimizer.population_size, folder)


def plot_cvs(colvar_path, cvs):
    """
    plot cvs against time
    """

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


def plot_cvs_per_generation(gen, cvs, population_size, solutions=None):
    """
    plot all cvs in a given generation
    """
    
    f, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 12), tight_layout=True)

    for sample in range(population_size):

        colvar_data = np.loadtxt(f"gen{gen}-sample{sample}-COLVAR")

        # time = colvar_data[:, 0]

        ax = axes.flatten()[sample]
        
        # for i, cv_label in enumerate(cvs):
        cv0 = colvar_data[:, 1]
        cv1 = colvar_data[:, 2]
        ax.scatter(cv0, cv1, s=1, marker='x')

        # Adding labels and legend
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        
        if solutions:
            ax.set_title(f"sample {sample}, kl_div = {solutions[sample][1]}")
        else:
            ax.set_title(f"sample {sample}")
    

    f.suptitle(f"generation {gen}")

    plt.savefig(f'images_cvs_against_each_other/generation{gen}.png', bbox_inches='tight')


def plot_cvs_per_generation_1plot(gen, cvs, population_size, solutions=None):

    colors = cm.rainbow(np.linspace(0, 1, population_size))

    for sample in range(population_size):

        colvar_data = np.loadtxt(f"gen{gen}-sample{sample}-COLVAR")

        # for i, cv_label in enumerate(cvs):
        phi = colvar_data[:, 1]
        psi = colvar_data[:, 2]
        plt.scatter(phi, psi, s=1, marker='x', color=colors[sample])

    # Adding labels and legend
    plt.xlabel("psi")
    plt.ylabel("phi")

    if solutions:
        plt.title(f"evaluate = {np.mean([s[1] for s in solutions])}")

    plt.savefig(f'plots_all_samples_in_1_plot/generation{gen}.png', bbox_inches='tight')



def plot_cvs_and_heights(gen, cvs, population_size, folder, solutions=None):

    colors = cm.rainbow(np.linspace(0, 1, population_size))

    for sample in range(population_size):

        colvar_data = np.loadtxt(f"gen{gen}-sample{sample}-COLVAR")
        hills_data = np.loadtxt(f"gen{gen}-sample{sample}-HILLS")

        # for i, cv_label in enumerate(cvs):
        phi = colvar_data[:, 1]
        psi = colvar_data[:, 2]
        height = hills_data[:, -2]
        phi_hills = hills_data[:, 1]
        psi_hills = hills_data[:, 2]
        plt.scatter(phi, psi, s=1, marker='x', color=colors[sample])

        for i in range(len(height)):
            circle = Circle((phi_hills[i], psi_hills[i]), radius=height[i]/750, fill=False, color='k', alpha=0.5, zorder=100)
            plt.gca().add_patch(circle)


        # patches.Circle((phi_hills, psi_hills), radius=height)

    # Adding labels and legend
    plt.xlabel("phi")
    plt.ylabel("psi")

    if solutions:
        plt.title(f"generation {gen}, evaluate mean = {np.mean([s[1] for s in solutions])}")

    plt.savefig(f'{folder}/generation{gen}.png', bbox_inches='tight')


def contourplot(heights, phi, psi, fig, ax, generation, first_cycle):

    # Create a 2D histogram
    bins = 40
    heatmap, xedges, yedges = np.histogram2d(phi, psi, bins=bins, weights=heights)
    print(heatmap)

    # ax.figure(figsize=(10, 8))
    im = ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_title(f'generation {generation}')
        # Plot colorbar only for the first cycle
    if first_cycle:
        fig.colorbar(im, ax=ax, label='Mean Height of all samples')
    # plt.show()


def contourplot_animation(first_gen, last_gen, pop_size, folder=""):
    
    heights_list =  []
    for gen in range(first_gen, last_gen):
        heights_per_sample = []

        for sample in range(pop_size):
            hills_data = np.loadtxt(f"gen{gen}-sample{sample}-HILLS")
            height = hills_data[:, -2]
            heights_per_sample.append(height)

        phi = hills_data[:, 1]
        psi = hills_data[:, 2]

        heights_per_sample_arr = np.array(heights_per_sample)

        heights_list.append(np.mean(heights_per_sample_arr, axis=0))

    
    # Create figure and axis
    fig, ax = plt.subplots()

    # Function to update plot for each frame of animation
    first_cycle = True
    def update(frame):
        nonlocal first_cycle
        ax.clear()
        gen = first_gen + frame
        contourplot(heights_list[frame], phi, psi, fig, ax, gen, first_cycle) # Use heights for the current frame
        first_cycle = False


    # Create animation
    ani = FuncAnimation(fig, update, frames=last_gen-first_gen, interval=500, repeat=False)
    # fig.colorbar(fig, ax=ax, label='Height')    

    # Show animation
    plt.show()
    # plt.savefig(f'{folder}/animation.png', bbox_inches='tight')


if __name__ == "__main__":

    # test = MolSimCMA(0.15, 10, "TEMPLATE_HILLS")
    test = MolSimCMA(0.8, 50, "TEMPLATE_HILLS")
    test.CMA("plots_s7_B(0,40)_width0.8_n50", 40, 7)

# P=CMA population size,
# S=CMA sigma,
# B=(min,max)=CMA bounds
# width=METAD gaussian sigma, 
# height=METAD gaussian height,
# n=METAD number of gaussians in 1 dimension, 
