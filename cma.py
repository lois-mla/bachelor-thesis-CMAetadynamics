from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
# import openmmtools 
# from openmmplumed import PlumedForce
import numpy as np
from cmaes import CMA
from cmaes import SepCMA
import matplotlib.pyplot as plt
import os
import re
from bias import MolSim
from sklearn import preprocessing
import matplotlib.cm as cm
# import matplotlib.patches as patches
from matplotlib.patches import Circle
# from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 
from IPython import display 
import shutil
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import pickle


class MolSimCMA:
    def __init__(self, width, number_of_gaussians, nsteps, cma_number_of_generations, cma_sigma, cma_upper_bound, cma_lower_bound=0):
        self.width = width
        self.number_of_gaussians = number_of_gaussians
        self.cma_number_of_generations = cma_number_of_generations
        self.cma_sigma = cma_sigma
        self.cma_upper_bound = cma_upper_bound
        self.cma_lower_bound = cma_lower_bound
        self.nsteps = nsteps

        subfolder = "output_eval"

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        self.output_path =  f"{subfolder}/cma_width{width}_n{number_of_gaussians}_S{cma_sigma}_B{cma_lower_bound}-{cma_upper_bound}_nsteps{nsteps}_1"

        self.template_hills_file = "TEMPLATE_HILLS"


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
            step_size = 2*np.pi / self.number_of_gaussians
            for phi in np.arange(-np.pi, np.pi, step_size):
                for psi in np.arange(-np.pi, np.pi, step_size):
                    hills.write(f"0 {phi} {psi} {self.width} {self.width} h{height_index} 1\n")

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
        for i in range(self.number_of_gaussians**2):
            pattern = rf'\bh{i}\b'
            content = re.sub(pattern, str(x[i]), content)

        # write to output_hills_file, specifying the generation & sample
        with open(f"{self.output_path}/HILLS/gen{gen}-sample{sample}-HILLS", "w") as hills:
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
        metad: METAD ARG=phi,psi PACE=500000000 HEIGHT=0 SIGMA=0.15,0.15 FILE={self.output_path}/HILLS/gen{gen}-sample{sample}-HILLS

        # monitor the two variables and the metadynamics bias potential
        PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE={self.output_path}/COLVAR/gen{gen}-sample{sample}-COLVAR
        """
        cvs = ["phi", "psi"]

        print("alanine-dipeptide-implicit.pdb", forcefield, cvs, self.nsteps, bias_script, gen, sample)

        # make MolSim object
        molsim = MolSim("alanine-dipeptide-implicit.pdb", forcefield, cvs, self.nsteps, bias_script, gen, sample)

        # run simulation
        molsim.run_sim(self.output_path)
 
        colvar_data = np.loadtxt(f'{self.output_path}/COLVAR/gen{gen}-sample{sample}-COLVAR')

        # get the data from the resulting COLVAR file
        phi = colvar_data[:, 1]
        psi = colvar_data[:, 2]
        # bias = molsim.colvar_data[:, -1]

        # use this data to make a probability histogram
        hist = np.histogram2d(phi, psi, bins=self.number_of_gaussians, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=None)
        # print("hist", hist)

        return hist[0]



    def evaluate(self, prob_hist):
        """
        evaluate the prob_hist based on the kl divergence, and return its absolute value
        """

        hist_no_zeros = prob_hist + 0.00001

        normalized_hist = hist_no_zeros / np.sum(hist_no_zeros)

        print(np.sum(normalized_hist))

        shape = normalized_hist.shape

        # print("normalized_hist", normalized_hist)
        num_bins = np.prod(shape)

        # goal value
        goal_value = 1/num_bins
        
        # calculate the kl divergence based on the provided probability histogram
        div_kl = np.sum(normalized_hist * np.log(normalized_hist/goal_value))

        # print("div_kl", div_kl)

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
        normalized_hist = prob_hist / np.sum(prob_hist)

        # calculate mean squared error
        mse = np.square(np.subtract(normalized_hist,goal_value)).mean()

        return mse


    def evaluate3(self, prob_hist):
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
        normalized_hist = prob_hist / np.sum(prob_hist)

        # calculate mean squared error
        mse = abs(np.subtract(normalized_hist,goal_value)).mean()

        return mse

    
    def CMA(self, start_from_pickle_gen=-1):
        """
        execute CMA-ES
        """
        # make output directory

        # if start_from_pickle_gen is set to -1, don't restart from pickle
        if start_from_pickle_gen == -1:
            run_n = 2

            while os.path.exists(self.output_path):
                self.output_path = self.output_path[:-2] + str(run_n)
                run_n += 1

            os.mkdir(self.output_path)
            os.mkdir(self.output_path + "/HILLS")
            os.mkdir(self.output_path + "/COLVAR")
            os.mkdir(self.output_path + "/plots")
            os.mkdir(self.output_path + "/animation")
            os.mkdir(self.output_path + "/free_energy")
            os.mkdir(self.output_path + "/pickled_files")
            os.mkdir(self.output_path + "/eval_values")

        # initialize optimizer
        if start_from_pickle_gen != -1:
            with open(f"{self.output_path}/pickled_files/optimizer{str(start_from_pickle_gen)}.pickled", 'rb') as f:
                optimizer = pickle.load(f)
        else:
            optimizer = SepCMA(mean=np.zeros(self.number_of_gaussians**2), sigma=self.cma_sigma, bounds=np.array([(0, self.cma_upper_bound)] * self.number_of_gaussians**2))

            # write header for file to store eval values
            with open((f"{self.output_path}/eval_values/evaluation_values.txt"), "w") as eval_file:
                # Write the header
                header = "Generation\\Sample" + "\t" + "\t".join(str(i) for i in range(optimizer.population_size)) + "\n"
                eval_file.write(header)


        generations = self.cma_number_of_generations

        # initialize eval_matrix to store all evaluation values in

        # eval_matrix = np.zeros((generations, optimizer.population_size))
        evaluation_values = []

        for generation in range(start_from_pickle_gen + 1, generations):
            solutions = []
            evaluation_values_per_gen = [] 
            
            for sample in range(optimizer.population_size):

                # ask optimizer for a sample
                x = optimizer.ask()

                # run the simulation on this sample and get the corresponding probability histogram
                prob_hist = self.run_simulation(x, generation, sample)

                # evaluate the prob_hist
                value = self.evaluate(prob_hist)

                # update eval_matrix with value
                # eval_matrix[generation, sample] = value

                calculate_free_energy(self.output_path, generation, sample)                

                # append solutions by both x and its evaluate value
                solutions.append((x, value))
                evaluation_values_per_gen.append(value)
                
                print(f"#{generation} {value} (x={x})")

            with open((f"{self.output_path}/eval_values/evaluation_values.txt"), "w") as eval_file:
                row = f"{generation}" + "\t" + "\t".join(f"{val:.4f}" for val in evaluation_values_per_gen) + "\n"
                eval_file.write(row)

            
            plot_cvs_and_heights(self, generation, ["phi", "psi"], optimizer.population_size, self.output_path + "/plots", solutions)

            # print(eval_matrix)

            # find the sample which has the best evaluate value
            # best_eval_sample = max(solutions, key=lambda item: item[1])[2]
            best_eval_sample = np.argmax(evaluation_values_per_gen)

            # plot the bias in a 2d heatmap plot for the best evaluated sample
            plot_bias_2d(self.output_path, generation, best_eval_sample)
            
            # plot_cvs_per_generation(generation, ["phi", "psi"], 50, solutions)
            # plot_bias3d(self.output_path, generation, 1)

            with open(f"{self.output_path}/pickled_files/optimizer{str(generation)}.pickled", "wb") as f: 
                pickle.dump(optimizer, f)

            evaluation_values.append(evaluation_values_per_gen)

            # tell the optimizer the solutions
            optimizer.tell(solutions)

        eval_matrix = np.array(evaluation_values)
        
        # contourplot_animation(self, 0, generations, optimizer.population_size, self.output_path + "/animation")
        plot_evaluate(eval_matrix, self.output_path + "/plots")


        # calculate_free_energy(self.output_path, generations, optimizer.population_size)

    
def plot_cvs_and_heights(cmaObj, gen, cvs, population_size, dir, solutions=None):

    plt.figure()  # Create a new figure

    colors = cm.rainbow(np.linspace(0, 1, population_size))

    for sample in range(population_size):

        colvar_data = np.loadtxt(f"{cmaObj.output_path}/COLVAR/gen{gen}-sample{sample}-COLVAR")
        hills_data = np.loadtxt(f"{cmaObj.output_path}/HILLS/gen{gen}-sample{sample}-HILLS")

        # for i, cv_label in enumerate(cvs):
        phi = colvar_data[:, 1]
        psi = colvar_data[:, 2]
        height = hills_data[:, -2]
        phi_hills = hills_data[:, 1]
        psi_hills = hills_data[:, 2]
        plt.scatter(phi, psi, s=1, marker='x', color=colors[sample])

        for i in range(len(height)):
            circle = Circle((phi_hills[i], psi_hills[i]), radius=height[i]/1000, fill=False, color='k', alpha=0.5, zorder=100)
            plt.gca().add_patch(circle)

 
        # patches.Circle((phi_hills, psi_hills), radius=height)

    # Adding labels and legend
    plt.xlabel("phi")
    plt.ylabel("psi")

    if solutions:
        plt.title(f"generation {gen}, evaluate mean = {np.mean([s[1] for s in solutions])}")

    plt.savefig(f'{dir}/generation{gen}.png', bbox_inches='tight')

    plt.close()  # Close the current figure to prevent accumulation



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

    return [im]


def contourplot_animation(cmaObj, first_gen, last_gen, pop_size, dir=""):
    
    heights_list =  []
    for gen in range(first_gen, last_gen):
        heights_per_sample = []

        for sample in range(pop_size):
            hills_data = np.loadtxt(f"{cmaObj.output_path}/HILLS/gen{gen}-sample{sample}-HILLS")
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
        plot = contourplot(heights_list[frame], phi, psi, fig, ax, gen, first_cycle) # Use heights for the current frame
        first_cycle = False

        return plot


    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=last_gen-first_gen, interval=500, blit=True, repeat=False)
    # fig.colorbar(fig, ax=ax, label='Height')    

    # Show animation
    writervideo = animation.FFMpegWriter(fps=5) 
    ani.save(f'{dir}/animation.mp4', writer=writervideo)
    plt.close() 



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
    # plt.show()
    plt.close()  # Close the current figure to prevent accumulation



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
    plt.close()  # Close the current figure to prevent accumulation



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
    plt.close()  # Close the current figure to prevent accumulation



def run_plumed_command(command):
    try:
        # Execute the PLUMED command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        # Check if the command executed successfully
        if process.returncode == 0:
            print("PLUMED command executed successfully.")
            # Process the output if needed
            print("Output:", output.decode())
        else:
            print("Error executing PLUMED command:")
            print(error.decode())
    except Exception as e:
        print("An error occurred:", e)


def calculate_free_energy(path, gen, sample):
    run_plumed_command(f"plumed sum_hills --hills {path}/HILLS/gen{gen}-sample{sample}-HILLS --outfile {path}/free_energy/fes_gen{gen}_{sample}.dat")

def calculate_free_energy_phi(path, gen, pop_size):
    if not os.path.exists(path + "/free_energy"):
        os.mkdir(path + "/free_energy")

    for sample in range(pop_size):
        run_plumed_command(f"plumed sum_hills --hills {path}/HILLS/gen{gen}-sample{sample}-HILLS --idw phi --kt 2.5 --stride 500 --mintozero --outfile {path}/free_energy/fes{sample}_phi.dat")


def plot_free_energy_3d(path, pop_size):
    # plot free energy as a function of simulation time

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for sample in range(pop_size):   
        # import fes file into pandas dataset
        data = np.loadtxt(f"{path}/fes{sample}.dat")

        phi = data[:, 0]
        psi = data[:, 1]
        free_energy = data[:, 2]  

        # Plot the data points
        ax.scatter(phi, psi, free_energy, c=free_energy, cmap='viridis', marker='o')

        # # plot fes
        # plt.plot(phi, free_energy, label=f"Sample {sample}") 

    # Set labels and title
    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_zlabel('Free Energy')
    ax.set_title('3D Plot of Phi, Psi, and Free Energy')
    # # labels
    # plt.xlabel("phi [rad]")
    # plt.ylabel("free energy [kJ/mol]")
    # plt.legend(ncol=3)

    plt.savefig(f'{path}/free_energy_phi.png', bbox_inches='tight')
    plt.close()  # Close the current figure to prevent accumulation



def plot_bias3d(path, gen, sample):
      # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # import fes file into pandas dataset
    data = np.loadtxt(f"{path}/free_energy/fes_gen{gen}_{sample}.dat")

    phi = data[:, 0]
    psi = data[:, 1]
    bias = data[:, 2] * -1

    # Plot the data points
    ax.scatter(phi, psi, bias, c=bias, cmap='viridis', marker='o')

    # Set labels and title
    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_zlabel('Bias')
    ax.set_title('3D Plot of Phi, Psi, and Bias')

    # Function to update the view angle
    def update_view_angle(elev, azim):
        ax.view_init(elev=elev, azim=azim)

    # Create interactive sliders for elevation and azimuth
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_azim = plt.axes([0.25, 0.15, 0.65, 0.03])
    s_elev = plt.Slider(ax_elev, 'Elevation', -180, 180, valinit=30)
    s_azim = plt.Slider(ax_azim, 'Azimuth', -180, 180, valinit=30)

    # Define function to update plot when sliders change
    def update(val):
        elev = s_elev.val
        azim = s_azim.val
        update_view_angle(elev, azim)

    # Register the update function with the sliders
    s_elev.on_changed(update)
    s_azim.on_changed(update)
    plt.show()

    # plt.savefig(f'free_energy_3d_compare.png', bbox_inches='tight')
    plt.close()  # Close the current figure to prevent accumulation



def plot_bias_2d(path, gen, sample):
    # import fes file 
    data = np.loadtxt(f"{path}/free_energy/fes_gen{gen}_{sample}.dat")

    phi = data[:, 0]
    psi = data[:, 1]
    bias = data[:, 2] * -1

    # Create a 2D contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(phi, psi, bias, levels=100, cmap='viridis')
    
    # Add a color bar
    plt.colorbar(contour, label='Bias')
    
    # Set labels and title
    plt.xlabel('Phi')
    plt.ylabel('Psi')
    plt.title('2D Heatmap of Phi, Psi, and Bias')

    plt.savefig(f'{path}/plots/bias_gen{gen}_sample{sample}.png', bbox_inches='tight')
    plt.close()  # Close the current figure to prevent accumulation

    
    # plt.show()


def plot_free_energy_2d(path, pop_size):

    for sample in range(pop_size):

        # import fes file into pandas dataset
        data = np.loadtxt(f"{path}/fes{sample}_phi.dat0.dat")

        phi = data[:, 0]
        free_energy = data[:, 1]  

        # plot fes
        plt.plot(phi, free_energy) 

    # labels
    plt.xlabel("phi [rad]")
    plt.ylabel("free energy [kJ/mol]")
    # plt.legend(ncol=3)

    plt.savefig(f'{path}/free_energy2d.png', bbox_inches='tight')
    plt.close()  # Close the current figure to prevent accumulation


# OUD
def plot_bias(colvar_file, save_path):

    colvar_data = np.loadtxt(colvar_file)

    # for i, cv_label in enumerate(cvs):
    phi = colvar_data[:, 1]
    psi = colvar_data[:, 2]
    bias = colvar_data[:, 3]

    # Create a 2D histogram
    bins = 50
    heatmap, xedges, yedges = np.histogram2d(phi, psi, bins=bins, weights=bias)
    print(heatmap)

    # ax.figure(figsize=(10, 8))
    im = plt.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.xlabel('Phi')
    plt.ylabel('Psi')
        # Plot colorbar only for the first cycle
    plt.colorbar(im, label='bias')
    plt.savefig(save_path)

    plt.close()  # Close the current figure to prevent accumulation



def plot_evaluate(eval_matrix, path):
    """
    Plots each the evaluation values for each sample over the generations

    Parameters:
    eval_matrix (numpy.ndarray): A 2D numpy array of shape (N, M).
    """
    # Number of rows (N) and columns (M)
    N, M = eval_matrix.shape
    generation = np.arange(N)
    
    # Create a scatter plot for each row in the matrix
    plt.figure(figsize=(10, 6))
    
    for i in range(M):
        plt.scatter(generation, eval_matrix[:, i], label=f'sample {i+1}')

    avg_eval = np.mean(eval_matrix, axis=1)
    
    plt.scatter(generation, avg_eval, label=f'avg')

    plt.title('Eval values')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    # plt.legend()
    plt.grid(True)

    plt.savefig(f'{path}/evaluate_values')

    plt.close()  # Close the current figure to prevent accumulation



if __name__ == "__main__":

    number_of_gaussians = 20
    cma_number_of_generations = 20
    time_steps = 100000

    cma_upper_bound = 10
    # cma_lower_bound = 

    # calculate the distance between the gaussians
    d = (2 * np.pi) / number_of_gaussians

    # set width 
    # width = round(1.5 * d*(1/(np.sqrt(8 * np.log(2)))), 3)
    width = 0.25

    # set cma_sigma to be 20% of the upper bound
    cma_sigma = 0.2 * cma_upper_bound

    test = MolSimCMA(width, number_of_gaussians, time_steps, cma_number_of_generations, cma_sigma, cma_upper_bound)
    test.CMA()

    # plot_bias_2d("correct_output/cma_width0.25_n20_S2.0_B0-10_nsteps100000new_run", 19, 3)


    # plot_bias3d("correct_output/cma_width0.25_n20_S2.0_B0-10_nsteps100000new_run", 19, 3)
    # plot_bias3d("correct_output/cma_width0.3_n20_S2.0_B0-10_nsteps100000new_run", 19, 15)


    # plot_bias("output/cma_width0.628_n10_gens200_S2.0_B0-10_nsteps500000/COLVAR/gen23-sample2-COLVAR", "test")

    # run_plumed_command("plumed sum_hills --hills output/cma_width1.3_n10_gens100_S7_B0-40/HILLS/gen50-sample1-HILLS ")
    # calculate_free_energy_phi("output/cma_width0.628_n10_gens50_S2.0_B0-10", 49, 17)
    # plot_free_energy_2d("output/cma_width0.628_n10_gens50_S2.0_B0-10/free_energy", 17)


# P=CMA population size,
# S=CMA sigma,
# B=(min,max)=CMA bounds
# width=METAD gaussian sigma, 
# height=METAD gaussian height,
# n=METAD number of gaussians in 1 dimension, 
    # contourplot_animation(1, 21, 26, "plots_sigma0.7_N40_bounds0-30")
