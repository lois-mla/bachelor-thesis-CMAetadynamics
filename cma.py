import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt



class MolSimCMA:
    def __innit__(self, default_sigma, resolution):
        self.default_sigma = default_sigma
        self.resolution = resolution 
    
    def create_hills_file(self):
        with open ("HILLS", "x") as hills:
            # f.write("#! FIELDS time phi psi sigma_phi sigma_psi height biasf\n
                #! SET multivariate false\n
                #! SET kerneltype stretched-gaussian\n
                #! SET min_phi -pi\n
                #! SET max_phi pi\n
                #! SET min_psi -pi\n
                #! SET max_psi pi")

            # need to generalize this later on!
            for phi in range(-np.pi, np.pi, self.resolution):
                for psi in range(-np.pi, np.pi, self.resolution):
                    hills.write(f"")
                

    def run_simulation(x):
        """
        
        """
        # write x to HILLS file 

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
   with open("HILLS") as file:
       file