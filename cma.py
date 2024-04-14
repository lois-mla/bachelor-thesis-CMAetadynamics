import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt

def evaluate(x):
    # run metadynamics simulation on these values & find the probability distribution
    # P = ...
    return np.sum(P * np.log(P))

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(50), sigma=1.3)

    generations = 50
    for generation in range(generations):
        solutions = []
        
        # pick ideal population size!
        for _ in range(optimizer.population_size):
            x = optimizer.ask()

            value = evaluate(x)

            # append solutions by the point and its value according to 
            # the evaluate function
            solutions.append((x, value))
            
            print(f"#{generation} {value} (x={x})")

        optimizer.tell(solutions)
