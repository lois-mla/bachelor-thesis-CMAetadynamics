import numpy as np
from cmaes import CMA
import os
import time
import sys
from skopt import Optimizer
import pickle
from scipy import integrate

def lammps(T,X,eSS,sSS,eTS,sTS,g,s):
    os.system("cp lammps/* generation"+str(g)+"/sample"+str(sample)+"/")
    os.system("sed -i 's|STR_T|"+str(T)+"|' generation"+str(g)+"/sample"+str(sample)+"/input")
    os.system("sed -i 's|STR_X|"+str(X)+"|' generation"+str(g)+"/sample"+str(sample)+"/input")
    os.system("sed -i 's|STR_eSS|"+str(eSS)+"|' generation"+str(g)+"/sample"+str(sample)+"/ts.sw")
    os.system("sed -i 's|STR_sSS|"+str(sSS)+"|' generation"+str(g)+"/sample"+str(sample)+"/ts.sw")
    os.system("sed -i 's|STR_eTS|"+str(eTS)+"|' generation"+str(g)+"/sample"+str(sample)+"/ts.sw")
    os.system("sed -i 's|STR_sTS|"+str(sTS)+"|' generation"+str(g)+"/sample"+str(sample)+"/ts.sw")
    os.system("sed -i 's|STR_KAPPA|"+str(T*0.000086*0.5)+"|' generation"+str(g)+"/sample"+str(sample)+"/plumed.dat")
    os.system("cd generation"+str(g)+"/sample"+str(sample)+" && sbatch bsub_lammps.job")
    return

def predict(g,s):
    #read COLVAR files
    cvs = []
    cvs.append(np.loadtxt('generation'+str(g)+'/sample'+str(s)+'/COLVAR'))
    cvs = np.asarray(cvs)[0,:,2]
    prob = -1*np.mean(cvs[-5:-1])/1056
    return prob

def denormalize(x):
    p = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    p[0] = 270 + 100*x[0]
    p[1] = 0.5 + 0.45*x[1]
    p[2] = 0.01 + 0.04*x[2]
    p[3] = 3.0 + 3.6*x[3]
    p[4] = 0.01 + 0.04*x[4]
    p[5] = 3.0 + 3.6*x[5]
    return p

if __name__ == "__main__":
    #T,X,eSS,sSS,eTS,sTS
    #optimizer = CMA(population_size=8, mean=np.array([0.5,0.5,0.5,0.5,0.5,0.5]), sigma=0.16, bounds=np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]) )
    opt_file = open('optimizer19.pickled','rb')
    optimizer = pickle.load(opt_file)

    for generation in range(20,50):
        xs = []
        ps = []
        solutions = []
        os.system("mkdir generation"+str(generation))
        for sample in range(optimizer.population_size):
            x = optimizer.ask()
            xs.append(x)
            p = denormalize(x)
            ps.append(p)
            os.system("echo '"+f"# {generation} {ps[sample][0]} {ps[sample][1]} {ps[sample][2]} {ps[sample][3]} {ps[sample][4]} {ps[sample][5]} '")
            os.system("mkdir generation"+str(generation)+"/sample"+str(sample))
            lammps(p[0],p[1],p[2],p[3],p[4],p[5],generation,sample)
        done=False
        while (not done):
            checks = [False]*optimizer.population_size
            time.sleep(1)
            for sample in range(optimizer.population_size):
                checks[sample]=os.path.exists("generation"+str(generation)+"/sample"+str(sample)+"/flag.dat")
                done = all(checks)
        for sample in range(optimizer.population_size):
            value = predict(generation,sample)
            solutions.append((xs[sample], value))
            os.system("echo '"+f"{generation} {value} {ps[sample][0]} {ps[sample][1]} {ps[sample][2]} {ps[sample][3]} {ps[sample][4]} {ps[sample][5]} '")
        optimizer.tell(solutions)
        with open("optimizer"+str(generation)+".pickled", "wb") as f:
            pickle.dump(optimizer, f)