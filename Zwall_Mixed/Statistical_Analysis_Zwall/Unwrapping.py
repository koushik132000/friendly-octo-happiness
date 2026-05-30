##Unwrap coordinates of a chain in a box with periodic boundary conditions
#The analysis is for a code with periodic boundary conditions
#Box size is 7x7x20
#Requires modified bead coordinate file for 36 chains
import math
import pandas as pd
import os
from string import *
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Identifies current working directory
cwd = os.getcwd()
print("Working in", cwd)

ipname = 'echains'
opname = 'unwrapc'

xmax, ymax, zmax = 7, 7, 20
nb = 20
nchains = 36

for i in range(1, 1001):
    infname = ipname +'.'+ str(rank) +'.'+str(i)+'.'+ 'csv'
    pos = pd.read_csv(infname)

    xs = pos[['x']]
    ys = pos[['y']]
    zs = pos[['z']]
    # creating a file
    opfname=opname+'.'+str(rank)+'.'+str(i)+'.'+'csv'
    f = open(opfname, "w")
    f.write("x,y,z\n")

    for c in range(0, nchains):
        f.write(f"{xs.at[20*c,'x']},{ys.at[20*c,'y']},{zs.at[20*c,'z']}\n")
        x_unw = xs.at[20*c, 'x']
        y_unw = ys.at[20*c, 'y']
        z_unw = zs.at[20*c, 'z']

        for k in range(1, nb):
            dx = xs.at[20*c + k, 'x'] - xs.at[20*c + k-1, 'x'] # index labelling
            dy = ys.at[20*c + k, 'y'] - ys.at[20*c + k-1, 'y']
            dz = zs.at[20*c + k, 'z'] - zs.at[20*c + k-1, 'z']

            # for x 
            if (dx > 0.5 * xmax):
                dx -= xmax
            elif (dx < -0.5 * xmax):
                dx += xmax
            #for y 
            if (dy > 0.5 * ymax):
                dy -= ymax
            elif (dy < -0.5 * ymax):
                dy += ymax

            x_unw += dx
            y_unw += dy
            z_unw += dz

             # writing the result to the file 
            f.write(f"{x_unw},{y_unw},{z_unw}\n")
    print("file is ready")
