########################################################################################  
# This script perform a mechanism reduction using the DRG method                       #  
# Author: Felipe C. Minuzzi							       #  
# Date: 30/04/2018								       #  
# Last Version: 16/05/2020                                                             #                                     
#										       # 
########################################################################################
from __future__ import division
from __future__ import print_function
import cantera as ct
import numpy as np
from timeit import default_timer
import ast
import main
import time
import csv
import pandas as pd
print('Runnning Cantera version: ' + ct.__version__)

print('########################################################################################')
print('#                                                                                      #')
print('#         Skeletal Mechanism Reduction based on DRG (directed relation graph)          #')
print('#                                                                                      #')
print('########################################################################################')

########################################################################################
inputfile=open('./input','r')
c = inputfile.readlines()
inputfile.close
########################################################################################
mechanism                       = c[1].strip()
gas                             = ct.Solution(mechanism)
rea                             = gas.reactions							
nsp                             = gas.n_species						
nrx                             = gas.n_reactions				    
npoints                         = int(c[3])
npoints2                        = int(c[5])
npoints3                        = int(c[7])
target                          = c[23].strip()
targets                         = ast.literal_eval(target)
epsilon                         = float(c[9])
pre0                            = float(c[11])
tin0                            = float(c[13])
phi0                            = float(c[15])
pref                            = float(c[17])
tinf                            = float(c[19])
phif                            = float(c[21])
npoints4                        = int(c[27])
nj                              = int(c[29])
ep0                             = float(c[31])
ep1                             = float(c[33])
app                             = int(c[35])
data                            = []
datay                           = []
fuel_species                    = c[25].strip()

print('Number of species in the detailed mechanism is: {} and reactions is: {}'.format(nsp,nrx))

t1 = default_timer()
if app==1:
 main.reactor(gas,nsp,fuel_species,tin0,pre0,tinf,npoints,npoints3,phi0,phif,npoints2,pref,data,datay)
else:
# main.flamespeed(gas,nsp,fuel_species,tin0,pre0,npoints3,phi0,phif)
 from pandas.core.common import flatten
 data2 = []
 datay2= []
 stoich_O2                       = gas.n_atoms(fuel_species,'C') + 0.25*gas.n_atoms(fuel_species,'H') \
                                        - 0.5*gas.n_atoms(fuel_species,'O')
 air_N2_O2_ratio                 = 3.76
 ifuel                           = gas.species_index(fuel_species)
 io2                             = gas.species_index('o2')
 in2                             = gas.species_index('n2')

 for j in range(npoints3):
  pre             = ct.one_atm
  phi             = phi0 +((phif-phi0)/npoints3)*j

  x               = np.zeros(nsp,'d')
  x[io2]          = stoich_O2
  x[in2]          = stoich_O2*air_N2_O2_ratio
  x[ifuel]        = phi
  gas.TPX         = 298, pre, x
  initial_grid    = 2*np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03],'d')  
  f               = ct.FreeFlame(gas, initial_grid)
  f.inlet.X       = x
  f.inlet.T       = 298
  f.energy_enabled= False
  tol_ss          = [1.0e-5, 1.0e-9]
  tol_ts          = [1.0e-5, 1.0e-9]
  f.flame.set_steady_tolerances(default=tol_ss)
  f.flame.set_transient_tolerances(default=tol_ts)
  f.set_max_jac_age(50, 50)
  f.set_time_step(1.0e-5, [2, 5, 10, 20, 80])
  f.set_refine_criteria(ratio = 10.0, slope = 1, curve = 1)
  loglevel        = 1
  refine_grid     = False
  f.solve(loglevel, refine_grid)
  f.save('adiabatic.xml','no_energy','solution without the energy equation')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 10.0, slope = 1, curve = 1)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 9.0, slope = 0.9, curve = 0.9)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 7.0, slope = 0.7, curve = 0.7)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 5.0, slope = .5, curve = .5)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 3.75, slope = 0.375, curve = 0.375)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  f.energy_enabled= True
  f.set_refine_criteria(ratio = 3.355, slope = 0.355, curve = 0.355)
  f.solve(loglevel=1, refine_grid=True)
  f.save('adiabatic.xml','energy','solution with the energy equation enabled')
  temper  = list(f.T)
  data2.append(temper)
  thermos = list(f.Y)
  datay2.append(thermos)
 data  = list(flatten(data2))
 datay = list(flatten(datay2))
t2 = default_timer()

tam                             = len(data)
tempe                           = np.zeros(tam,'d')
tempe                           = np.array(data)
mfrac                           = np.zeros((tam,nsp))
if app==1:
 mfrac                          = np.array(datay)
else:
 for i in range(tam):
  mfrac[i,0:]                   = datay[i]
datas                           = np.zeros((tam,gas.n_species))
for u in range(tam):
   datas[u,0:] = mfrac[u,0:]   

sn                              = int(npoints4/(nsp+1))
sample                          =np.zeros((sn,nsp+1))
sample[:,0]                     =np.random.choice(tempe,sn)
for k in range(nsp):
 sample[:,k+1] = np.random.choice(datas[:,k],sn)

variatio                        = np.zeros(nj,'d')
for k in range(nj):
 variatio[k] = ep0+k*ep1

t3 = default_timer()
for j in range(nj):
    species                     = []
    species2                    = []
    speciessk                   = []
    epsilon                     = variatio[j]
    main.coefficient(mechanism,gas,sn,sample,targets,species,nrx,species2,speciessk,epsilon,pre0)
########################################################################################
#                            Defining skeletal mechanism       	                       #
########################################################################################
    species.append('N2')	
    species2                     = list(set(species))
    print(species2)
    speciessk                    = [gas.species(name) for name in species2]
    all_reactions                = ct.Reaction.listFromFile(mechanism)
    reactionssk                  = []
    for P in all_reactions:
     if not all(reactant in species2 for reactant in P.reactants):
      continue
     if not all(product in species2 for product in P.products):
      continue
     reactionssk.append(P)

    skel                         = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', \
                                      species=speciessk,reactions=reactionssk)
    nsp1                         = skel.n_species
    nrx1                         = skel.n_reactions
    print('For threshold: {} the number of species in the skeletal mechanism is: {} and reactions is: {}'.format(epsilon,nsp1,nrx1))
    main.write_cti(skel,fuel_species,nsp1,epsilon)
    main.write_inp(skel,fuel_species,nsp1,epsilon)

t4 = default_timer()

print('########################################################################################')
print('#                                                                                      #')
print('#                            Reduction successfully finished!                          #')
print('#                                                                                      #')
print('########################################################################################')

print('Time for application simulation: {} seconds'.format(t2-t1))
print('Time for reduction (all epsilon variation): {} seconds'.format(t4-t3))
print('Time for each epsilon reduction (average): {} seconds'.format((t4-t3)/nj))
