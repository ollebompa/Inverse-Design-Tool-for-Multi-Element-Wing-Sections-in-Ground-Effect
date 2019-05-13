# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:59:48 2019

@author: ollen
"""
import numpy as np


from Aerofoil_Generation import NACA4series
from Analysis_Method import LOAD, PANEL, ASOLVE
from Post import CP, PLOT_CP
from Post import FIELD_U_V_CP, STREAMLINES
from Post import FIELD_PLOT_CP_U_V,FIELD_PLOT_STREAMLINES


thickness=0.12
M=120
c=1
alpha=15
GE=False
h=0.0
save=True
savename = 'NACA0012Example'


def Analyse_NACA(thickness,M,c,alpha,GE,h):
    XP,ZP=NACA4series(thickness,M=120,c=1,plot=True)
    GAMMA,PSI,CL=ASOLVE(XP,ZP,alpha,UINF=1,GE=GE,h=0.2)
    CPX=CP(GAMMA,XP)
    PLOT_CP(CPX,XP,ZP,GE=GE,save=save,savename=savename)
    
    
filepath='C:\\Users\\ollen\\Documents\\Aeronautics and Astronautics part 3\\IP\\Test cases\\' 
filename= 'Inverted_NRL _071_coordinates.txt'   
M=120
alpha=0
GE=True
h=0.2
save=True
savename = 'txtfileExample'

    
def Analyse_txtfile(filepath,filename,M,alpha,GE,h):
    XC,ZC=LOAD(filepath,filename,plot=False)
    XP,ZP=PANEL(XC,ZC,M=M,plot=False)
    GAMMA,PSI,CL=ASOLVE(XP,ZP,alpha,UINF=1,GE=GE,h=0.2)
    CPX=CP(GAMMA,XP)
    PLOT_CP(CPX,XP,ZP,GE=GE,save=save,savename=savename)
    
    

xlimit=[-0.5,1.8]
zlimit=[-0.1,1.0]
NX=40
NZ=40   
    
    
def Plot_Field(xlimit,zlimit,NX,NZ,filepath,filename,M,alpha,GE,h):
    XC,ZC=LOAD(filepath,filename,plot=False)
    XP,ZP=PANEL(XC,ZC,M=M,plot=False)
    GAMMA,PSI,CL=ASOLVE(XP,ZP,alpha,UINF=1,GE=GE,h=0.2)
    
    FIELD_PSI=np.array((0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0))
    
    PSIFIELD=np.concatenate((np.ravel(PSI),FIELD_PSI))
    
    #Calculate velocity and Cp field
    (U, W, VM, CPF, XG, ZG)=FIELD_U_V_CP(xlimit,zlimit,NX,NZ,GAMMA,XP,ZP,alpha,h)
    #plot velocity and Cp field
    FIELD_PLOT_CP_U_V(U, W, VM, CPF, XG, ZG, XP, ZP,zlimit, h=h,mirror=False)
    #Calculate Stramlines,    
    XZPSI=STREAMLINES(xlimit,zlimit,NX,NZ,GAMMA,XP,ZP,alpha,PSIFIELD,h)
    #plot 
    FIELD_PLOT_STREAMLINES(XZPSI, XP, ZP,zlimit, h=h,mirror=False,savename='test',save=True)
