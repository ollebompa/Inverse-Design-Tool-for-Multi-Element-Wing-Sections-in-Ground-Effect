# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:20:43 2019

@author: ollen
"""
import numpy as np
import math as m
import scipy as sci

from scipy.interpolate import interp1d

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (Locator,FixedLocator,MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sci.ma.masked_array(sci.interp(value, x, y))

      
  
'''
Calculate CP
'''
def CP(GAMMA,XP):
      K = len(XP)
      CP = [0 for k in range(K)]
      
      for k in range(K):
            CP[k] = 1-GAMMA[k]**2
            
      return(CP)


'''
Plot CP
'''

def PLOT_CP(CPX,XP,ZP,GE=False,save=True,savename='test'):
      fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(7,7))   
      for k in range(len(XP)):
          ax[0].plot(XP[k],CPX[k],color='#1f77b4',linestyle='--',marker='None')
          ax[1].plot(XP[k], ZP[k], color='k', linewidth=1.5)
            
      
      fig.tight_layout(h_pad=-12,w_pad=0) 
      
      #Set limits
      cprange = np.max(CPX)-np.min(CPX)
      
      
     #Set Ticks
     
      multiple = np.around(cprange*0.2,0)
      majorLocatory = MultipleLocator(multiple)
      majorFormattery = FormatStrFormatter('%.1f')
      ax[0].yaxis.set_major_locator(majorLocatory)
      ax[0].yaxis.set_major_formatter(majorFormattery)
      
      majorLocatorx = FixedLocator([0.5,1.0])
      majorFormatterx = FormatStrFormatter('%.1f')
      minorLocatorx = MultipleLocator(0.1)
      ax[0].xaxis.set_major_locator(majorLocatorx)
      ax[0].xaxis.set_major_formatter(majorFormatterx)
      ax[0].xaxis.set_minor_locator(minorLocatorx)
      
      ax[0].xaxis.set_tick_params(bottom='True', top='False', direction='inout')
      ax[0].yaxis.set_tick_params(left='True', right='False', direction='inout')
      
      if(GE==False):
          ax[0].invert_yaxis()
          
      ax[0].spines['top'].set_visible(False)
      ax[0].spines['right'].set_visible(False)
      ax[0].spines['bottom'].set_position('zero')
      ax[0].set_ylabel(r"\textbf{C\textsubscript{P}}",rotation='horizontal')
      ax[0].set_xlabel(r"\textbf{X/C}",x=1.05)
      
      ax[1].spines['top'].set_visible(False)
      ax[1].spines['right'].set_visible(False)
      ax[1].spines['bottom'].set_visible(False)
      ax[1].spines['left'].set_visible(False)
      ax[1].set_xticks([])
      ax[1].set_yticks([])
      ax[1].axis('scaled')
      
      if save==True:
          plt.savefig(savename+'_cp.pdf',dpi=600,bbox_inches='tight')
      
      return(fig)


'''
Calculate Feild
'''


def FIELD_U_V_CP(xlimit,zlimit,NX,NZ,GAMMA,XP,ZP,ALPHA,h):
      #Calculate full feild parameters for plotting. Currently Velocity and CP
      #and Velocity Potential + Streamlines that find stagnation points 
      ALPHA=ALPHA*m.pi/180
        
      K = len(XP)
      M = len(XP[0])-1
      
      if(len(XP)!=len(GAMMA)):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      
          
      AF = np.zeros((NX*NZ,K*(M+1)))
      BF = np.zeros((NX*NZ,K*(M+1)))
      x = np.linspace(xlimit[0], xlimit[1], NX)
      z = np.linspace(zlimit[0], zlimit[1], NZ)
      XG, ZG = np.meshgrid(x, z, indexing='xy')
      XGV = np.ravel(XG)
      ZGV = np.ravel(ZG)
      
      BETA = np.asarray([np.zeros((M,1)) for k in range(K)]) 
      
      for k,i in np.ndindex(K,M):
            DX = XP[k][i+1]-XP[k][i]
            DZ = ZP[k][i+1]-ZP[k][i]
            BETA[k][i] = m.atan2(DZ,DX)
      

      for k in range(K):
            
            for i,j in np.ndindex(len(XGV),M):
                  Aj=k*M + k + j
                 
                  XI=XGV[i]
                  ZI=ZGV[i]
                  XJ=XP[k][j]
                  ZJ=ZP[k][j]
                  XJ_1=XP[k][j+1]
                  ZJ_1=ZP[k][j+1]

                  
                  XIP = (XI-XJ)*m.cos(BETA[k][j]) + (ZI-ZJ)*m.sin(BETA[k][j])
                  ZIP = -(XI-XJ)*m.sin(BETA[k][j]) + (ZI-ZJ)*m.cos(BETA[k][j])
                  XJ_1P = (XJ_1-XJ)*m.cos(BETA[k][j]) + (ZJ_1-ZJ)*m.sin(BETA[k][j])
                  # XJ,ZJ and ZJ_2 will always be zero in panel coordinates
     
                  R1 = m.sqrt(XIP**2 + ZIP**2)
                  R2 = m.sqrt((XIP-XJ_1P)**2 + ZIP**2)
                  THETA1 = m.atan2(ZIP,XIP)
                  THETA2 = m.atan2(ZIP, (XIP-XJ_1P))
                  

#                  U1P = -(ZIP*m.log(R2/R1) + (XIP-XJ_1P)*(THETA2-THETA1))\
#                              /(2*m.pi*XJ_1P)
#                  U2P = (ZIP*m.log(R2/R1) + XIP*(THETA2-THETA1))\
#                        /(2*m.pi*XJ_1P)
#                  W1P = -((XJ_1P-ZIP*(THETA2-THETA1))+(XJ_1P-XIP)*m.log(R1/R2))\
#                        /(2*m.pi*XJ_1P)
#                  W2P = ((XJ_1P-ZIP*(THETA2-THETA1))-XIP*m.log(R1/R2))\
#                        /(2*m.pi*XJ_1P)
                        
                        
                  U1P = (THETA2-THETA1)
                  U2P = ((2*XIP-XJ_1P)*U1P +2*ZIP*m.log(R2/R1))/XJ_1P
                  W1P = m.log(R2/R1)
                  W2P = ((2*XIP-XJ_1P)*W1P -2*ZIP*(THETA2-THETA1) +2*XJ_1P)/XJ_1P
                  
                  
                  U1 = U1P*m.cos(-BETA[k][j])+W1P*m.sin(-BETA[k][j])
                  U2 = U2P*m.cos(-BETA[k][j])+W2P*m.sin(-BETA[k][j])
                  W1 = -U1P*m.sin(-BETA[k][j])+W1P*m.cos(-BETA[k][j])
                  W2 = -U2P*m.sin(-BETA[k][j])+W2P*m.cos(-BETA[k][j])
                  
                  
                  if j == 0:
                        AF[i,Aj] = (W1-W2)/(4*m.pi)
                        WHOLD = (W1+W2)/(4*m.pi)
                        
                        BF[i,Aj] = (U1-U2)/(4*m.pi)
                        UHOLD = (U1+U2)/(4*m.pi)
                        
                  elif j == M-1:
                        AF[i,Aj] = (W1-W2)/(4*m.pi) + WHOLD
                        AF[i,Aj+1] = (W1+W2)/(4*m.pi)
                        
                        BF[i,Aj] = (U1-U2)/(4*m.pi) + UHOLD
                        BF[i,Aj+1] = (U1+U2)/(4*m.pi)
                        
                  else: 
                        AF[i,Aj] = (W1-W2)/(4*m.pi) + WHOLD
                        WHOLD = (W1+W2)/(4*m.pi)
                        
                        BF[i,Aj] = (U1-U2)/(4*m.pi) + UHOLD
                        UHOLD = (U1+U2)/(4*m.pi)
                  
#                  if(j==0):
#                        AF[i,Aj] = W1
#                        HOLDAF = W2
#                        BF[i,Aj] = U1
#                        HOLDBF = U2
#                  elif(j==M-1):
#                        AF[i,Aj] = W1 +HOLDAF
#                        AF[i,Aj+1] = W2
#                        BF[i,Aj] = U1 +HOLDBF
#                        BF[i,Aj+1] = U2
#                  else:
#                        AF[i,Aj] = W1 + HOLDAF
#                        HOLDAF =  W2
#                        BF[i,Aj] = U1 + HOLDBF
#                        HOLDBF = U2     

      AF[AF == 0] = 'nan'
      BF[BF == 0] = 'nan'
      
      U = np.zeros((NX*NZ,1))
      W = np.zeros((NX*NZ,1))
      VM = np.zeros((NX*NZ,1))
      CPF = np.zeros((NX*NZ,1))
      
      i = 0
      while(i<len(XGV)):
             VELW = 0
             VELU = 0
             for k in range(K):
                   j = 0
                   Aj = k*M+ k
                   while(j<=M):
                         VELW=VELW +AF[i,Aj]*GAMMA[k][j]
                         VELU=VELU +BF[i,Aj]*GAMMA[k][j]
                         j = j +1
                         Aj = Aj +1
                       
             W[i] = VELW + m.sin(ALPHA)
             U[i] = VELU + m.cos(ALPHA)
             VM[i] = m.sqrt(W[i]**2 + U[i]**2)
             CPF[i] = 1 - VM[i]**2
             i = i+1
             
      U = np.reshape(U,(NX,NZ))
      W = np.reshape(W,(NX,NZ))
      VM = np.reshape(VM,(NX,NZ))
      CPF = np.reshape(CPF,(NX,NZ))
      
      return(U, W, VM, CPF, XG, ZG)


def STREAMLINES(xlimit,zlimit,NX,NZ,GAMMA,XP,ZP,ALPHA,PSI,h):
      #Calculate full feild parameters for plotting. Currently Velocity and CP
      #and Velocity Potential + Streamlines that find stagnation points 
      
      ALPHA=ALPHA*m.pi/180
        
      K = len(XP)
      M = len(XP[0])-1
      
      if(len(XP)!=len(GAMMA)):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      
      
      AF = np.zeros((NX*NZ,K*(M+1)))
      BF = np.zeros((NX*NZ,K*(M+1)))
      x = np.linspace(xlimit[0], xlimit[1], NX)
      z = np.linspace(zlimit[0], zlimit[1], NZ)
      XG, ZG = np.meshgrid(x, z, indexing='xy')
      XGV = np.ravel(XG)
      ZGV = np.ravel(ZG)
      
      BETA = np.asarray([np.zeros((M,1)) for k in range(K)]) 
      
      for k,i in np.ndindex(K,M):
            DX = XP[k][i+1]-XP[k][i]
            DZ = ZP[k][i+1]-ZP[k][i]
            BETA[k][i] = m.atan2(DZ,DX)
      

      for k in range(K):
            
            for i,j in np.ndindex(len(XGV),M):
                  Aj=k*M + k + j
                 
                  XI=XGV[i]
                  ZI=ZGV[i]
                  XJ=XP[k][j]
                  ZJ=ZP[k][j]
                  XJ_1=XP[k][j+1]
                  ZJ_1=ZP[k][j+1]

                  
                  XIP = (XI-XJ)*m.cos(BETA[k][j]) + (ZI-ZJ)*m.sin(BETA[k][j])
                  ZIP = -(XI-XJ)*m.sin(BETA[k][j]) + (ZI-ZJ)*m.cos(BETA[k][j])
                  XJ_1P = (XJ_1-XJ)*m.cos(BETA[k][j]) + (ZJ_1-ZJ)*m.sin(BETA[k][j])
                  # XJ,ZJ and ZJ_2 will always be zero in panel coordinates
     
                  R1 = m.sqrt(XIP**2 + ZIP**2)
                  R2 = m.sqrt((XIP-XJ_1P)**2 + ZIP**2)
                  THETA1 = m.atan2(ZIP,XIP)
                  THETA2 = m.atan2(ZIP, (XIP-XJ_1P))
                  
                  
                        
                  PSI1 = XIP*m.log(R1)-(XIP-XJ_1P)*m.log(R2) -XJ_1P\
                              + ZIP*(THETA2-THETA1)
                  PSI2 = ((2*XIP-XJ_1P)*PSI1 + R2**2*m.log(R2)\
                          - R1**2*m.log(R1)\
                                + (XIP**2-(XIP-XJ_1P)**2)/2)/(XJ_1P)
                  
                  if j == 0:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi)
                        PSIHOLD = (PSI1+PSI2)/(4*m.pi)
                        
                  elif j == M-1:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                        AF[i,Aj+1] = (PSI1+PSI2)/(4*m.pi)
                        
                  else:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                        PSIHOLD = (PSI1+PSI2)/(4*m.pi)
    

#      AF[AF == 0] = 'nan'
#      BF[BF == 0] = 'nan'
      
      PSIA = np.zeros((NX*NZ,1))
      i = 0
      while(i<len(XGV)):
             PSIX = 0
             for k in range(K):
                   j = 0
                   Aj = k*M+ k
                   while(j<=M):
                         PSIX=PSIX +AF[i,Aj]*GAMMA[k][j]
                         j = j +1
                         Aj = Aj +1
                       
             PSIA[i] = PSIX +  (ZGV[i]*m.cos(ALPHA) -\
                                             XGV[i]*m.sin(ALPHA))
             i = i+1
             
      PSIA = np.reshape(PSIA,(NX,NZ))
      #interpolate Values of PSI
      XZPSI=[np.zeros((NX,2)) for psi in range(len(PSI))]

      for psi in range(len(PSI)):
            for n in range(NX):
                  f=interp1d(PSIA[:,n], ZG[:,n], kind='linear',fill_value='extrapolate')
                  ZPSI=f(PSI[psi])
                  XPSI=XG[:,n][0]
                  XZPSI[psi][n][0]=XPSI
                  XZPSI[psi][n][1]=ZPSI

      
      

      return(XZPSI)


def POTENTIAL(xlimit,zlimit,NX,NZ,GAMMA,XP,ZP,ALPHA,h):
      #Calculate full feild parameters for plotting. Currently Velocity and CP
      #and Velocity Potential + Streamlines that find stagnation points 
      
      ALPHA=ALPHA*m.pi/180
        
      K = len(XP)
      M = len(XP[0])-1
      
      if(len(XP)!=len(GAMMA)):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      
      
      AF = np.zeros((NX*NZ,K*(M+1)))
      BF = np.zeros((NX*NZ,K*(M+1)))
      x = np.linspace(xlimit[0], xlimit[1], NX)
      z = np.linspace(zlimit[0], zlimit[1], NZ)
      XG, ZG = np.meshgrid(x, z, indexing='xy')
      XGV = np.ravel(XG)
      ZGV = np.ravel(ZG)
      
      BETA = np.asarray([np.zeros((M,1)) for k in range(K)]) 
      
      for k,i in np.ndindex(K,M):
            DX = XP[k][i+1]-XP[k][i]
            DZ = ZP[k][i+1]-ZP[k][i]
            BETA[k][i] = m.atan2(DZ,DX)
      

      for k in range(K):
            
            for i,j in np.ndindex(len(XGV),M):
                  Aj=k*M + k + j
                 
                  XI=XGV[i]
                  ZI=ZGV[i]
                  XJ=XP[k][j]
                  ZJ=ZP[k][j]
                  XJ_1=XP[k][j+1]
                  ZJ_1=ZP[k][j+1]

                  
                  XIP = (XI-XJ)*m.cos(BETA[k][j]) + (ZI-ZJ)*m.sin(BETA[k][j])
                  ZIP = -(XI-XJ)*m.sin(BETA[k][j]) + (ZI-ZJ)*m.cos(BETA[k][j])
                  XJ_1P = (XJ_1-XJ)*m.cos(BETA[k][j]) + (ZJ_1-ZJ)*m.sin(BETA[k][j])
                  # XJ,ZJ and ZJ_2 will always be zero in panel coordinates
     
                  R1 = m.sqrt(XIP**2 + ZIP**2)
                  R2 = m.sqrt((XIP-XJ_1P)**2 + ZIP**2)
                  THETA1 = m.atan2(ZIP,XIP)
                  THETA2 = m.atan2(ZIP, (XIP-XJ_1P))
                  
                  
                        
                  PSI1 = THETA2*(XIP-XJ_1P) -THETA1*XIP +ZIP*m.log(R2/R1)
                  PSI2 = ((2*XIP-XJ_1P)*PSI1 + (R1**2)*THETA1 -(R2**2)*THETA2 +ZIP*XJ_1P)/XJ_1P
                  
                  if j == 0:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi)
                        PSIHOLD = (PSI1+PSI2)/(4*m.pi)
                        
                  elif j == M-1:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                        AF[i,Aj+1] = (PSI1+PSI2)/(4*m.pi)
                        
                  else:
                        AF[i,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                        PSIHOLD = (PSI1+PSI2)/(4*m.pi)
    

      AF[AF == 0] = 'nan'
      BF[BF == 0] = 'nan'
      
      PSIA = np.zeros((NX*NZ,1))
      i = 0
      while(i<len(XGV)):
             PSIX = 0
             for k in range(K):
                   j = 0
                   Aj = k*M+ k
                   while(j<=M):
                         PSIX=PSIX +AF[i,Aj]*GAMMA[k][j]
                         j = j +1
                         Aj = Aj +1
                       
             PSIA[i] = PSIX+\
             (XGV[i]*m.cos(ALPHA) +\
                                             ZGV[i]*m.sin(ALPHA))
             i = i+1
             
      PSIA = np.reshape(PSIA,(NX,NZ))
     
      return(PSIA, XG, ZG)


      
'''
Plot Feild
'''


def FIELD_PLOT_CP_U_V(U, W, VM, CPF, XG, ZG, XP, ZP, zlimit,h=0,mirror=False,savename='test',save=True):
      K = int(len(XP))
            
      if(mirror==True):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      
      if h!=0 and mirror==False:
            Zshift=(h-np.min(ZP))
      else:
            Zshift=0
            
      fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(7,7))
      for k in range(K):
            ax[0].plot(XP[k], ZP[k]+Zshift, color='k', linestyle='-',\
                         marker='None', linewidth=2, zorder=3)
            ax[0].fill(XP[k], ZP[k]+Zshift,\
                    color='w', linestyle='solid', linewidth=2, zorder=2)
            ax[1].plot(XP[k], ZP[k]+Zshift, color='k', linestyle='-',\
                   marker='None', linewidth=2, zorder=3)
            ax[1].fill(XP[k], ZP[k]+Zshift,\
                    color='w', linestyle='solid', linewidth=2, zorder=2)
      
      norm1 = MidpointNormalize(vmin=np.min(CPF), vmax=1, midpoint=0)
      cpfplot = ax[0].contourf(XG, ZG, CPF, 30, cmap='seismic',norm=norm1, zorder= 0)
      divider1 = make_axes_locatable(ax[0])
      cax1 = divider1.append_axes("right", size="5%", pad=0.1)
      cbar1 = plt.colorbar(cpfplot, cax=cax1)
      cbar1.set_label(r"$\textrm{C\textsubscript{P}}$", fontsize=16)
      ax[0].axis('scaled',adjustable='box')
      if h!=0:
            ax[0].axhline(y=0, xmin=0, xmax=1, color='k',linewidth=2,zorder = 4)
            ax[0].axhspan(zlimit[0], ymax=0, xmin=0, xmax=1, facecolor = 'w', hatch='/')
      
      norm2 = MidpointNormalize(vmin=0, vmax=np.max(VM), midpoint=1)
      vmplot = ax[1].contourf(XG, ZG, VM, 30, cmap='jet',norm=norm2, zorder= 0)
      divider2 = make_axes_locatable(ax[1])
      cax2 = divider2.append_axes("right", size="5%", pad=0.1)
      cbar2 = plt.colorbar(vmplot, cax=cax2)
      cbar2.set_label(r"$\displaystyle |\mathbf{u}|/U_\infty$", fontsize=16)


      if h!=0:
            ax[1].axhline(y=0, xmin=0, xmax=1, color='k',linewidth=2, zorder = 4)
            ax[1].axhspan(zlimit[0], ymax=0, xmin=0, xmax=1, facecolor = 'w', hatch='/')
      
      ax[1].axis('scaled',adjustable='box')
      ax[0].set(ylim=(zlimit[0],zlimit[1]))
      ax[1].set(ylim=(zlimit[0],zlimit[1]))
      
      
      ax[0].set_ylabel(r"\textbf{Z/C}")
      ax[1].set_ylabel(r"\textbf{Z/C}")
      ax[1].set_xlabel(r"\textbf{X/C}")

      fig.tight_layout(h_pad=1,w_pad=0)
      
      if save==True:
          plt.savefig(savename+'_field.pdf',dpi=600,bbox_inches='tight')
          
      return(fig,ax)


def FIELD_PLOT_STREAMLINES(XZPSI, XP, ZP, zlimit, h=0,mirror=False,savename='test',save=True):
      K = int(len(XP))
            
      if(mirror==True):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
      
      if h!=0 and mirror==False:
            Zshift=(h-np.min(ZP))
      else:
            Zshift=0

      for k in range(K):
#            if k<=int(K/2):
            ax.plot(XP[k], ZP[k]+Zshift, color='k', linestyle='-',\
                         marker='None', linewidth=2, zorder=3)
#            else:
#                  ax.plot(XP[k], ZP[k]+Zshift, color='k', linestyle='--',\
#                               marker='None', linewidth=1.5, zorder=3,alpha=0.8)
                  
            ax.fill(XP[k], ZP[k]+Zshift,\
                    color='w', linestyle='solid', linewidth=1, zorder=2)
      for psi in range(len(XZPSI)):
            ax.plot(XZPSI[psi][:,0],XZPSI[psi][:,1],color='darkblue',zorder=0,linewidth=2)
      

      if h!=0:
            ax.axhline(y=0, xmin=0, xmax=1, color='k',linewidth=2,zorder = 4)
            ax.axhspan(zlimit[0], ymax=0, xmin=0, xmax=1, facecolor = 'w', hatch='/')
      
      
      ax.axis('scaled',adjustable='box')
      ax.set(xlim=(-0.5,1.5))
      ax.set(ylim=(-0.1,1.0))
      
      ax.set_ylabel(r"\textbf{Z/C}")
      ax.set_xlabel(r"\textbf{X/C}")

      fig.tight_layout(h_pad=0,w_pad=0) 
      
      if save==True:
          plt.savefig(savename+'_streamlines.pdf',dpi=600,bbox_inches='tight')
      
      return(fig,ax)


def FIELD_PLOT_POTENTIAL(PHI, XG, ZG, XP, ZP, h=0,mirror=False):
      K = int(len(XP))
            
      if(mirror==True):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            MINZ=[]
            for k in range(K):
                  min(ZP[k])
                  MINZ.append(min(ZP[k]))
            Minz=min(MINZ)
            for k in range(K):
                  XP2[k] = XP[k]
                  XP2[k+K] = XP[k]
                  ZP2[k] = ZP[k] + (h-Minz)
                  ZP2[k+K] = -ZP[k] -(h-Minz)
            XP=XP2
            ZP=ZP2
            K=len(XP)
      fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(7,7))
      
      if h!=0 and mirror==False:
            Zshift=(h-np.min(ZP))
      else:
            Zshift=0

      for k in range(K):
            ax.plot(XP[k], ZP[k]+Zshift, color='k', linestyle='-',\
                         marker='None', linewidth=2, zorder=3)
            ax.fill(XP[k], ZP[k]+Zshift,\
                    color='w', linestyle='solid', linewidth=2, zorder=2)
      
      norm1 = MidpointNormalize(vmin=np.min(PHI), vmax=np.max(PHI), midpoint=0)
      cpfplot = ax.contourf(XG, ZG, PHI, 5, cmap='seismic',norm=norm1, zorder= 0)
      divider1 = make_axes_locatable(ax)
      cax1 = divider1.append_axes("right", size="5%", pad=0.1)
      cbar1 = plt.colorbar(cpfplot, cax=cax1)
      cbar1.set_label(r"$\Phi$", fontsize=16,rotation=90)

      if h!=0:
            ax.axhline(y=0, xmin=0, xmax=1, color='k',linewidth=2,zorder = 4)
            ax.axhspan(zlimit[0], ymax=0, xmin=0, xmax=1, facecolor = 'w', hatch='/')

      
      ax.axis('scaled',adjustable='box')
      ax.set(ylim=(zlimit[0],zlimit[1]))
      ax.set(xlim=(xlimit[0],xlimit[1]))
      
      
      ax.set_ylabel(r"\textbf{Z/C}")
      ax.set_xlabel(r"\textbf{X/C}")

      fig.tight_layout(h_pad=1,w_pad=0)
      
      return(fig,ax)
      