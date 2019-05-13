# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:39:59 2019

@author: ollenil
"""
import numpy as np
import math as m
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt


def LOAD(filepath,filename,plot=False):
      
      path = filepath+filename
      
      XC = []
      ZC = []
      
      with open(path, 'r') as CR:
            XCR, ZCR = np.loadtxt(CR, dtype=float, delimiter='\t', unpack=True)
            
      k = 0
      holdi = 0
      for i in range(len(XCR)):
            if XCR[i] == 999.0 and k == 0:
                  XC.append(XCR[holdi:i])
                  ZC.append(ZCR[holdi:i])
                  holdi = i
                  k = k + 1
            elif XCR[i] == 999.0 and k != 0:
                  XC.append(XCR[holdi+1:i])
                  ZC.append(ZCR[holdi+1:i])
                  holdi = i
                  k = k + 1
            else:
                  pass
      if k == 0:
            XC.append(XCR[holdi:])
            ZC.append(ZCR[holdi:])
      else:
            XC.append(XCR[holdi+1:])
            ZC.append(ZCR[holdi+1:])

      if plot==True:
            width = 8
            plt.figure(figsize=(width, width))
            plt.grid()
            plt.xlabel('x', fontsize=16)
            plt.ylabel('z', fontsize=16)
            plt.axis('scaled', adjustable='box')
            for k in range(len(XC)):
                  plt.plot(XC[k], ZC[k], color='k', linestyle='-',\
                           marker='None', linewidth=2) 
                  
#            plt.xlim(np.min(XC)-0.1,np.max(XC)+0.1)
#            plt.ylim(np.min(ZC)-0.1,np.max(ZC)+0.1)
                              
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.3,0.9)
            
      return(XC,ZC)
      
def PANEL(XC,ZC,M=100,plot=False):
      
      K=len(XC)
      N = M+2
      
      L = [0 for k in range(K)]
      LT = [0 for k in range(K)]
      XCP = [[0,0] for k in range(K)]
      ZCP = [[0,0] for k in range(K)]
      XPP = [[0,0] for k in range(K)]
      ZPP = [[0,0] for k in range(K)]
      XP = [[0,0] for k in range(K)]
      ZP = [[0,0] for k in range(K)]
      A = [[0,0] for k in range(K)]
      LE = [0 for k in range(K)]
      
      for k in range(K):
            L[k] = np.where(XC[k] == XC[k].min())[0][0]
            
            while True:
                  A[k][0] = m.atan2(ZC[k][0]-ZC[k][L[k]],XC[k][0]-XC[k][L[k]])
                  XCP[k][0] = (XC[k][:L[k]+1]-XC[k][L[k]])*m.cos(A[k][0]) + (ZC[k][:L[k]+1]-ZC[k][L[k]])*m.sin(A[k][0])
                  ZCP[k][0] = (ZC[k][:L[k]+1]-ZC[k][L[k]])*m.cos(A[k][0]) - (XC[k][:L[k]+1]-XC[k][L[k]])*m.sin(A[k][0])
                  A[k][1] = m.atan2(ZC[k][-1]-ZC[k][L[k]],XC[k][-1]-XC[k][L[k]])
                  XCP[k][1] = (XC[k][L[k]:]-XC[k][L[k]])*m.cos(A[k][1]) + (ZC[k][L[k]:]-ZC[k][L[k]])*m.sin(A[k][1])
                  ZCP[k][1] = (ZC[k][L[k]:]-ZC[k][L[k]])*m.cos(A[k][1]) - (XC[k][L[k]:]-XC[k][L[k]])*m.sin(A[k][1])
                  XCPT = np.concatenate((XCP[k][0],XCP[k][1][1:]))
                  LT[k]=np.where(XCPT == XCPT.min())[0][0]
                  
                  if L[k] == LT[k]:
                        break
                  else:
                        L[k] = LT[k]
                        
            XPP[k][0] = (XCP[k][0][0]+XCP[k][0][-1])/2.0*(1+np.cos(np.linspace(0.0, m.pi, int(N/2))))
            XPP[k][1] = (XCP[k][1][0]+XCP[k][1][-1])/2.0*(1+np.cos(np.linspace(m.pi, 2*m.pi, int(N/2))))
            f1 = interp1d(XCP[k][0][::-1], ZCP[k][0][::-1],kind='linear')
            ZPP[k][0] = f1(XPP[k][0][::-1])
            ZPP[k][0] = ZPP[k][0][::-1]
            f2 = interp1d(XCP[k][1],ZCP[k][1],kind='linear')
            ZPP[k][1]=f2(XPP[k][1])
            XP[k][0] = XC[k][L[k]] + XPP[k][0]*m.cos(A[k][0]) - ZPP[k][0]*m.sin(A[k][0])
            LE[k] = np.where(XP[k][0] == XP[k][0].min())[0][0]
            ZP[k][0] = ZC[k][L[k]] + ZPP[k][0]*m.cos(A[k][0]) + XPP[k][0]*m.sin(A[k][0]) 
            XP[k][1] = XC[k][L[k]] + XPP[k][1]*m.cos(A[k][1]) - ZPP[k][1]*m.sin(A[k][1])
            ZP[k][1] = ZC[k][L[k]] + ZPP[k][1]*m.cos(A[k][1]) + XPP[k][1]*m.sin(A[k][1]) 
            XP[k] = np.concatenate((XP[k][0],XP[k][1][1:]))
            ZP[k] = np.concatenate((ZP[k][0],ZP[k][1][1:]))
            
            if ZP[k][LE[k]-1]>ZP[k][LE[k]+1]:
                  XP[k] = XP[k][::-1]
                  ZP[k] = ZP[k][::-1]
                             
      if plot == True:          
            width = 8
            plt.figure(figsize=(width, width))
            plt.grid()
            plt.xlabel('x', fontsize=16)
            plt.ylabel('z', fontsize=16)
            plt.axis('scaled', adjustable='box')
            XLIM=[]
            ZLIM=[]
            for k in range(K):
                  plt.plot(XP[k], ZP[k], color='r', linestyle='-', marker='x',markeredgecolor='r', linewidth=2) 
                  plt.plot(XC[k], ZC[k], color='k', linestyle='-', marker='None', linewidth=2)
                  XLIM.append(max(XP[k]))
                  XLIM.append(min(XP[k]))
                  ZLIM.append(max(ZP[k]))
                  ZLIM.append(min(ZP[k]))
            plt.xlim(min(XLIM)-0.1,max(XLIM)+0.1)
            plt.ylim(min(ZLIM)-0.1,max(ZLIM)+0.1) 
     
      return(XP,ZP)
 

def ASOLVE(XP,ZP,ALPHA=0,UINF=1,GE=False,h=0.2): 
      #SOLVE THE ANALYSIS PROBLEM
      ALPHA=ALPHA*m.pi/180
      K = len(XP)
      M = len(XP[0])-1
      
      if(GE==True):
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

#       INITIATE ARRAYS
      BETA = np.asarray([np.zeros((M,1)) for k in range(K)])              
      PANEL_LENGTH = np.asarray([np.zeros((M,1)) for k in range(K)])   
      A = np.zeros((K*(M+2),K*(M+2)))             
      PSIINFITY = np.zeros((K*(M+2),1))      

      for k,i in np.ndindex(K,M):
            DX = XP[k][i+1]-XP[k][i]
            DZ = ZP[k][i+1]-ZP[k][i]
            BETA[k][i] = m.atan2(DZ,DX)
      

      for ki,kj in np.ndindex(K,K):
            
            for i,j in np.ndindex(M+1,M):
                  Ai=ki*(M+2) + i
                  Aj=kj*(M+1) + j
                 
                  
                  XI=XP[ki][i]
                  ZI=ZP[ki][i]
                  XJ=XP[kj][j]
                  ZJ=ZP[kj][j]
                  XJ_1=XP[kj][j+1]
                  ZJ_1=ZP[kj][j+1]

                  
                  XIP = (XI-XJ)*m.cos(BETA[kj][j]) + (ZI-ZJ)*m.sin(BETA[kj][j])
                  ZIP = -(XI-XJ)*m.sin(BETA[kj][j]) + (ZI-ZJ)*m.cos(BETA[kj][j])
                  XJ_1P = (XJ_1-XJ)*m.cos(BETA[kj][j]) + (ZJ_1-ZJ)*m.sin(BETA[kj][j])
                  # XJ,ZJ and ZJ_2 will always be zero in panel coordinates
                  
                  if (i == 0) and ki == kj:
                        PANEL_LENGTH[kj][j] = XJ_1P
                        
                  R1 = m.sqrt(XIP**2 + ZIP**2)
                  R2 = m.sqrt((XIP-XJ_1P)**2 + ZIP**2)
                  THETA1 = m.atan2(ZIP,XIP)
                  THETA2 = m.atan2(ZIP, (XIP-XJ_1P))
                  
                  
                  if j==i and ki == kj:
                        PSI1 = (XIP-XJ_1P)*(1-m.log(R2))
                        PSI2 = -PSI1 - (XIP-XJ_1P)*m.log(R2) + (XIP-XJ_1P)/2
                        
                  elif j==i-1 and ki == kj:
                        PSI1 = XIP*(m.log(R1)-1)
                        PSI2 = PSI1 - XIP*m.log(R1) + XIP/2
                        
                  elif i==0 and j==M-1 and ki == kj:
                        PSI1 = XIP*(m.log(R1)-1)
                        PSI2 = PSI1 - XIP*m.log(R1) + XIP/2      
                        
                  elif i==M and j==0 and ki == kj:
                        PSI1 = (XIP-XJ_1P)*(1-m.log(R2))
                        PSI2 = -PSI1 - (XIP-XJ_1P)*m.log(R2) + (XIP-XJ_1P)/2
                                                
                  else:     
                        PSI1 = XIP*m.log(R1)-(XIP-XJ_1P)*m.log(R2) -XJ_1P\
                                    + ZIP*(THETA2-THETA1)
                        PSI2 = ((2*XIP-XJ_1P)*PSI1 + R2**2*m.log(R2)\
                                - R1**2*m.log(R1)\
                                      + (XIP**2-(XIP-XJ_1P)**2)/2)/(XJ_1P)
                  
                
                  if i==M:
                        if ki == kj:
                              if j==0:
                                    A[Ai,Aj]=1
                                    A[Ai+1,Aj]=1
                                    
                              elif j==1:  
                                    A[Ai,Aj]=-2     
                              
                              elif j==2:
                                    A[Ai,Aj]=1
                                    
                              elif j==M-1:                     
                                    A[Ai,Aj]=2
                                    A[Ai,Aj+1]=-1
                                    A[Ai+1,Aj+1] = 1  
                              
                              elif j==M-2:   
                                    A[Ai,Aj]=-1

                           
                  else:            
                        if j == 0:
                              A[Ai,Aj] = (PSI1-PSI2)/(4*m.pi)
                              PSIHOLD = (PSI1+PSI2)/(4*m.pi)
                              
                        elif j == M-1:
                              A[Ai,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                              A[Ai,Aj+1] = (PSI1+PSI2)/(4*m.pi)
                              
                        else:
                              A[Ai,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                              PSIHOLD = (PSI1+PSI2)/(4*m.pi)
         
                        if (j == 0) and ki == kj:
                              PSIINFITY[Ai] = UINF*(-ZP[ki][i]*m.cos(ALPHA) +\
                                             XP[ki][i]*m.sin(ALPHA))
                              A[Ai,-(K-ki)]=-1

      SOL = np.linalg.solve(A,PSIINFITY)
      GAMMA = SOL[0:K*(M+1)]
      GAMMAK=[0 for k in range(K)]
      
      for k in range(K):
            GAMMAK[k]=GAMMA[k*(M+1):(k+1)*(M+1)]
      PSI = SOL[K*(M+1)::]
      
      #Calculate CL
      if(GE==True):
            K=int(K/2)
      CL=[0 for k in range(K)]
      for k,i in np.ndindex(K,M):
            CL[k] = CL[k]+(GAMMAK[k][i]+GAMMAK[k][i+1])*PANEL_LENGTH[k][i]
      PSI = PSI[0:K]
      return(GAMMAK,PSI,CL)
