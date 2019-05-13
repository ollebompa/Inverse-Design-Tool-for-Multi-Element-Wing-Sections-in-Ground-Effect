# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:33:46 2019

@author: ollen
"""
import math as m
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
import copy
import time
  



def ISOLVE(XP,ZP,GAMMAI,PSII,GAMMASP,XSP,iS,iE,ALPHA=0,UINF=1,GE=False,h=0.2):  
      #SOLVE THE INVERSE PROBLEM
      # INITIATE ARRAYS
      ALPHA=ALPHA*m.pi/180
      K = len(XP)
      M = len(XP[0])-1
      
      RES=[]
      RESTIME=[]
      restimestart=time.time()
      RTIME=[]
      RCOUNT=[]
      
      JACTIME=[]
      JACCOUNT=[]
      
      if(GE==True):
            XP2 = [0 for k in range(2*K)]
            ZP2 = [0 for k in range(2*K)]
            iS2 = [0 for k in range(2*K)]
            iE2 = [0 for k in range(2*K)]
            XSP2 = [0 for k in range(2*K)]
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
                  
                  XSP2[k] = XP[k]
                  XSP2[k+K] = XP[k]
                  
                  iS2[k]=iS[k]
                  iS2[k+K]=iS[k]
                  iE2[k]=iE[k]
                  iE2[k+K]=iE[k]
            XP=XP2
            ZP=ZP2
            XSP=XSP2
            iS=iS2
            iE=iE2
            K=len(XP)

      BETA = np.asarray([np.zeros((M,1)) for k in range(K)])    #PANELS ANGLES
      PANEL_LENGTH = np.asarray([np.zeros((M,1)) for k in range(K)]) #PANEL LENGTHS
      ARR = np.zeros((K*(M+2),K*(M+2)))          
      PSIINFITY = np.zeros((K*(M+2),1)) 
      W0=[0 for k in range(K)]
      OUT=[0 for k in range(K)]
      GC=[0 for k in range(K)]
      Wlen=[0 for k in range(K)]
#      GAMMASPK=[0 for k in range(K)]
      for k in range(K):   
            length= iE[k]-iS[k]+M+5
            lengthGC= iE[k]-iS[k]-1
            W0[k]=np.zeros((length,))
            OUT[k] = np.zeros((length,1)) 
            GC[k]= np.zeros((lengthGC,1))
            Wlen[k]=length
            
      GAMMA = np.asarray([np.zeros((M+1,1)) for k in range(K)]) 
      n= np.asarray([np.zeros((M+1,2)) for k in range(K)]) 
      
      
      XPI = copy.deepcopy(XP)
      ZPI = copy.deepcopy(ZP)
      
      XiS=[0 for k in range(K)]
      XiE=[0 for k in range(K)]
      ZiS=[0 for k in range(K)]
      ZiE=[0 for k in range(K)]
      
      for k in range(K): 
            XiS[k] = XP[k][iS[k]]
            XiE[k] = XP[k][iE[k]]
            ZiS[k] = ZP[k][iS[k]]
            ZiE[k] = ZP[k][iE[k]]
      
      #INITIAL GUESSES
      for k in range(K):
                  W0[k][:(iE[k]-iS[k])+1] = XP[k][iS[k]:iE[k]+1]
                  W0[k][(iE[k]-iS[k])+1:2*(iE[k]-iS[k])+2] = ZP[k][iS[k]:iE[k]+1]
                  W0[k][2*(iE[k]-iS[k])+2:(iE[k]-iS[k])+M+2] = np.concatenate((np.ravel(GAMMAI[k][:iS[k]]),\
                     np.ravel(GAMMAI[k][iE[k]+1:])))
                  W0[k][(iE[k]-iS[k])+M+2]=PSII[k]
                  W0[k][(iE[k]-iS[k])+M+3] = 0 #A
                  W0[k][(iE[k]-iS[k])+M+4] = 0 #B
                  


      #COMPUTE ALLOWED DIRECTION OF MOVEMENT
      for k in range(K):
            i=0
            while i<M+1:
                  if i==0:
                        DZ1 = ZP[k][i+1] -ZP[k][i]
                        DX1 = XP[k][i+1] -XP[k][i]
                        MAG1 = m.sqrt(DZ1**2+DX1**2)
                        n[k][i,0] = -DZ1/MAG1
                        n[k][i,1] = DX1/MAG1
                  elif i==(M):
                        DZ1 = ZP[k][i] -ZP[k][i-1]
                        DX1 = XP[k][i] -XP[k][i-1]
                        MAG1 = m.sqrt(DZ1**2+DX1**2)
                        n[k][i,0] = -DZ1/MAG1
                        n[k][i,1] = DX1/MAG1
                  else:
                        DZ1 = ZP[k][i+1] -ZP[k][i]
                        DX1 = XP[k][i+1] -XP[k][i]
                        DZ2 = ZP[k][i] -ZP[k][i-1]
                        DX2 = XP[k][i] -XP[k][i-1]
                        MAG1 = m.sqrt(DZ1**2+DX1**2)
                        MAG2 = m.sqrt(((DZ1+DZ2)/2)**2+((DX1+DX2)/2)**2)
                        n[k][i,0] = -(DZ1+DZ2)/(2*MAG2)
                        n[k][i,1] = (DX1+DX2)/(2*MAG2)
                        
                  i = i + 1
                  
      W0=np.concatenate([k for k in W0])

      #COMPUTE FUNCTION VALUES
      def R(W):
            #SPLIT W INTO X, Z, GAMMA, PSY, A/B ARRAYS
            X=[0 for k in range(K)]
            Z=[0 for k in range(K)]
            GAMMAW=[0 for k in range(K)]
            PSY=[0 for k in range(K)]
            A=[0 for k in range(K)]
            B=[0 for k in range(K)]
            print('R')
            rtimestart=time.time()
            for k in range(K):
                  startindex=sum(Wlen[:k])
                  endindex=sum(Wlen[:k+1])
                  WK=W[startindex:endindex]
                  
                  X[k] = WK[0:(iE[k]-iS[k])+1]
                  Z[k] = WK[(iE[k]-iS[k])+1:2*(iE[k]-iS[k])+2]
                  GAMMAW[k] = WK[2*(iE[k]-iS[k])+2:(iE[k]-iS[k])+M+2]
                  PSY[k] = WK[(iE[k]-iS[k])+M+2]
                  A[k] = WK[(iE[k]-iS[k])+M+3]
                  B[k] = WK[(iE[k]-iS[k])+M+4]
                        
            #REPALACE OLD COORDINATES WITH NEW GUESS
            for k in range(K):
                  XPI[k][iS[k]:iE[k]+1] = X[k]
                  ZPI[k][iS[k]:iE[k]+1] = Z[k]
                  #CALCULATE DISTANCE OF NEW POINT TO ALLOWED DIRECTION OF MOVMENT 
                  i = iS[k]+1
                  while iS[k]<i<iE[k]:
                        t = (n[k][i][0]*(XPI[k][i]-XP[k][i]) + n[k][i][1]*(ZPI[k][i]-ZP[k][i]))\
                              /(n[k][i][0]**2 + n[k][i][1]**2 )
                        GCx = XP[k][i]-XPI[k][i] + t*n[k][i][0]
                        GCz = ZP[k][i]-ZPI[k][i] + t*n[k][i][1]
                        GC[k][i-iS[k]-1] = m.sqrt(GCx**2 + GCz**2)
                        i = i+1
                  
                  OUT[k][M+2:M+(iE[k]-iS[k])+1] = GC[k]
                  OUT[k][-4] = X[k][0]-XiS[k]
                  OUT[k][-3] = Z[k][0]-ZiS[k]
                  OUT[k][-2] = X[k][-1]-XiE[k]
                  OUT[k][-1] = Z[k][-1]-ZiE[k]
                  
           
                  #CALCULATE GAMMA FOR INVERSE SEGEMENT
                  QSP = interp1d(XSP[k][iS[k]:iE[k]+1],\
                           np.reshape(GAMMASP[k][iS[k]:iE[k]+1],\
                          (len(GAMMASP[k][iS[k]:iE[k]+1]),)),kind='linear',\
                   fill_value='extrapolate')(X[k])
                  Q = np.zeros_like(X[k])
                  
                  i=0
                  while i<len(X[k]):
                        
                        Q[i] = QSP[i]\
                        +A[k]*f(X[k][i],XiS[k],XiE[k]) + B[k]*g(X[k][i],XiS[k],XiE[k])
                        i = i + 1
    
                  GAMMA[k][:iS[k]]=np.reshape(GAMMAW[k][0:iS[k]],(len(GAMMAW[k][0:iS[k]]),1))
                  GAMMA[k][iE[k]+1:]=np.reshape(GAMMAW[k][iS[k]:],(len(GAMMAW[k][iS[k]:]),1))
                  GAMMA[k][iS[k]:iE[k]+1] = np.reshape(Q,(len(Q),1))
                  

            for k,i in np.ndindex(K,M):
                  DX = XPI[k][i+1]-XPI[k][i]
                  DZ = ZPI[k][i+1]-ZPI[k][i]
                  BETA[k][i] = m.atan2(DZ,DX)
            
            #R1 to Rm
            for ki,kj in np.ndindex(K,K):
                  
                  for i,j in np.ndindex(M+1,M):
                        Ai=ki*(M+2) + i
                        Aj=kj*(M+1) + j  
                        XI=XPI[ki][i]
                        ZI=ZPI[ki][i]
                        XJ=XPI[kj][j]
                        ZJ=ZPI[kj][j]
                        XJ_1=XPI[kj][j+1]
                        ZJ_1=ZPI[kj][j+1]
        
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
                                          ARR[Ai,Aj]=1
                                          ARR[Ai+1,Aj]=1
                                          
                                    elif j==1:  
                                          ARR[Ai,Aj]=-2     
                                    
                                    elif j==2:
                                          ARR[Ai,Aj]=1
                                          
                                    elif j==M-1:                     
                                          ARR[Ai,Aj]=2
                                          ARR[Ai,Aj+1]=-1
                                          ARR[Ai+1,Aj+1] = 1  
                                    
                                    elif j==M-2:   
                                          ARR[Ai,Aj]=-1

                           
                        else:            
                              if j == 0:
                                    ARR[Ai,Aj] = (PSI1-PSI2)/(4*m.pi)
                                    PSIHOLD = (PSI1+PSI2)/(4*m.pi)
                                    
                              elif j == M-1:
                                    ARR[Ai,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                                    ARR[Ai,Aj+1] = (PSI1+PSI2)/(4*m.pi)
                                    
                              else:
                                    ARR[Ai,Aj] = (PSI1-PSI2)/(4*m.pi) + PSIHOLD
                                    PSIHOLD = (PSI1+PSI2)/(4*m.pi)
               
                              if (j == 0) and ki == kj:
                                    PSIINFITY[Ai] = UINF*(-ZPI[ki][i]*m.cos(ALPHA) +\
                                                   XPI[ki][i]*m.sin(ALPHA))
                                    ARR[Ai,-(K-ki)]=-1
            
            GAMMAOUT=np.concatenate([k for k in GAMMA])
            UNKOWNSOUT=np.concatenate((GAMMAOUT,np.reshape(PSY,(len(PSY),1))))
            OUTR=[]
            for k in range(K):
                  OUTKA=(ARR@UNKOWNSOUT-PSIINFITY)[k*(M+2):(k+1)*(M+2)]
                  OUTK=np.concatenate((OUTKA,OUT[k][M+2:]))
                  OUTR.append(OUTK)
          
            OUTR=np.concatenate([k for k in OUTR])
            norm = np.linalg.norm(OUTR)
            print(norm)
            RES.append(norm)
            restimeend=time.time()
            timeiter=restimeend-restimestart
            RESTIME.append(timeiter)
            rtimeend=time.time()
            rtime=rtimeend-rtimestart
            RTIME.append(rtime)
            RCOUNT.append(1)
            return(np.reshape(OUTR,(len(OUTR),)))

      def JAC(W):
            #SPLIT W INTO X, Z, GAMMA, PSY, A2/A2 ARRAYS
            X=[0 for k in range(K)]
            Z=[0 for k in range(K)]
            GAMMAW=[0 for k in range(K)]
            PSY=[0 for k in range(K)]
            A=[0 for k in range(K)]
            B=[0 for k in range(K)]
            print('JAC')
            jactimestart=time.time()
            for k in range(K):
                  startindex=sum(Wlen[:k])
                  endindex=sum(Wlen[:k+1])

                  WK=W[startindex:endindex]
                  X[k] = WK[0:(iE[k]-iS[k])+1]
                  Z[k] = WK[(iE[k]-iS[k])+1:2*(iE[k]-iS[k])+2]
                  GAMMAW[k] = WK[2*(iE[k]-iS[k])+2:(iE[k]-iS[k])+M+2]
                  PSY[k] = WK[(iE[k]-iS[k])+M+2]
                  A[k] = WK[(iE[k]-iS[k])+M+3]
                  B[k] = WK[(iE[k]-iS[k])+M+4]
            
            #REPALACE OLD COORDINATES WITH NEW GUESS
            for k in range(K):
   
                  XPI[k][iS[k]:iE[k]+1] = X[k]
                  ZPI[k][iS[k]:iE[k]+1] = Z[k]
      
                  QSP = interp1d(XSP[k][iS[k]:iE[k]+1],\
                           np.reshape(GAMMASP[k][iS[k]:iE[k]+1],\
                          (len(GAMMASP[k][iS[k]:iE[k]+1]),)),kind='linear',\
                              fill_value='extrapolate')(X[k])
                  Q = np.zeros_like(X[k])
                  
                  i=0
                  while i<len(X[k]):
                        Q[i] = QSP[i]\
                        +A[k]*f(X[k][i],XiS[k],XiE[k]) + B[k]*g(X[k][i],XiS[k],XiE[k])
                        i = i + 1

                  GAMMA[k][:iS[k]]=np.reshape(GAMMAW[k][0:iS[k]],(len(GAMMAW[k][0:iS[k]]),1))
                  GAMMA[k][iE[k]+1:]=np.reshape(GAMMAW[k][iS[k]:],(len(GAMMAW[k][iS[k]:]),1))
                  GAMMA[k][iS[k]:iE[k]+1] = np.reshape(Q,(len(Q),1))
                        
            for k,i in np.ndindex(K,M):
                  DX = XPI[k][i+1]-XPI[k][i]
                  DZ = ZPI[k][i+1]-ZPI[k][i]
                  BETA[k][i] = m.atan2(DZ,DX)
                  
            J=np.zeros((K, K),dtype=object)
            for ki,kj in np.ndindex(K,K):
                 kjshape=M+5+iE[kj]-iS[kj]
                 kishape=M+5+iE[ki]-iS[ki]
                 
                 J[ki,kj]=np.zeros((kishape,kjshape)) 

            #R1 to Rm
            for ki,kj in np.ndindex(K,K):
                  i = 0
                  while(i<M):
                        j = 0
                        while(j<M):
                              XI=XPI[ki][i]
                              ZI=ZPI[ki][i]
                              XJ=XPI[kj][j]
                              ZJ=ZPI[kj][j]
                              XJ_1=XPI[kj][j+1]
                              ZJ_1=ZPI[kj][j+1]
               
      
                              XIP = (XI-XJ)*m.cos(BETA[kj][j]) +(ZI-ZJ)*m.sin(BETA[kj][j])
                              ZIP = -(XI-XJ)*m.sin(BETA[kj][j]) + (ZI-ZJ)*m.cos(BETA[kj][j])
                              XJ_1P = (XJ_1-XJ)*m.cos(BETA[kj][j]) + (ZJ_1-ZJ)*m.sin(BETA[kj][j])
                              # X1 and Z2 will always be zero in panel coordinates
                              
                                    
                              if (i == 0) and ki == kj:
                                    PANEL_LENGTH[kj][j] = XJ_1P
                                    
                              R1 = m.sqrt(XIP**2 + ZIP**2)
                              R2 = m.sqrt((XIP-XJ_1P)**2 + ZIP**2)
                              THETA1 = m.atan2(ZIP,XIP)
                              THETA2 = m.atan2(ZIP, (XIP-XJ_1P))
                              
                              PSI1=PSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M)[0]
                              PSI2=PSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M)[1]
                              
                              if j == 0:
                                    aij = (PSI1-PSI2)/(4*m.pi)
                                    aijHOLD = (PSI1+PSI2)/(4*m.pi)
                                    
                              elif j == M-1:
                                    aij = (PSI1-PSI2)/(4*m.pi) + aijHOLD
                                    aiM = (PSI1+PSI2)/(4*m.pi)
                                    
                              else:
                                    aij = (PSI1-PSI2)/(4*m.pi) + aijHOLD
                                    aijHOLD = (PSI1+PSI2)/(4*m.pi)   
                              
                              if j==0:
                                    HOLD_dPSI_XI=0
                                    HOLD_dPSI_ZI=0
                                    HOLD_dPSI_XJ=0
                                    HOLD_dPSI_ZJ=0
                                    HOLD_dPSI_XJ_1=0
                                    HOLD_dPSI_ZJ_1=0
                                    

                              gradT=GRADT(XI,ZI,XJ,ZJ,XJ_1,ZJ_1,BETA[kj][j])
                              gradPSI=GRADPSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M)
                              dPSI_dq=np.dot(gradPSI,gradT)
                              
                              dPSI1_dXI=dPSI_dq[0][0][0]
                              dPSI1_dZI=dPSI_dq[0][0][1]
                              dPSI1_dXJ=dPSI_dq[0][0][2]
                              dPSI1_dZJ=dPSI_dq[0][0][3]
                              dPSI1_dXJ_1=dPSI_dq[0][0][4]
                              dPSI1_dZJ_1=dPSI_dq[0][0][5]
                              
                              dPSI2_dXI=dPSI_dq[1][0][0]
                              dPSI2_dZI=dPSI_dq[1][0][1]
                              dPSI2_dXJ=dPSI_dq[1][0][2]
                              dPSI2_dZJ=dPSI_dq[1][0][3]
                              dPSI2_dXJ_1=dPSI_dq[1][0][4]
                              dPSI2_dZJ_1=dPSI_dq[1][0][5]
                              
                              dJ_XI=(dPSI1_dXI-dPSI2_dXI)/(4*m.pi)
                              dJ_ZI=(dPSI1_dZI-dPSI2_dZI)/(4*m.pi)
                              dJ_XJ=(dPSI1_dXJ-dPSI2_dXJ)/(4*m.pi)
                              dJ_ZJ=(dPSI1_dZJ-dPSI2_dZJ)/(4*m.pi)
                              dJ_XJ_1=(dPSI1_dXJ_1-dPSI2_dXJ_1)/(4*m.pi)
                              dJ_ZJ_1=(dPSI1_dZJ_1-dPSI2_dZJ_1)/(4*m.pi)
                              
                              #aij w.r.t. XI,ZI 
                              if iS[ki]<=i<=iE[ki]:
                                    J[ki,ki][i,i-iS[ki]]+=GAMMA[kj][j]*(dJ_XI+HOLD_dPSI_XI)
                                    J[ki,ki][i,(iE[ki]-2*iS[ki])+1+i]+=GAMMA[kj][j]*(dJ_ZI+HOLD_dPSI_ZI)
                             
                              #aij w.r.t. XJ-1,ZJ-1      
                              if iS[kj]+1<=j<=iE[kj]+1:      
                                    J[ki,kj][i,j-iS[kj]-1]+=GAMMA[kj][j]*HOLD_dPSI_XJ
                                    J[ki,kj][i,(iE[kj]-2*iS[kj])+j]+=GAMMA[kj][j]*HOLD_dPSI_ZJ
                              
                              #aij w.r.t. XJ,ZJ
                              if iS[kj]<=j<=iE[kj]:
                                    J[ki,kj][i,j-iS[kj]]+=GAMMA[kj][j]*(dJ_XJ+HOLD_dPSI_XJ_1)
                                    J[ki,kj][i,(iE[kj]-2*iS[kj])+1+j]+=GAMMA[kj][j]*(dJ_ZJ+HOLD_dPSI_ZJ_1)
                              
                              #aij w.r.t. XJ+1,ZJ+1
                              if iS[kj]-1<=j<=iE[kj]-1:
                                    J[ki,kj][i,j-iS[kj]+1]+=GAMMA[kj][j]*dJ_XJ_1
                                    J[ki,kj][i,(iE[kj]-2*iS[kj])+2+j]+=GAMMA[kj][j]*dJ_ZJ_1
                              
                              #Holds w.r.t aij to carry to next loop
                              HOLD_dPSI_XI=(dPSI1_dXI+dPSI2_dXI)/(4*m.pi)
                              HOLD_dPSI_ZI=(dPSI1_dZI+dPSI2_dZI)/(4*m.pi)
                              HOLD_dPSI_XJ=(dPSI1_dXJ+dPSI2_dXJ)/(4*m.pi)
                              HOLD_dPSI_ZJ=(dPSI1_dZJ+dPSI2_dZJ)/(4*m.pi)
                              HOLD_dPSI_XJ_1=(dPSI1_dXJ_1+dPSI2_dXJ_1)/(4*m.pi)
                              HOLD_dPSI_ZJ_1=(dPSI1_dZJ_1+dPSI2_dZJ_1)/(4*m.pi)

                              #Gamma w.r.t. R
                              if j<iS[kj] or j>iE[kj]:
                                    if j<iS[kj]:
                                          J[ki,kj][i,2*(iE[kj]-iS[kj])+2+j] = aij
                                    else:
                                          J[ki,kj][i,(iE[kj]-iS[kj])+1+j] = aij
                                    
                              if iS[kj]<=j<=iE[kj]:       
                                    #Gamma w.r.t. XJ
                                    J[ki,kj][i,j-iS[kj]]+=aij*(A[kj]*GRADf(XJ,XiS[kj],XiE[kj])+B[kj]*GRADg(XJ,XiS[kj],XiE[kj]))
                                    #Gamma w.r.t. A and B           
                                    J[ki,kj][i,-2]+=aij*f(XJ,XiS[kj],XiE[kj])
                                    J[ki,kj][i,-1]+=aij*g(XJ,XiS[kj],XiE[kj])
                              
                              if j == M-1:
                                    if iS[ki]<=i<=iE[ki]:
                                          #aiM w.r.t. XI,ZI
                                          J[ki,ki][i,i-iS[ki]]+=GAMMA[kj][M]*HOLD_dPSI_XI
                                          J[ki,ki][i,(iE[ki]-2*iS[ki])+1+i]+=GAMMA[kj][M]*HOLD_dPSI_ZI
                                    
                                    if iE[ki]==M:
                                          #aiM w.r.t. XJ-1,ZJ-1       
                                          J[ki,kj][i,j-iS[kj]]+=GAMMA[kj][M]*HOLD_dPSI_XJ
                                          J[ki,kj][i,(iE[kj]-2*iS[kj])+j+1]+=GAMMA[kj][M]*HOLD_dPSI_ZJ
                                          #aij w.r.t. XJ,ZJ
                                          J[ki,kj][i,j-iS[kj]+1]+=GAMMA[kj][M]*HOLD_dPSI_XJ_1
                                          J[ki,kj][i,(iE[kj]-2*iS[kj])+2+j]+=GAMMA[kj][M]*HOLD_dPSI_ZJ_1
                                          #Gamma w.r.t. XJ
                                          J[ki,kj][i,j-iS[kj]+1]+=aiM*(A[kj]*GRADf(XJ_1,XiS[kj],XiE[kj])+B[kj]*GRADg(XJ_1,XiS[kj],XiE[kj]))
                                          #Gamma w.r.t. A and B           
                                          J[ki,kj][i,-2]+=aiM*f(XJ_1,XiS[kj],XiE[kj])
                                          J[ki,kj][i,-1]+=aiM*g(XJ_1,XiS[kj],XiE[kj])
                                          
                                    elif iE[ki]==M-1:
                                          J[ki,kj][i,j-iS[kj]]+=GAMMA[kj][M]*HOLD_dPSI_XJ
                                          J[ki,kj][i,(iE[kj]-2*iS[kj])+j+1]+=GAMMA[kj][M]*HOLD_dPSI_ZJ
                                          J[ki,kj][i,(iE[kj]-iS[kj])+2+j] = aiM
                                          
                                    else:
                                          J[ki,kj][i,(iE[kj]-iS[kj])+2+j] = aiM
                                          
                                    
                                               
                              j = j +1
                              
                        if ki == kj:
                              if iS[ki]<=i<=iE[ki]:         
                                    J[ki,kj][i,i-iS[ki]]+=-UINF*m.sin(ALPHA)
                                    J[ki,kj][i,(iE[ki]-2*iS[ki])+1+i]+=UINF*m.cos(ALPHA)
                                    
                              J[ki,kj][i,(iE[ki]-iS[ki])+M+2]=-1
                        
                        i= i +1     

            #Calculate indicies of non-zero entries in Rm+1
            entriesRM1=[0,1,2,M-2,M-1,M]
            for k in range(K):
                  for j in entriesRM1:
                        XJ=XPI[k][j]
                        if j<iS[k] or j>iE[k]:
                              if j==0:
                                    J[k,k][M,2*(iE[k]-iS[k])+2]=1
                                    
                              elif j==1:
                                    J[k,k][M,2*(iE[k]-iS[k])+3]=-2
                                    
                              elif j==2:
                                    J[k,k][M,2*(iE[k]-iS[k])+4]=1
                                    
                              elif j==M-2:
                                    J[k,k][M,-6]=-1
                                    
                              elif j==M-1:
                                    J[k,k][M,-5]=2
                                    
                              elif j==M:
                                    J[k,k][M,-4]=-1
                                    
                        if iS[k]<=j<=iE[k]:
                              if j==0:
                                    dRM1_dXI=A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k])
                                    dRM1_dA=f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB

                                    
                              elif j==1:
                                    dRM1_dXI=-2*(A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k]))
                                    dRM1_dA=-2*f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=-2*g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB

                              elif j==2:
                                    dRM1_dXI=A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k])
                                    dRM1_dA=f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=g(XJ,XiS[k],XiE[k])
                                    
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB
                                    
                              elif j==M-2:
                                    dRM1_dXI=-(A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k]))
                                    dRM1_dA=-f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=-g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB
                          
                              elif j==M-1:
                                    dRM1_dXI=2*(A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k]))
                                    dRM1_dA=2*f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=2*g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB
                        
                              elif j==M:
                                    dRM1_dXI=-(A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k]))
                                    dRM1_dA=-f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=-g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M,-2]+=dRM1_dA
                                    J[k,k][M,-1]+=dRM1_dB   

            #Calculate indicies of non-zero entries in Rm+2
            entriesRM2=[0,M]
            for k in range(K):      
                  for j in entriesRM2:
                        XJ=XPI[k][j]
                        
                        if j<iS[k] or j>iE[k]:
                              if j==0:
                                    J[k,k][M+1,2*(iE[k]-iS[k])+2]=1
                                    
                              elif j==M:
                                    J[k,k][M+1,-4]=1
                        
                        if iS[k]<=j<=iE[k]:
                              if j==0:
                                    dRM1_dXI=A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k])
                                    dRM1_dA=f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M+1,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M+1,-2]+=dRM1_dA
                                    J[k,k][M+1,-1]+=dRM1_dB
                                    
                                    
                              elif j==M:
                                    dRM1_dXI=A[k]*GRADf(XJ,XiS[k],XiE[k])+B[k]*GRADg(XJ,XiS[k],XiE[k])
                                    dRM1_dA=f(XJ,XiS[k],XiE[k])
                                    dRM1_dB=g(XJ,XiS[k],XiE[k])
                                    
                                    J[k,k][M+1,j-iS[k]]+=dRM1_dXI
                                    J[k,k][M+1,-2]=+dRM1_dA
                                    J[k,k][M+1,-1]=+dRM1_dB
                                    
            
            #Rm+3 to Rm +(iE-iS) +1 (Node Movement Compatability)
            for k in range(K):
                  i = iS[k]+1
                  while iS[k]<i<iE[k]:
                        XI=XPI[k][i]
                        ZI=ZPI[k][i]
                        XS=XP[k][i]
                        ZS=ZP[k][i]
                        ni = n[k][i]
                        dGCd_dXI,dGCd_dZI=GRADGCd(XI,ZI,XS,ZS,ni)
                        
                        J[k,k][M+1+i-iS[k],i-iS[k]]=dGCd_dXI
                        J[k,k][M+1+i-iS[k],i+iE[k]-2*iS[k]+1]=dGCd_dZI
                        i=i+1
                        
                  #Rm+2(iE-IS) to Rm+5(iE-IS) (geometric continuity)
                  J[k,k][M+(iE[k]-iS[k])+1,0]=1
                  J[k,k][M+(iE[k]-iS[k])+2,(iE[k]-iS[k])+1]=1
                  J[k,k][M+(iE[k]-iS[k])+3,(iE[k]-iS[k])]=1 
                  J[k,k][M+(iE[k]-iS[k])+4,2*(iE[k]-iS[k])+1]=1
            
            J=np.concatenate(([np.concatenate(J[k],axis=1) for k in range(K)]),axis=0)
            
            jactimeend=time.time()
            jactime=jactimeend-jactimestart
            JACTIME.append(jactime)
            JACCOUNT.append(1)
            return(J)


      start= time.clock()     
      sol = optimize.root(R,W0,method='hybr',jac=JAC,options={\
        'xtol': 1.49012e-09, 'maxfev': 15})
      end= time.clock() 
      print(end-start)   
      
      #Extract Results
      SOL=[0 for k in range(K)]
      XSOL=[0 for k in range(K)]
      ZSOL=[0 for k in range(K)]
      GAMMASOL=[0 for k in range(K)]
      PSISOL=[0 for k in range(K)]
      ASOL=[0 for k in range(K)]
      BSOL=[0 for k in range(K)]
      XNEW=[0 for k in range(K)]
      ZNEW=[0 for k in range(K)]
      QSPSOL=[0 for k in range(K)]
      GAMMANEW=[0 for k in range(K)]
      
      for k in range(K):
            startindex=sum(Wlen[:k])
            endindex=sum(Wlen[:k+1])
            SOL[k]=sol.x[startindex:endindex]
            XSOL[k]=SOL[k][0:(iE[k]-iS[k])+1]
            ZSOL[k]=SOL[k][(iE[k]-iS[k])+1:2*(iE[k]-iS[k])+2]
            GAMMASOL[k]=SOL[k][2*(iE[k]-iS[k])+2:(iE[k]-iS[k])+M+2]
            PSISOL[k]=SOL[k][(iE[k]-iS[k])+M+2]
            ASOL[k]=SOL[k][(iE[k]-iS[k])+M+3]
            BSOL[k]=SOL[k][(iE[k]-iS[k])+M+4]
            XNEW[k]=np.concatenate((XP[k][:iS[k]],XSOL[k],XP[k][iE[k]+1:]))
            ZNEW[k]=np.concatenate((ZP[k][:iS[k]],ZSOL[k],ZP[k][iE[k]+1:]))
            
            QSPSOL[k] = interp1d(XSP[k][iS[k]:iE[k]+1],\
                           np.reshape(GAMMASP[k][iS[k]:iE[k]+1],\
                          (len(GAMMASP[k][iS[k]:iE[k]+1]),)),kind='linear',\
                              fill_value='extrapolate')(XSOL[k])
                           
            Q = np.zeros_like(XSOL[k])
            i=0
            while i<len(XSOL[k]):
                  Q[i] = QSPSOL[k][i]\
                  +ASOL[k]*f(XSOL[k][i],XiS[k],XiE[k]) +\
                        BSOL[k]*g(XSOL[k][i],XiS[k],XiE[k])
                  i = i + 1
                  
            GAMMANEW[k]=np.concatenate((GAMMASOL[k][:iS[k]],Q,GAMMASOL[k][iS[k]:]))

      #Calculate CL
      CLNEW=[0 for k in range(K)]
      for k,i in np.ndindex(K,M):
            CLNEW[k] = CLNEW[k]+(GAMMANEW[k][i]+GAMMANEW[k][i])*PANEL_LENGTH[k][i]


      return(XNEW,ZNEW,GAMMANEW,PSISOL,CLNEW,[XSOL,ZSOL,GAMMASOL,QSPSOL,ASOL,BSOL]\
             ,[GAMMASP,GAMMAI],[RES,RESTIME,RTIME,RCOUNT,JACTIME,JACCOUNT])

  
      
def PSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M):
      '''Calculate the the unit streamfunction influence
         w.r.t the local panel coordinates
      '''
      if j==i and ki == kj:      
            PSI1 = (XIP-XJ_1P)*(1-np.log(R2))
      
            PSI2 = -PSI1 - (XIP-XJ_1P)*np.log(R2) + (XIP-XJ_1P)/2
      elif j==i-1 and ki == kj:
            
            PSI1 = XIP*(np.log(R1)-1)
      
            PSI2 = PSI1 - XIP*np.log(R1) + XIP/2
      elif i==0 and j==M-1 and ki == kj:
      
            PSI1 = XIP*(np.log(R1)-1)
      
            PSI2 = PSI1 - XIP*np.log(R1) + XIP/2      
      elif i==M and j==0 and ki == kj:
            
            PSI1 = (XIP-XJ_1P)*(1-np.log(R2))
      
            PSI2 = -PSI1 - (XIP-XJ_1P)*np.log(R2) + (XIP-XJ_1P)/2 
      else:     
            PSI1 = XIP*np.log(R1)-(XIP-XJ_1P)*np.log(R2) -XJ_1P + ZIP*(THETA2-THETA1)
            
            PSI2 = ((2*XIP-XJ_1P)*PSI1 + R2**2*np.log(R2) - R1**2*np.log(R1)\
                          + (XIP**2-(XIP-XJ_1P)**2)/2)/(XJ_1P)
            
      return(np.array((PSI1,PSI2)))
            
        
    
def GRADPSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M):
      '''Calculate the gradient of the unit streamfunction influence
         w.r.t the local panel coordinates
      '''
      gradPSI=np.asarray([np.zeros((1,3)) for i in range(2)])
      
      if j==i and ki == kj:
            #PSI1 derivatives
            dPSI1_dXIP= (1-m.log(R2)) - (XIP-XJ_1P)**2/(R2**2)
            
            dPSI1_dZIP= -ZIP*(XIP-XJ_1P)/(R2**2)
            
            dPSI1_dXJ_1P= (m.log(R2)-1) + (XIP-XJ_1P)**2/(R2**2)
            
            #PSI2 derivatives
            dPSI2_dXIP= -dPSI1_dXIP -m.log(R2) -(XIP-XJ_1P)**2/(R2**2) + 0.5
            
            dPSI2_dZIP= -dPSI1_dZIP -ZIP*(XIP-XJ_1P)/(R2**2)
            
            dPSI2_dXJ_1P= -dPSI1_dXJ_1P +m.log(R2) +(XIP-XJ_1P)**2/(R2**2) -0.5
            
      elif j==i-1 and ki == kj:
             #PSI1 derivatives
            dPSI1_dXIP= (m.log(R1)-1) +(XIP**2)/(R1**2)
            
            
            dPSI1_dZIP= (ZIP*XIP)/(R1**2)
            
            dPSI1_dXJ_1P= 0
            
            #PSI2 derivatives
            dPSI2_dXIP= dPSI1_dXIP -m.log(R1) -(XIP**2)/(R1**2) +0.5
            
            dPSI2_dZIP= dPSI1_dZIP -(ZIP*XIP)/(R1**2)
            
            dPSI2_dXJ_1P= 0
            
      elif i==0 and j==M-1 and ki == kj:
             #PSI1 derivatives
            dPSI1_dXIP= (m.log(R1)-1) +(XIP**2)/(R1**2)
            
            dPSI1_dZIP= (ZIP*XIP)/(R1**2)
            
            dPSI1_dXJ_1P= 0
            
            #PSI2 derivatives
            dPSI2_dXIP= dPSI1_dXIP -m.log(R1) -(XIP**2)/(R1**2) +0.5
            
            dPSI2_dZIP= dPSI1_dZIP -(ZIP*XIP)/(R1**2)
            
            dPSI2_dXJ_1P= 0
            
      elif i==M and j==0 and ki == kj:
            #PSI1 derivatives
            dPSI1_dXIP= (1-m.log(R2)) - (XIP-XJ_1P)**2/(R2**2)
            
            dPSI1_dZIP= -ZIP*(XIP-XJ_1P)/(R2**2)
            
            dPSI1_dXJ_1P= (m.log(R2)-1) + (XIP-XJ_1P)**2/(R2**2)
            
            #PSI2 derivatives
            dPSI2_dXIP= -dPSI1_dXIP -m.log(R2) -(XIP-XJ_1P)**2/(R2**2) + 0.5
            
            dPSI2_dZIP= -dPSI1_dZIP -ZIP*(XIP-XJ_1P)/(R2**2)
            
            dPSI2_dXJ_1P= -dPSI1_dXJ_1P +m.log(R2) +(XIP-XJ_1P)**2/(R2**2) -0.5
                
      else:
             #PSI1 derivatives
            dPSI1_dXIP= m.log(R1) +(XIP**2)/(R1**2) -m.log(R2)\
            -(XIP-XJ_1P)**2/(R2**2) -ZIP*((ZIP)/(R2**2) - (ZIP)/(R1**2))
            
            dPSI1_dZIP= (ZIP*XIP)/(R1**2) -ZIP*(XIP-XJ_1P)/(R2**2)\
            +(THETA2-THETA1) +ZIP*((XIP-XJ_1P)/(R2**2) - (XIP)/(R1**2))
            
            dPSI1_dXJ_1P= m.log(R2) +(XIP-XJ_1P)**2/(R2**2) -1 +(ZIP**2)/(R2**2)
            
            #PSI2 derivatives
            PSI1=PSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M)[0]
            PSI2=PSI(XIP,ZIP,XJ_1P,R1,R2,THETA1,THETA2,i,j,ki,kj,M)[1]
            
            dPSI2_dXIP= (2*PSI1 +(2*XIP-XJ_1P)*dPSI1_dXIP\
                         +(XIP-XJ_1P)*(2*m.log(R2)+1) -XIP*(2*m.log(R1)+1)\
                               + XJ_1P)/(XJ_1P)     
            
            dPSI2_dZIP= ((2*XIP-XJ_1P)*dPSI1_dZIP +ZIP*(2*m.log(R2)+1)\
                         -ZIP*(2*m.log(R1)+1))/(XJ_1P)
            
            dPSI2_dXJ_1P= (-PSI2 -PSI1 +(2*XIP-XJ_1P)*dPSI1_dXJ_1P\
                           -(XIP-XJ_1P)*(2*m.log(R2)+1) +(XIP-XJ_1P))/(XJ_1P)
            
      dPSI1=np.stack(([dPSI1_dXIP],[dPSI1_dZIP],[dPSI1_dXJ_1P]),axis=1)
      dPSI2=np.stack(([dPSI2_dXIP],[dPSI2_dZIP],[dPSI2_dXJ_1P]),axis=1)
      
      gradPSI[0],gradPSI[1]= dPSI1,dPSI2
      
      return(gradPSI)
            


def GRADT(XI,ZI,XJ,ZJ,XJ_1,ZJ_1,BETA):
      '''Calculate the gradient of the coordinate transform from global
         to panel reference frame
      '''
      gradT=np.asarray([np.zeros((3,1)) for i in range(6)])
      
      #XI' derivatives
      dXIP_dXI=m.cos(BETA)
      
      dXIP_dZI=m.sin(BETA)
      
      dXIP_dXJ= (ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((ZI-ZJ)*m.cos(BETA)\
                 -(XI-XJ)*m.sin(BETA)) - m.cos(BETA)
      
      dXIP_dZJ= (XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.sin(BETA)\
                 -(ZI-ZJ)*m.cos(BETA)) - m.sin(BETA)
      
      dXIP_dXJ_1= (ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.sin(BETA)\
                   -(ZI-ZJ)*m.cos(BETA))
      
      dXIP_dZJ_1= (XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((ZI-ZJ)*m.cos(BETA)\
                   -(XI-XJ)*m.sin(BETA))

      #ZI' derivatives
      
      dZIP_dXI=-m.sin(BETA)
      
      dZIP_dZI=m.cos(BETA)
      
      dZIP_dXJ= -(ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.cos(BETA)\
                  +(ZI-ZJ)*m.sin(BETA)) + m.sin(BETA)
      
      dZIP_dZJ= (XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.cos(BETA)\
                 +(ZI-ZJ)*m.sin(BETA)) - m.cos(BETA)
      
      dZIP_dXJ_1= (ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.cos(BETA)\
                   +(ZI-ZJ)*m.sin(BETA))
      
      dZIP_dZJ_1= -(XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XI-XJ)*m.cos(BETA)\
                    +(ZI-ZJ)*m.sin(BETA))
      
      #XJ+1' derivatives
      
      dXJ_1P_dXI=0
      
      dXJ_1P_dZI=0
      
      dXJ_1P_dXJ= (ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((ZJ_1-ZJ)*m.cos(BETA)\
                   -(XJ_1-XJ)*m.sin(BETA)) - m.cos(BETA)
      
      dXJ_1P_dZJ= (XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XJ_1-XJ)*m.sin(BETA)\
                   -(ZJ_1-ZJ)*m.cos(BETA)) - m.sin(BETA)
      
      dXJ_1P_dXJ_1= (ZJ_1-ZJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((XJ_1-XJ)*m.sin(BETA)\
                     -(ZJ_1-ZJ)*m.cos(BETA)) + m.cos(BETA)
      
      dXJ_1P_dZJ_1= (XJ_1-XJ)/((XJ_1-XJ)**2+(ZJ_1-ZJ)**2)*((ZJ_1-ZJ)*m.cos(BETA)\
                     -(XJ_1-XJ)*m.sin(BETA)) + m.sin(BETA)
      
      d_dXI=np.stack(([dXIP_dXI],[dZIP_dXI],[dXJ_1P_dXI]))
      d_dZI=np.stack(([dXIP_dZI],[dZIP_dZI],[dXJ_1P_dZI]))
      d_dXJ=np.stack(([dXIP_dXJ],[dZIP_dXJ],[dXJ_1P_dXJ]))
      d_dZJ=np.stack(([dXIP_dZJ],[dZIP_dZJ],[dXJ_1P_dZJ]))
      d_dXJ_1=np.stack(([dXIP_dXJ_1],[dZIP_dXJ_1],[dXJ_1P_dXJ_1]))
      d_dZJ_1=np.stack(([dXIP_dZJ_1],[dZIP_dZJ_1],[dXJ_1P_dZJ_1]))
      
      gradT[0]=d_dXI
      gradT[1]=d_dZI
      gradT[2]=d_dXJ
      gradT[3]=d_dZJ
      gradT[4]=d_dXJ_1
      gradT[5]=d_dZJ_1
      
      return(gradT)
      


#gamma shape function 1     
def f(X,XIS,XIE):
      return((np.power((XIE-X)/(XIE-XIS),2.0)))


#gamma shape function 2      
def g(X,XIS,XIE):
      return((np.power((XIS-X)/(XIS-XIE),2.0)))
      
      
def GRADf(X,XIS,XIE):
      return(-2*(XIE-X)/(XIE-XIS)**2)

def GRADg(X,XIS,XIE):
      return(-2*(XIS-X)/(XIS-XIE)**2)
      
      

def GCd(XI,ZI,XS,ZS,n):
      eps1 = 10**(-15)
      eps2 = 5*10**(-15)
      
      t = (n[0]*(XI-XS) + n[1]*(ZI-ZS))\
                        /(n[0]**2 + n[1]**2)
      GCx = XS-XI + t*n[0]
      GCz = ZS-ZI + t*n[1]
      if GCx==0.0 and GCz==0.0:
            XI = XI + eps1
            ZI = ZI + eps2
            t = (n[0]*(XI-XS) + n[1]*(ZI-ZS))\
                        /(n[0]**2 + n[1]**2)
            GCx = XS-XI + t*n[0]
            GCz = ZS-ZI + t*n[1]

      gc=m.sqrt(GCx**2 + GCz**2)
      return(gc)


def GRADGCd(XI,ZI,XS,ZS,n):
      eps1 = 10**(-15)
      eps2 = 5*10**(-15)
      
      t = (n[0]*(XI-XS) + n[1]*(ZI-ZS))\
                        /(n[0]**2 + n[1]**2)
      GCx = XS-XI + t*n[0]
      GCz = ZS-ZI + t*n[1]
      if GCx==0.0 and GCz==0.0:
            XI = XI + eps1
            ZI = ZI + eps2
            t = (n[0]*(XI-XS) + n[1]*(ZI-ZS))\
                        /(n[0]**2 + n[1]**2)
            
      dt_dXI=n[0]/(n[0]**2+n[1]**2)
      dt_dZI=n[1]/(n[0]**2+n[1]**2)
      
      dGC_dXI=((XS-XI+t*n[0])*(n[0]*dt_dXI-1) +(ZS-ZI+t*n[1])*n[1]*dt_dXI)/GCd(XI,ZI,XS,ZS,n)
      dGC_dZI=((XS-XI+t*n[0])*n[0]*dt_dZI +(ZS-ZI+t*n[1])*(n[1]*dt_dZI-1))/GCd(XI,ZI,XS,ZS,n)
      
      return(dGC_dXI,dGC_dZI)
      
