#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:32:20 2023

@author: lheller
"""
#import sys
#from importlib import import_module
import math
import numpy as np
from numpy.linalg import inv
from numpy import sqrt as sqrt
import sys
from importlib import import_module
from effective_elastic_constants_functions import  *
from projlib import  *
from crystallography_functions import *
from scipy import interpolate
class calcMT:  
    def __init__(self,modulepath = "/home/lheller/python"): 
        pass
        #self.importmodules = True # "True" Or "False"
        #if self.importmodules:
        #    sys.path.append(modulepath)
        #    modules=['crystallography_functions','effective_elastic_constants_functions','projlib']
        #    for module in modules:
        #        self.importedModule = import_module(module)
        
        
    #defining class methods for input parameters 
    def setLatticeMatrices(self,LA,LM):  
        self.LA=LA
        self.LM=LM  
    def setTensorsOfElastCompliance(self,STA,STM):  
        self.STA=STA
        self.STM=STM 
    def setLatticeCorrespondence(self,Cd,CId=None):  
        self.Cd=Cd
        self.CId=CId
    def setLatticeCorrespondencePlanes(self,Cp,CIp=None):  
        self.Cp=Cp
        self.CIp=CIp
    def setLatticeCorrespondence(self,Cd,CId=None):  
        self.Cd=Cd
        self.CId=CId
    def setOrientations(self,oris):  
        self.oris=oris
    def setStress(self,stress):
        self.stress=stress
    def setStressSpace(self):
        self.stressSpace=np.tile(self.stress, (self.oris.shape[1],self.Cd.shape[2],1)).T

    #defining class methods for computations
    #directional modulus
    def dirModulus(self,ST,oris=None):   
        if oris is None:
            oris=self.oris
        dirM=[]
        for ori in oris.T:
            #print(ori)
            dirM.append(directional_modulus_onedir(ST,ori))
        return np.array(dirM)
    #deformation along straindir due to uniaxial loading along a loaddir
    def dirDeformation(self,UniaxialStress,loaddir,ST,straindir):           
        if type(loaddir)==list:
            loaddir=np.array(loaddir)
        loaddir=loaddir/np.sqrt(loaddir.dot(loaddir))
        if type(straindir)==list:
            straindir=np.array(straindir)
        straindir=straindir/np.sqrt(straindir.dot(straindir))
        #generate transformation matrix such that X2Loaddir such that X becomes loaddir
        v2=perpendicular_vector(loaddir)
        v3=np.cross(loaddir,v2)
        X2Loaddir=np.vstack((loaddir,v2,v3)).T
        #Get deformation gradients for elastically distorted lattices and all correspondence variants
        #Applied uniaxial stress along X direction
        #UniaxialStress=500
        applied_stress=np.zeros((3,3))
        applied_stress[0,0]=UniaxialStress
        #Rotating stress tensor such that the uniaxial stress is along loaddir
        applied_stress_rot=np.einsum('ia,jb,ab->ij',X2Loaddir, X2Loaddir,applied_stress)
        Strain = np.einsum('ijkl,kl',ST,applied_stress_rot)
        F=Strain+np.eye(3)
        StrainDir=(np.sqrt(straindir.dot(F.T.dot(F.dot(straindir))))-1)
        return StrainDir
    def dirDefOris(self,UniaxialStress,oris,ST,straindir):
        dirDefOris=[]
        for k in range(oris.shape[1]):
            loaddir=oris[:,k]
            dirDefOris.append(self.dirDeformation(UniaxialStress,loaddir,ST,straindir))
        self.currentDirDefOris=np.array(dirDefOris)
    def getStressfreeDefGrads(self):
        F_AM, U_AM, Q_M, T_MA, T_AM=def_gradient_stressfree(self.Cd,self.LA, self.LM, CId=self.CId)
        keys=['F_AM', 'U_AM', 'Q_M', 'T_MA', 'T_AM']
        StressfreeDefGrads={}
        for key in keys:
            exec(f"StressfreeDefGrads['{key}']={key}")
        self.StressfreeDefGrads=StressfreeDefGrads
        
    def getStressfreeRefs(self,oris=None):
        if oris is None:
            oris=self.oris
        else:
            self.oris=oris
        #computations of epsilon_2, lambda_2, and orientation dependence of transformation strain for all variants
        #Transformation strain and reference epsilon_2 for stress-free case
        #stress-free deformation gradients and related operators
        F_AM, U_AM, Q_M, T_MA, T_AM=def_gradient_stressfree(self.Cd,self.LA, self.LM, CId=self.CId)
        StressfreeRefs={}
        for space in ['austenite space','martensite space']:
            TransformationStrain=[]
            EigVals=[]
            EigVecs=[]
            epsilon_2=[]
            lambda_2=[]
            strain=[]
            for var in range(0,F_AM.shape[2]):
                if space=='martensite space':
                    #in Martensite space
                    #def. grad in mart space
                    F_AM_M=T_AM[:,:,var].dot(F_AM[:,:,var].dot(T_AM[:,:,var].T))
                    F=T_AM[:,:,var].dot(F_AM[:,:,var].dot(T_AM[:,:,var].T))
                    U_AM_M = scipy.linalg.sqrtm(F_AM_M.T.dot(F_AM_M))
                    Q_M_M = F_AM_M.dot(inv(U_AM_M));
                    
                    #F_AM_M[:,:,var]=T_AM[:,:,var].dot(F_AM[:,:,var].dot(T_AM[:,:,var].T))
                    #U_AM_M[:,:,var]=U
                    #Q_MM[:,:,var]=Q
                    
                else:
                    F=F_AM[:,:,var]
                TransformationStrain.append(np.sqrt(np.sum(oris*(F.T.dot(F.dot(oris))),axis=0))-1) 
                #right Cauchy-Green deformation tensor https://www.comsol.com/multiphysics/analysis-of-deformation
                C=F.T.dot(F)
                #Get eigenvalues (squares of principal stretches) and eigenvectors
                D,V = np.linalg.eig(C)
                #matmul(C,V) - matmul(V,D*np.eye(3))
                Idxs = np.argsort(D)
                #Sorted eigenvalues and eigenvectors
                EigVals.append(D[Idxs])
                EigVecs.append(V[:,Idxs])
                #Eigenstrains using Green-Lagrange strain tensor
                Eps=0.5*(EigVals[-1]-1)
                strain.append(0.5*(EigVals[-1]-1))
                epsilon_2.append(Eps[1])
                lambda_2.append(EigVals[-1][1])


            #Convert to array 12 x number of  orientations
            TransformationStrain=np.array(TransformationStrain)    
            StressfreeRefs[space]={}
            
            keys=['F_AM', 'U_AM', 'Q_M', 'F_AM_M', 'U_AM_M', 'Q_M_M', 'T_MA', 'T_AM','TransformationStrain','EigVals','EigVecs','epsilon_2','lambda_2','strain']
            for key in keys:
                try:
                    exec(f"StressfreeRefs['{space}']['{key}']={key}")
                except:
                    pass
        self.StressfreeRefs=StressfreeRefs
    
    def solveCompProblemWithStress(self,loaddir, UniaxialStress,var=None,STA=None):
        #solve compatibility equation for elastically stressed lattices
        if type(loaddir)==list:
            loaddir=np.array(loaddir)
        loaddir=loaddir/np.sqrt(loaddir.dot(loaddir))
        #generate transformation matrix such that X2Loaddir such that X becomes loaddir
        v2=perpendicular_vector(loaddir)
        v3=np.cross(loaddir,v2)
        X2Loaddir=np.vstack((loaddir,v2,v3)).T
        #Get deformation gradients for elastically distorted lattices and all correspondence variants
        #Applied uniaxial stress along X direction
        #UniaxialStress=500
        applied_stress=np.zeros((3,3))
        applied_stress[0,0]=UniaxialStress
        #Rotating stress tensor such that the uniaxial stress is along loaddir
        applied_stress_rot=np.einsum('ia,jb,ab->ij',X2Loaddir, X2Loaddir,applied_stress)
        #print(applied_stress_rot)
        ##assign self parameters self.X to local variable X
        #keys=['Cd','CId','LA','LM','STA','STM']
        #for key in keys:
        #    print(f"{key}=self.{key}")
        #    exec(f"{key}=self.{key}")
        ##CId=self.CId
        #print(self.CId)
        #print(CId)
        if STA is None:
            STA=self.STA
        if var is None:
            var=list(range(Cd.shape[2]))
        #print(STA[0,0,0])
        #solve compatibility equation for elastically stressed lattices
        if self.CId is None: 
            F_AMStress, U_AMStress, Q_MStress, T_MA, T_AM, LAStress, LMStress, Parent_strain, Product_strain, F_parent, F_product=def_gradient(self.Cd[:,:,var],self.LA, self.LM,StressT=applied_stress_rot,STA=STA,STM=self.STM,CId=self.CId)
        else:
            F_AMStress, U_AMStress, Q_MStress, T_MA, T_AM, LAStress, LMStress, Parent_strain, Product_strain, F_parent, F_product=def_gradient(self.Cd[:,:,var],self.LA, self.LM,StressT=applied_stress_rot,STA=STA,STM=self.STM,CId=self.CId[:,:,var])
        TrStrain,epsilon_2,lambda_2,epsilon_1,lambda_1,epsilon_3,lambda_3=[],[],[],[],[],[],[]
        ALLEigVals=[]
        ALLEigVecs=[]
        ALLTrStrain=[]
        ALLEigStrain=[]
        ALLEigVecEps2=[]
        Strain_inA_along_Eps2=[]
        Strain_inM_along_Eps2=[]
        #print(var)
        #print(range(F_AMStress.shape[2]))
        #print(F_AMStress.shape)
        for vari in range(F_AMStress.shape[2]):
            F=F_AMStress[:,:,vari];

            #right Cauchy-Green deformation tensor https://www.comsol.com/multiphysics/analysis-of-deformation
            C=F.T.dot(F)
            #Get eigenvalues (squares of principal stretches) and eigenvectors
            D,V = np.linalg.eig(C)
            #matmul(C,V) - matmul(V,D*np.eye(3))
            Idxs = np.argsort(D)
            #Sorted eigenvalues and eigenvectors          
            EigVals=D[Idxs]
            ALLEigVals.append(EigVals)
            EigVecs = V[:,Idxs]
            ALLEigVecs.append(EigVecs)
            #Eigenstrains using Green-Lagrange strain tensor
            Eps=0.5*(EigVals-1)
            ALLEigStrain.append(Eps)
            epsilon_2.append(Eps[1])
            lambda_2.append(EigVals[1])
            epsilon_1.append(Eps[0])
            lambda_1.append(EigVals[0])
            epsilon_3.append(Eps[2])
            lambda_3.append(EigVals[2])
            ALLEigVecEps2.append(EigVecs[:,1])
            # Green-Lagrange strain tensor
            Strain=1./2.*(F.T.dot(F)-np.eye(3))
            #Transformation strain along loading direction
            TrStrain.append((np.sqrt(loaddir.dot(F.T.dot(F.dot(loaddir))))-1)*100)
            #Strain of parent along epsilon_2
            eps2vec_inA=EigVecs[:,1]
            Strain_inA_along_Eps2.append((np.sqrt(eps2vec_inA.dot(F_parent.T.dot(F_parent.dot(eps2vec_inA))))-1)*100)
            #Strain of product along epsilon_2
            eps2vec_inM=T_AM[:,:,vari].dot(EigVecs[:,1])
            Strain_inM_along_Eps2.append((np.sqrt(eps2vec_inM.dot(F_product[:,:,vari].T.dot(F_product[:,:,vari].dot(eps2vec_inM))))-1)*100)
            
            
        CurrentCompWithStress={}
        keys='F_AMStress U_AMStress Q_MStress T_MA T_AM LAStress LMStress Parent_strain Product_strain F_parent F_product \
              ALLEigVals ALLEigVecs ALLTrStrain ALLEigStrain TrStrain epsilon_2 lambda_2  epsilon_1 lambda_1  epsilon_3 lambda_3 ALLEigVecEps2 Strain_inA_along_Eps2 Strain_inM_along_Eps2'        
        for key in keys.split():
            exec(f"CurrentCompWithStress['{key}']={key}")
        #print(CurrentCompWithStress.keys())
        self.CurrentCompWithStress=CurrentCompWithStress
            
    def solveCompProblemWithStressOris(self,idxs=None,oris=None,stress=None,stressSpace=None,vars=None):
        #assign self parameters self.X to local variable X
        if oris is None:
            oris=self.oris
        else:
            self.oris=oris
        if stress is None:
            stress=self.stress
        else:
            self.stress=stress            
        #keys='Cd CId LA LM STA STM oris'
        #for key in keys.split():
        #    exec(f"{key}=self.{key}")
        keysS=['epsilon_2','lambda_2','epsilon_1','lambda_1','epsilon_3','lambda_3','TrStrain','Strain_inA_along_Eps2', 'Strain_inM_along_Eps2']
        Solution={}
        if stressSpace is None:
            self.setStressSpace()
            #self.stressSpace=np.tile(self.stress, (self.oris.shape[1],self.Cd.shape[2],1)).T
            Solution['StressSpace']=self.stressSpace
        else:
            self.stressSpace=stressSpace
            Solution['StressSpace']=self.stressSpace
        for key in keysS:
            #we create an array to store variables computed for all stresses, variants, and orientations
            #it is indexed [i,j,k] - i for stress, j for variant, k for orientation
            #Solution[key]=np.empty((StressSpace.shape[0],Cd.shape[2],oris.shape[1]))
            Solution[key]=np.empty((Solution['StressSpace'].shape))
            Solution[key][:]=np.nan
        #Storing vectors
        keysV=['ALLEigVecEps2','ALLEigVals']
        for key in keysV:
            Solution[key] = np.empty((Solution['StressSpace'].shape + tuple([3])))
            Solution[key][:]=np.nan
        #Storing matrices
        keysM=['F_AMStress', 'U_AMStress', 'Q_MStress','T_MA', 'T_AM']#,'LAStress','LMStress']
        for key in keysM:
            Solution[key] = np.empty((Solution['StressSpace'].shape + tuple([3]) + tuple([3])))
            Solution[key][:]=np.nan
        keysML=['ALLEigVecs','LAStress','LMStress']
        for key in keysML:
            Solution[key] = np.empty((Solution['StressSpace'].shape + tuple([3]) + tuple([3])))
            Solution[key][:]=np.nan
        if idxs is None:
            idxs=np.where(np.zeros(Solution['epsilon_2'].shape)==0)
        if vars==None:
            vars=list(range(Solution['StressSpace'].shape[1]))
        for i,j,k in zip(idxs[0],idxs[1],idxs[2]):
            if j in vars:
                loaddir=oris[:,k]
                UniaxialStress = Solution['StressSpace'][i,j,k]
                #print(loaddir)
                #print(UniaxialStress)
                
                self.solveCompProblemWithStress(loaddir,UniaxialStress,var=[j])
                for key in keysS:
                    exec(f"Solution['{key}'][i,j,k]=self.CurrentCompWithStress['{key}'][0]")
                #for key in 'Strain_inA_along_Eps2 Strain_inM_along_Eps2'.split():
                #    Solution[key][i,j,k]=self.CurrentCompWithStress[key][0]
                for key in keysV:
                    exec(f"Solution['{key}'][i,j,k,:]=self.CurrentCompWithStress['{key}'][0]")
    
                #Solution['eigvec_epsilon_2'][i,j,k,:]=self.CurrentCompWithStress['ALLEigVecEps2'][0]
                for key in keysM:
                    #exec(f"print({key}.shape:self.CurrentCompWithStress['{key}'].shape")
                    exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}'][:,:,0]")
                for key in keysML:
                    #exec(f"print({key}.shape:self.CurrentCompWithStress['{key}'].shape")
                    if key=='ALLEigVecs':
                        exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}'][0]")
                    else:
                        exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}']")
        self.CurrentCompWithStressOris=Solution
    def solveCompProblemWithCpOris(self,idxs=None,oris=None,stress=None,CPspace=None,vars=None):
        S_Af=compliance_from_tensor2voight_notation(self.STA)
        C_Af=np.linalg.inv(S_Af)
        C11i=C_Af[0,0]
        C12i=C_Af[0,1]
        C44i=C_Af[3,3]
        
        STA=[]
        
        #assign self parameters self.X to local variable X
        if oris is None:
            oris=self.oris
        else:
            self.oris=oris
        if stress is None:
            stress=self.stress
        else:
            self.stress=stress            
        #keys='Cd CId LA LM STA STM oris'
        #for key in keys.split():
        #    exec(f"{key}=self.{key}")
        #keys=['epsilon_2','lambda_2','TrStrain']
        keysS=['epsilon_2','lambda_2','epsilon_1','lambda_1','epsilon_3','lambda_3','TrStrain','Strain_inA_along_Eps2', 'Strain_inM_along_Eps2']
        Solution={}
        if CPspace is None:
            self.CPspace=np.tile(self.CP, (self.oris.shape[1],self.Cd.shape[2],1)).T
            Solution['CPspace']=self.CPspace
        else:
            self.CPspace=CPspace
            Solution['CPspace']=self.CPspace
        for key in keysS:
            #we create an array to store variables computed for all stresses, variants, and orientations
            #it is indexed [i,j,k] - i for stress, j for variant, k for orientation
            #Solution[key]=np.empty((StressSpace.shape[0],Cd.shape[2],oris.shape[1]))
            Solution[key]=np.empty((Solution['CPspace'].shape))
            Solution[key][:]=np.nan
            
        keysC = ['C11','C12','C44'] 
        for key in keysC:
            #we create an array to store variables computed for all stresses, variants, and orientations
            #it is indexed [i,j,k] - i for stress, j for variant, k for orientation
            #Solution[key]=np.empty((StressSpace.shape[0],Cd.shape[2],oris.shape[1]))
            Solution[key]=np.empty((Solution['CPspace'].shape))
            Solution[key][:]=np.nan

        #Storing vectors
        keysV=['ALLEigVecEps2','ALLEigVals']
        for key in keysV:
            Solution[key] = np.empty((Solution['CPspace'].shape + tuple([3])))
            Solution[key][:]=np.nan
        #Storing matrices
        keysM=['F_AMStress','U_AMStress', 'Q_MStress','T_MA', 'T_AM']#,'LAStress','LMStress']
        for key in keysM:
            Solution[key] = np.empty((Solution['CPspace'].shape + tuple([3]) + tuple([3])))
            Solution[key][:]=np.nan
        keysML=['ALLEigVecs','LAStress','LMStress']
        for key in keysML:
            Solution[key] = np.empty((Solution['CPspace'].shape + tuple([3]) + tuple([3])))
            Solution[key][:]=np.nan
        if idxs is None:
            idxs=np.where(np.zeros(Solution['epsilon_2'].shape)==0)
        if vars==None:
            vars=list(range(Solution['CPspace'].shape[1]))
        for i,j,k in zip(idxs[0],idxs[1],idxs[2]):
            if j in vars:
                loaddir=oris[:,k]
                UniaxialStress = self.stress  
                STAijk,parent_elastic_constants=self.get_STA(Solution['CPspace'][i,j,k],C11i,C12i,C44i,output='STAC')
                STA.append(STAijk)
                
                #print(loaddir)
                #print(UniaxialStress)
                #print(j)
                self.solveCompProblemWithStress(loaddir,UniaxialStress,var=[j],STA=STA[-1])
                Solution['C11'][i,j,k]=parent_elastic_constants['11']
                Solution['C12'][i,j,k]=parent_elastic_constants['12']
                Solution['C44'][i,j,k]=parent_elastic_constants['44']
                for key in keysS:
                    exec(f"Solution['{key}'][i,j,k]=self.CurrentCompWithStress['{key}'][0]")
                #for key in 'Strain_inA_along_Eps2 Strain_inM_along_Eps2'.split():
                #    Solution[key][i,j,k]=self.CurrentCompWithStress[key][0]
                for key in keysV:
                    exec(f"Solution['{key}'][i,j,k,:]=self.CurrentCompWithStress['{key}'][0]")
    
                #Solution['eigvec_epsilon_2'][i,j,k,:]=self.CurrentCompWithStress['ALLEigVecEps2'][0]
                for key in keysM:
                    #exec(f"print({key}.shape:self.CurrentCompWithStress['{key}'].shape")
                    exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}'][:,:,0]")
                for key in keysML:
                    #exec(f"print({key}.shape:self.CurrentCompWithStress['{key}'].shape")
                    #exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}'][0]")
                    #print(self.CurrentCompWithStress[key])
                    #print(key)
                    if key=='ALLEigVecs':
                        exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}'][0]")
                    else:
                        exec(f"Solution['{key}'][i,j,k,:,:]=self.CurrentCompWithStress['{key}']")

        self.STAspace=STA
        self.CurrentCompWithCpOris=Solution
    def findRootsEpsilon2vsStress(self,idxs=None):
        Solution=self.CurrentCompWithStressOris
        #Find where epsilon_2 intersects zero and refine the root search for sigma where epsilon_2=0
        if False:
            FirstBelowZeroIdxs=np.argmax(np.select([Solution['epsilon_2']<0],[Solution['epsilon_2']],-np.inf),
                                         axis=0,keepdims=True)
            FirstEpsBelowZero=np.take_along_axis(Solution['epsilon_2'], FirstBelowZeroIdxs, 0)[0,:,:]
            FirstStressEpsBelowZero = np.take_along_axis(Solution['StressSpace'], FirstBelowZeroIdxs, 0)[0,:,:]
    
            FirstAboveZeroIdxs=np.argmin(np.select([Solution['epsilon_2']>0],[Solution['epsilon_2']],np.inf),
                                         axis=0,keepdims=True)
            FirstEpsAboveZero=np.take_along_axis(Solution['epsilon_2'], FirstAboveZeroIdxs, 0)[0,:,:]
            FirstStressEpsAboveZero = np.take_along_axis(Solution['StressSpace'], FirstAboveZeroIdxs, 0)[0,:,:]
        else:
            FirstAboveZeroIdxs=np.argmax(np.select([Solution['epsilon_2']<0],[Solution['epsilon_2']],np.inf),
                             axis=0,keepdims=True)
            FirstEpsAboveZero=np.take_along_axis(Solution['epsilon_2'], FirstAboveZeroIdxs, 0)[0,:,:]
            FirstStressEpsAboveZero = np.take_along_axis(Solution['StressSpace'], FirstAboveZeroIdxs, 0)[0,:,:]
            FirstBelowZeroIdxs=FirstAboveZeroIdxs-1
            #FirstBelowZeroIdxs=np.select([FirstBelowZeroIdxs<0],[FirstBelowZeroIdxs],0)
            FirstEpsBelowZero=np.take_along_axis(Solution['epsilon_2'], FirstBelowZeroIdxs, 0)[0,:,:]
            FirstStressEpsBelowZero = np.take_along_axis(Solution['StressSpace'], FirstBelowZeroIdxs, 0)[0,:,:]

        #exclude directions where epsilon_2 vs. stress does not cross zero and those for which deformation energy is negative, i.e. negative transformation strain for tension 
        if idxs is None:
            idxs=np.where((~np.all(Solution['epsilon_2']>0,axis=0))*
                 (~np.all(Solution['epsilon_2']<0,axis=0))*
                 (np.sign(self.StressfreeRefs['austenite space']['TransformationStrain'])==np.sign(Solution['StressSpace'][-1,:,:])))
#        idxs=np.where((~np.all(Solution['epsilon_2']>0,axis=0))*
#             (~np.all(Solution['epsilon_2']<0,axis=0))*
#             (np.sign(self.StressfreeRefs['austenite space']['TransformationStrain'])==1))
        #print(idxs)
        deltaStress=FirstStressEpsAboveZero-FirstStressEpsBelowZero
        deltaEps=FirstEpsAboveZero-FirstEpsBelowZero
        Solution['FirstStressEpsBelowZero']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['FirstStressEpsAboveZero']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['deltaEps']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['deltaStress']=np.empty(Solution['epsilon_2'][0,:,:].shape)

        Solution['FirstStressEpsBelowZero'][:,:]=np.nan
        Solution['FirstStressEpsAboveZero'][:,:]=np.nan
        Solution['deltaEps'][:,:]=np.nan
        Solution['deltaStress'][:,:]=np.nan
        
        Solution['FirstStressEpsBelowZero'][idxs[0],idxs[1]]=FirstStressEpsBelowZero[idxs[0],idxs[1]]
        Solution['FirstStressEpsAboveZero'][idxs[0],idxs[1]]=FirstStressEpsAboveZero[idxs[0],idxs[1]]
        Solution['deltaEps'][idxs[0],idxs[1]]=deltaEps[idxs[0],idxs[1]]
        Solution['deltaStress'][idxs[0],idxs[1]]=deltaStress[idxs[0],idxs[1]]
        

        
        #Solution['FirstStressEpsBelowZero']=FirstStressEpsBelowZero
        #Solution['FirstStressEpsAboveZero']=FirstStressEpsAboveZero
        #Solution['deltaEps']=deltaEps
        #Solution['deltaStress']=deltaStress

        
        Solution['CriticalStressSensitivity']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['CriticalStressSensitivity'][:,:]=np.nan
        Solution['CriticalStressSensitivity'][idxs[0],idxs[1]]=deltaEps[idxs[0],idxs[1]]/deltaStress[idxs[0],idxs[1]]
        Solution['CriticalStressSensitivity']=deltaEps/deltaStress
        Solution['CriticalStress']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['CriticalStress'][:,:]=np.nan
        EpsAtZeroStress=FirstEpsAboveZero-Solution['CriticalStressSensitivity']*FirstStressEpsAboveZero
        Solution['CriticalStress'][idxs[0],idxs[1]]=-1*EpsAtZeroStress[idxs[0],idxs[1]]/Solution['CriticalStressSensitivity'][idxs[0],idxs[1]]
        self.CurrentCompWithStressOris=Solution
    def findRootsEpsilon2vsCp(self,idxs=None):
        Solution=self.CurrentCompWithCpOris
        if False:
            FirstBelowZeroIdxs=np.argmax(np.select([Solution['epsilon_2']<0],[Solution['epsilon_2']],-np.inf),
                                         axis=0,keepdims=True)
            FirstEpsBelowZero=np.take_along_axis(Solution['epsilon_2'], FirstBelowZeroIdxs, 0)[0,:,:]
            FirstCpEpsBelowZero = np.take_along_axis(Solution['CPspace'], FirstBelowZeroIdxs, 0)[0,:,:]
    
            FirstAboveZeroIdxs=np.argmin(np.select([Solution['epsilon_2']>0],[Solution['epsilon_2']],np.inf),
                                         axis=0,keepdims=True)
            FirstEpsAboveZero=np.take_along_axis(Solution['epsilon_2'], FirstAboveZeroIdxs, 0)[0,:,:]
            FirstCpEpsAboveZero = np.take_along_axis(Solution['CPspace'], FirstAboveZeroIdxs, 0)[0,:,:]
        else:
            FirstAboveZeroIdxs=np.argmax(np.select([Solution['epsilon_2']<0],[Solution['epsilon_2']],np.inf),
                             axis=0,keepdims=True)
            FirstEpsAboveZero=np.take_along_axis(Solution['epsilon_2'], FirstAboveZeroIdxs, 0)[0,:,:]
            FirstCpEpsAboveZero = np.take_along_axis(Solution['CPspace'], FirstAboveZeroIdxs, 0)[0,:,:]           
        
            FirstBelowZeroIdxs=FirstAboveZeroIdxs-1
            #FirstBelowZeroIdxs=np.select([FirstBelowZeroIdxs<0],[FirstBelowZeroIdxs],0)
            FirstEpsBelowZero=np.take_along_axis(Solution['epsilon_2'], FirstBelowZeroIdxs, 0)[0,:,:]
            FirstCpEpsBelowZero = np.take_along_axis(Solution['CPspace'], FirstBelowZeroIdxs, 0)[0,:,:]


        #exclude directions where epsilon_2 vs. stress does not cross zero and those for which deformation energy is negative, i.e. negative transformation strain for tension 
        if idxs is None:
            idxs=np.where((~np.all(Solution['epsilon_2']>0,axis=0))*
                 (~np.all(Solution['epsilon_2']<0,axis=0))*
                 (np.sign(self.StressfreeRefs['austenite space']['TransformationStrain'])==np.sign(self.stress)))
#        idxs=np.where((~np.all(Solution['epsilon_2']>0,axis=0))*
#             (~np.all(Solution['epsilon_2']<0,axis=0))*
#             (np.sign(self.StressfreeRefs['austenite space']['TransformationStrain'])==1))
        #print(idxs)
        deltaCp=FirstCpEpsAboveZero-FirstCpEpsBelowZero
        deltaEps=FirstEpsAboveZero-FirstEpsBelowZero
        Solution['FirstCpEpsBelowZero']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['FirstCpEpsBelowZero'][:,:]=np.nan
        Solution['FirstCpEpsBelowZero'][idxs[0],idxs[1]]=FirstCpEpsBelowZero[idxs[0],idxs[1]]
        Solution['FirstCpEpsAboveZero']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['FirstCpEpsAboveZero'][:,:]=np.nan
        Solution['FirstCpEpsAboveZero'][idxs[0],idxs[1]]=FirstCpEpsAboveZero[idxs[0],idxs[1]]
        Solution['deltaEps']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['deltaEps'][:,:]=np.nan
        Solution['deltaEps'][idxs[0],idxs[1]]=deltaEps[idxs[0],idxs[1]]
        Solution['deltaCp']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['deltaCp'][:,:]=np.nan
        Solution['deltaCp'][idxs[0],idxs[1]]=deltaCp[idxs[0],idxs[1]]
        #Solution['FirstCpEpsBelowZero']=FirstCpEpsBelowZero
        #Solution['FirstCpEpsAboveZero']=FirstCpEpsAboveZero
        #Solution['deltaEps']=deltaEps
        #Solution['deltaCp']=deltaCp
        Solution['CriticalCpSensitivity']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['CriticalCpSensitivity'][:,:]=np.nan
        Solution['CriticalCpSensitivity'][idxs[0],idxs[1]]=deltaEps[idxs[0],idxs[1]]/deltaCp[idxs[0],idxs[1]]
        Solution['CriticalCpSensitivity']=deltaEps/deltaCp
        Solution['CriticalCp']=np.empty(Solution['epsilon_2'][0,:,:].shape)
        Solution['CriticalCp'][:,:]=np.nan
        EpsAtZeroCp=FirstEpsAboveZero-Solution['CriticalCpSensitivity']*FirstCpEpsAboveZero
        Solution['CriticalCp'][idxs[0],idxs[1]]=-1*EpsAtZeroCp[idxs[0],idxs[1]]/Solution['CriticalCpSensitivity'][idxs[0],idxs[1]]
        self.CurrentCompWithCpOris=Solution


    def refineStressspace(self,nsteps=20,Solution=None):
        #refine roots found by find_roots_epsilon_2_vs_stress
        #By refining StressSpace for each variant and orientation
        if Solution is None:
            Solution=self.CurrentCompWithStressOris
        
        RefinedStressSpace=np.array([Solution['FirstStressEpsBelowZero'],Solution['FirstStressEpsAboveZero']])
        #refined stress space around intersection of epsilon_2 with 0
        #nsteps=10
        x = np.array([0,1])
        f = interpolate.interp1d(x, RefinedStressSpace, axis=0)
        xnew = np.linspace(0,1,nsteps)
        Solution['StressSpace'] = f(xnew)
        return Solution['StressSpace']
    
    def refineCpspace(self,nsteps=20,Solution=None):
        #refine roots found by find_roots_epsilon_2_vs_stress
        #By refining StressSpace for each variant and orientation 
        if Solution is None:
            Solution=self.CurrentCompWithCpOris
        RefinedCpSpace=np.array([Solution['FirstCpEpsBelowZero'],Solution['FirstCpEpsAboveZero']])
        #refined stress space around intersection of epsilon_2 with 0
        #nsteps=10
        x = np.array([0,1])
        f = interpolate.interp1d(x, RefinedCpSpace, axis=0)
        xnew = np.linspace(0,1,nsteps)
        Solution['CPSpace'] = f(xnew)
        return Solution['CPSpace']
    
    def get_STA(self,CPi,C11i,C12i,C44i,output='STA'):
        if self.softeningmethod=='symsoft':
            Cpi=(C11i-C12i)/2
            dCp=(Cpi-CPi)
            C11=C11i-dCp*1
            C12=C12i+dCp*1
            C44=C44i
        elif self.softeningmethod=='c12hardening':
            Cpi=(C11i-C12i)/2
            dCp=(Cpi-CPi)
            C11=C11i-dCp*0
            C12=C12i+dCp*2
            C44=C44i     
        elif self.softeningmethod=='c11softening':
            Cpi=(C11i-C12i)/2
            dCp=(Cpi-CPi)
            C11=C11i-dCp*2
            C12=C12i+dCp*0
            C44=C44i     
        elif self.softeningmethod=='ren':
            Cpi=(C11i-C12i)/2
            dCp=(Cpi-CPi)
            C11=C11i-dCp*2
            C12=C12i+dCp*0
            C44=2.35*(C11-C12)/2    
        elif self.softeningmethod=='c44softening':
            C11=C11i
            C12=C12i
            C44=CPi    
            
        #print(C11)
        parent_elastic_constants={'11':C11,'22':C11,'33':C11,'12':C12,'13':C12,'23':C12,'44':C44,'55':C44,'66':C44}
        C_A = stiffness_matrix(parent_elastic_constants)
        STA=compliance_from_voight_notation2tensor(np.linalg.inv(C_A))
        S_A = np.linalg.inv(C_A)
        if output=='STA':
            return STA
        elif output=='SA':
            return S_A
        elif output=='CA':
            return C_A
        elif output=='STAC':
            return STA,parent_elastic_constants
    def get_CP(self):
        S_Af=compliance_from_tensor2voight_notation(self.STA)
        C_Af=np.linalg.inv(S_Af)
        C11i=C_Af[0,0]
        C12i=C_Af[0,1]
        C44i=C_Af[3,3]
        if self.softeningmethod=='c44softening':
            self.currentCP=C44i
        else:
            self.currentCP=(C11i-C12i)/2
        
    def generateSTspace(self,numst=10, CP=None, Cpmin='default', Cpmax='default'):
        S_Af=compliance_from_tensor2voight_notation(self.STA)
        C_Af=np.linalg.inv(S_Af)
        C11i=C_Af[0,0]
        C12i=C_Af[0,1]
        C44i=C_Af[3,3]
        Cpi=(C11i-C12i)/2
        if CP is None:
            if Cpmin=='default' and Cpmax=='default':
                CP=np.linspace(Cpi,0,numst+1)[:-1]
            elif Cpmax=='default':
                CP=np.linspace(Cpi,Cpmin,numst)
            elif Cpmin=='default':
                CP=np.linspace(Cpmax,0,numst+1)[:-1]
            else:
                CP=np.linspace(Cpmax,Cpmin,numst)
        #CP=np.linspace(Cpi,Cpmin,numst)
        #CP=np.linspace(Cpi,0,numst+1)[:-1]
        STA=[]
        #if method=='symsoft':
        for CPi in CP:
            STA.append(self.get_STA(CPi,C11i,C12i,C44i))
        self.STAspace=STA
        self.CP=CP
        self.Cpmin=Cpmin
        self.Cpmax=Cpmax
    
    def getCompatibilitySolution(self,Solution=None, Cp=False):
        
        if Solution is None:
            if Cp:
                Solution=self.CurrentCompWithCpOris
            else:
                Solution=self.CurrentCompWithStressOris
        if Cp:
            KEY='CriticalCp'
        else:
            KEY='CriticalStress'
        idxs=np.where(~np.isnan(Solution[KEY]))
        if Cp:
            self.solveCompProblemWithCpOris(idxs=(idxs[0]*0,idxs[0],idxs[1]),CPspace=np.array([Solution['CriticalCp']]))
            CriticalSolution=self.CurrentCompWithCpOris
            CriticalSolution['CriticalCp']=Solution['CriticalCp']
        else:
            self.solveCompProblemWithStressOris(idxs=(idxs[0]*0,idxs[0],idxs[1]),stressSpace=np.array([Solution['CriticalStress']]))
            CriticalSolution=self.CurrentCompWithStressOris
            CriticalSolution['CriticalStress']=Solution['CriticalStress']

        keysS='epsilon_1 epsilon_2 epsilon_3 lambda_1 lambda_2 lambda_3'.split()
        for key in keysS:
            CriticalSolution[key]=np.empty(Solution['epsilon_2'][0,:,:].shape)
            CriticalSolution[key][:]=np.nan
        keysV='eigvec_1 eigvec_2 eigvec_3 n1_a a1_a n2_a a2_a'.split()
        for key in keysV:
            CriticalSolution[key]=np.empty((Solution['epsilon_2'][0,:,:].shape + tuple([3])))
            CriticalSolution[key][:]=np.nan
        keysM='Q1_a Q2_a'.split()
        for key in keysM:
            CriticalSolution[key]=np.empty((Solution['epsilon_2'][0,:,:].shape + tuple([3]) + tuple([3])))
            CriticalSolution[key][:]=np.nan
        for soli in range(idxs[0].shape[0]):
            #Solution to Q_a*F_a - I = a_a x n_a 
            F_a=CriticalSolution['F_AMStress'][0,idxs[0][soli],idxs[1][soli]]   
            Ui=np.eye(3)
            #right Cauchy-Green deformation tensor https://www.comsol.com/multiphysics/analysis-of-deformation
            C=F_a.T.dot(F_a)
            #Get eigenvalues (squares of principal stretches) and eigenvectors
            D,V = np.linalg.eig(C)
            #matmul(C,V) - matmul(V,D*np.eye(3))
            Idxs = np.argsort(D)
            #Sorted eigenvalues and eigenvectors          
            Lambda=D[Idxs]
            V = V[:,Idxs]
            for i,idx in enumerate(Idxs):
                exec(f"epsilon_{str(i+1)}={0.5*(Lambda[i]-1)}")
                exec(f"lambda_{str(i+1)}={Lambda[i]}")
                exec(f"eigvec_{str(i+1)}=V[:,{i}]")
            for i,k in enumerate([-1,1]):
                twind={}
                Ui=np.eye(3)
                n_a=(sqrt(Lambda[2])-sqrt(Lambda[0]))/sqrt(Lambda[2]-Lambda[0])*(-1*sqrt(1-Lambda[0])*Ui.T.dot(V[:,0])+k*sqrt(Lambda[2]-1)*Ui.T.dot(V[:,2]));
                rho =1* norm(n_a)
                n_a=1/rho*n_a
                a_a=rho*(sqrt(Lambda[2]*(1-Lambda[0])/(Lambda[2]-Lambda[0]))*V[:,0]+k*sqrt(Lambda[0]*(Lambda[2]-1)/(Lambda[2]-Lambda[0]))*V[:,2]);
                Q_a=(np.outer(a_a,n_a)+Ui).dot(inv(F_a))
                for key in 'n_a a_a Q_a'.split():
                    exec(f"{key.replace('_',str(i+1)+'_')}={key}")

            for key in keysS:
                #print(f"CriticalSolution['{key}'][{idxs[0][soli]},{idxs[1][soli]}]=key")
                exec(f"CriticalSolution['{key}'][{idxs[0][soli]},{idxs[1][soli]}]={key}")
            for key in keysV:
                #print(f"CriticalSolution['{key}'][{idxs[0][soli]},{idxs[1][soli]}]=key")
                exec(f"CriticalSolution['{key}'][{idxs[0][soli]},{idxs[1][soli],:}]={key}")
            for key in keysM:
                exec(f"CriticalSolution['{key}'][{idxs[0][soli]},{idxs[1][soli]},:,:]={key}")
        
        if Cp:
            self.CurrentCompWithCpOris=CriticalSolution
        else:
            self.CurrentCompWithStressOris=CriticalSolution
        