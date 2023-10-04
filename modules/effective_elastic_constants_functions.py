#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:36:46 2019

@author: lheller
"""
#from matscipy import elasticity
import scipy.io
import math
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.stats import lognorm
import pickle as pckl
import os,sys
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import os,sys
#SysPath = "/home/lheller/DATA/msc/polycrystals/ulm/dataset_complete_nontransforming_part_all_orientations/python";

#sys.path.insert(0, SysPath)

#from export_marc_data import tranform2cylindrical
#from export_marc_data import  inverse_euler_matrix
#from export_marc_data import  read_pckl
#
def tranform2cylindrical(stress_tensor,strain_tensor,x,y):
    #print("matrix rotated")
    norm=np.sqrt(x**2+y**2);
    r_norm = [x/norm,y/norm];
    phi_norm=[-r_norm[1],r_norm[0]];
    RotMatrix = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    RotMatrix[0][0]=r_norm[0];
    RotMatrix[1][1]=phi_norm[1];
    RotMatrix[1][0]=phi_norm[0];
    RotMatrix[0][1]=r_norm[1];
    #M=[[1,2],[2,3]]
    #print('kuk')
    M=np.matrix([[r_norm[0],phi_norm[0],0.],[r_norm[1],phi_norm[1],0],[0.,0.,1]])
    RotMatrix=np.linalg.inv(M)
#check rot matrix 
#        x=10.;y=5.;       
#        new_vec=[0,0,0];
#        vec = [x/norm,y/norm,0]
#        for i in range(3):
#            for j in range(3):
#                new_vec[i]=new_vec[i]+RotMatrix[i][j]*vec[j];
#    stress_tensor_cyl =[[0,0,0],[0,0,0],[0,0,0]];
#    strain_tensor_cyl =[[0,0,0],[0,0,0],[0,0,0]];
    #stress_tensor =[[10.,0.5,2.],[0.5,20.,4.],[2.,30.,4.]];
    
    stress_tensor_cyl=np.ndarray.tolist(np.einsum('ia,jb,ab->ij', RotMatrix, RotMatrix, stress_tensor))
    strain_tensor_cyl=np.ndarray.tolist(np.einsum('ia,jb,ab->ij', RotMatrix, RotMatrix, strain_tensor))
#    for i in range(3):
#        for j in range(3):
#            for k in range(3):
#                for l in range(3):
#                    stress_tensor_cyl[i][j] = stress_tensor_cyl[i][j]+RotMatrix[i][k]*RotMatrix[j][l]*stress_tensor[k][l];
#                    strain_tensor_cyl[i][j] = strain_tensor_cyl[i][j]+RotMatrix[i][k]*RotMatrix[j][l]*strain_tensor[k][l];
#    
    return stress_tensor_cyl,strain_tensor_cyl

  
def euler_matrix(ai, aj, ak):
 
    s1, s2, s3 = math.sin(ai), math.sin(aj), math.sin(ak)
    c1, c2, c3 = math.cos(ai), math.cos(aj), math.cos(ak)

    M = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    
    M[0][0] = c1*c3-s1*s3*c2
    M[0][1] = s1*c3+c1*s3*c2
    M[0][2] = s3*s2
    
    M[1][0] = -c1*s3-s1*c3*c2 
    M[1][1] = -s1*s3+c1*c3*c2
    M[1][2] = c3*s2 
    
    M[2][0] = s1*s2 
    M[2][1] = -c1*s2
    M[2][2] = c2
  
    return M
def inverse_euler_matrix(ai, aj, ak): 
    s1, s2, s3 = math.sin(ai), math.sin(aj), math.sin(ak)
    c1, c2, c3 = math.cos(ai), math.cos(aj), math.cos(ak)
    M = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    M[0][0] = c1*c3-s1*s3*c2
    M[0][1] = -c1*s3-s1*c3*c2
    M[0][2] = s1*s2    
    M[1][0] = s1*c3+c1*s3*c2 
    M[1][1] = -s1*s3+c1*c3*c2
    M[1][2] = -c1*s2 
    M[2][0] = s3*s2 
    M[2][1] = c3*s2
    M[2][2] = c2   
    return M
def dir_modulus_cubic_onedir(C11,C12,C44,dirvin):
    
    S11 = (C11+C12)/((C11+2*C12)*(C11-C12))
    S12 = -C12/((C11+2*C12)*(C11-C12))
    S44 = 1/C44
    dirv=[0.,0.,0.]
    #compliance constant
    dirv[0] = dirvin[0]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[1] = dirvin[1]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[2] = dirvin[2]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    
    alpha = dirv[0]
    beta = dirv[1]
    gama = dirv[2]
    
    #directional modulus
    Edir = 1/(S11-2*((S11-S12)-1/2.*S44)*((alpha**2)*(beta**2)+(alpha**2)*(gama**2)+(beta**2)*(gama**2)))
    
    return Edir

def surface_dir_modulus_cubic_onedir_oneE(C11span, C12span, dirvin,Edirin):
#    npp=100;
#    C11span = np.linspace(149.,189.0,npp);
#    C12span = np.linspace(121.,161.0,npp);
#    npp=10;
#    C44span = np.linspace(23.,43.0,npp);
    Edir = np.empty([C11span.shape[0],C12span.shape[0]]);
    A = np.empty([C11span.shape[0],C12span.shape[0]]);
    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
    C44span = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ex = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ey = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ez = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ov=[0.,0.,0.]
    #print('neco')
    
    dirv=[0.,0.,0.]
    #compliance constant
    dirv[0] = dirvin[0]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[1] = dirvin[1]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[2] = dirvin[2]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    
    alpha = dirv[0]
    beta = dirv[1]
    gama = dirv[2]
    
    
    for j in range(0,C12span.shape[0]):
        C12=C12span[j]
        for i in range(0,C11span.shape[0]):
            C11=C11span[i]
            
            S11 = (C11+C12)/((C11+2*C12)*(C11-C12))
            S12 = -C12/((C11+2*C12)*(C11-C12))
           
            C44 = 1./(2.*(S11-S12)+(1./Edirin-S11)/((alpha**2)*(beta**2)+(alpha**2)*(gama**2)+(beta**2)*(gama**2)));
            #SA, S = compliance_tensor_cubic(C11,C12,C44)
            #Edir[i,j,k]=directional_modulus_onedir(S, dirv)
            
            Edir[i,j] = dir_modulus_cubic_onedir(C11,C12,C44,dirv)
            A[i,j] = 2*C44/(C11-C12);
#                Ovini = [C11,C12,C44];
#                absOvini = np.sqrt(Ovini[0]**2+Ovini[1]**2+Ovini[2]**2);
#                Ov[0]=Ovini[0]/absOvini;
#                Ov[1]=Ovini[1]/absOvini;
#                Ov[2]=Ovini[2]/absOvini;
            
            Xc[i,j] = C11;
            Yc[i,j] = C12;
            C44span[i,j] = C44;
#                Ex = Edir*Ov[0];
            
                    
    return Xc,Yc,Edir,A,C44span

def surface_dir_modulus_cubic_onedir(C11span, C12span, C44span, dirv):
#    npp=100;
#    C11span = np.linspace(149.,189.0,npp);
#    C12span = np.linspace(121.,161.0,npp);
#    npp=10;
#    C44span = np.linspace(23.,43.0,npp);
    Edir = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    A = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ex = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ey = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ez = np.empty([C11span.shape[0],C12span.shape[0]]);
#    Ov=[0.,0.,0.]
    #print('neco')
    for k in range(0,C44span.shape[0]):
        C44=C44span[k]
        for j in range(0,C12span.shape[0]):
            C12=C12span[j]
            for i in range(0,C11span.shape[0]):
                C11=C11span[i]
                #SA, S = compliance_tensor_cubic(C11,C12,C44)
                #Edir[i,j,k]=directional_modulus_onedir(S, dirv)
                
                Edir[i,j,k] = dir_modulus_cubic_onedir(C11,C12,C44,dirv)
                A[i,j,k] = 2*C44/(C11-C12);
#                Ovini = [C11,C12,C44];
#                absOvini = np.sqrt(Ovini[0]**2+Ovini[1]**2+Ovini[2]**2);
#                Ov[0]=Ovini[0]/absOvini;
#                Ov[1]=Ovini[1]/absOvini;
#                Ov[2]=Ovini[2]/absOvini;
                
                Xc[i,j] = C11;
                Yc[i,j] = C12;
                
#                Ex = Edir*Ov[0];
                
                    
    return Xc,Yc,Edir,A
                
def surface_dir_modulus_cubic_onedir02(C11span, C12span, C44span, dirv):
#    npp=100;
#    C11span = np.linspace(149.,189.0,npp);
#    C12span = np.linspace(121.,161.0,npp);
#    npp=10;
#    C44span = np.linspace(23.,43.0,npp);
    Edir = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
    
    for k in range(0,C44span.shape[0]):
        C44=C44span[k]
        for j in range(0,C12span.shape[0]):
            C12=C12span[j]
            for i in range(0,C11span.shape[0]):
                C11=C11span[i]
                Edir[i,j,k] = dir_modulus_cubic_onedir(C11,C12,C44,dirv)
                Ov = [C11,C12,C44]
                Xc[i,j] = C11;
                Yc[i,j] = C12;
                    
    return Xc,Yc,Edir

def perpendicular_vector(v):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    v2=np.array([1, 1, -1.0 * (v[0] + v[1])/v[2]])
    return v2/np.sqrt(v2.dot(v2))
                
    
def directional_modulus_onedir(St, dirvin, output='Young'):
    v=[0.,0.,0.]
    v[0] = dirvin[0]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    v[1] = dirvin[1]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    v[2] = dirvin[2]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    

    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            v2= np.cross(v, [0, 1, 0])
    else:
        v2=np.cross(v, [1, 0, 0])
        v2=perpendicular_vector(v)
    #print('ok')
    
    v3=np.cross(v2, v)
    
    A=np.transpose(np.array([[v3[0],v2[0],v[0]],[v3[1],v2[1],v[1]],[v3[2],v2[2],v[2]]]))
    
    St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,St)
    S3333 = St_rot[2,2,2,2];
    S6666 = St_rot[0,1,0,1];
    #S1111 = 0;
    Ov = np.dot(np.transpose(A),[0,0,1]);
    C3333 = 1/S3333;
    C66 = 1/(4*S6666);
    output=output.lower();   
    #print('okkkkkkkkkkkkkkkkkkkkkkkkk')
    if output=='young':
        return C3333
    elif output=='shear':
        return C66
    elif output=='youngshear':
        return C3333,C66
    


def directional_modulus(St, output='Young', phi1lim=2*np.pi,phi2lim=np.pi,npp=181):
    phi1 = np.linspace(0,phi1lim,npp);
    phi2 = np.linspace(0,phi2lim,npp);
    
    [PHI1,PHI2] = np.meshgrid(phi1,phi2);
    C33x = np.zeros(np.shape(PHI1))
    C33y = np.zeros(np.shape(PHI1))
    C33z = np.zeros(np.shape(PHI1))
    C66x = np.zeros(np.shape(PHI1))
    C66y = np.zeros(np.shape(PHI1))
    C66z = np.zeros(np.shape(PHI1))
    for i in range(0,np.shape(PHI1)[0]):
        for j in range(0,np.shape(PHI1)[1]):
            Phi1i = PHI1[i,j];
            Phi2i = PHI2[i,j];
            
            rotX = np.array([[1,0,0],[0,np.cos(Phi1i),np.sin(Phi1i)],[0,-np.sin(Phi1i),np.cos(Phi1i)]]);
            rotZ = np.array([[np.cos(Phi1i),np.sin(Phi1i),0],[-np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
            rotY = np.array([[np.cos(Phi2i),0,-np.sin(Phi2i)],[0,1,0],[np.sin(Phi2i),0,np.cos(Phi2i)]]);
            A=np.matmul(rotY,rotZ);
            St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,St)
            S3333 = St_rot[2,2,2,2];
            S6666 = St_rot[0,1,0,1];
            #S1111 = 0;
            Ov = np.dot(np.transpose(A),[0,0,1]);
            C3333 = 1/S3333;
            C66 = 1/(4*S6666);
            C33x[i,j] = C3333*Ov[0];
            C33y[i,j] = C3333*Ov[1];
            C33z[i,j] = C3333*Ov[2];
            C66x[i,j] = C66*Ov[0];
            C66y[i,j] = C66*Ov[1];
            C66z[i,j] = C66*Ov[2];
    output=output.lower();      
    if output=='young':
        return C33x,C33y,C33z
    elif output=='shear':
        return C66x,C66y,C66z
    elif output=='youngshear':
        return C33x,C33y,C33z,C66x,C66y,C66z

def plot_directional_surface(C33x_s,C33y_s,C33z_s,rstride=1, cstride=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(azim=0, elev=0)
    # Plot the surface.
    Xs = C33x_s
    Ys = C33y_s
    Zs = C33z_s
    Es = (Xs**2+Ys**2+Zs**2)**(1./2.)
    #np.amax(Es)
    
    #np.amax(Es);
    #np.amin(Es)
    cmap=plt.cm.get_cmap('jet', 12)
    vmin = np.amin(Es);
    vmax = np.amax(Es);
    my_col = cm.jet((Es-vmin)/(np.amax(Es)-vmin))
    my_col = cmap((Es-vmin)/(np.amax(Es)-vmin))
    
    col = 'b'
    surf=ax.plot_surface(Xs,Ys,Zs, rstride=rstride, cstride=cstride, linewidth=0.5, edgecolors='black',facecolors = my_col,vmin=vmin,vmax=vmax)
    #ax.plot_surface(Xs,Ys,Zs, rstride=5, cstride=5, linewidth=0.5, edgecolors='black',facecolors = my_col)
    #ax.contourf(Xs, Ys, Zs,20) 
    #LL=40.
    xa=ax.plot([0,vmax],[0,0],[0,0],linestyle='--',linewidth=1,color='black')
    ya=ax.plot([0,0],[0,-vmax ],[0,0],linestyle='--',linewidth=1,color='black')
    za=ax.plot([0,0],[0,0],[0,vmax],linestyle='--',linewidth=1,color='black')
    #ax.text(LL+5.,0.,0,'[010]',color='black',fontsize=SMALL_SIZE)
    #ax.text(0.,-LL-5.,-10,'[100]',color='black',fontsize=SMALL_SIZE)
    #ax.text(0.,0.,LL+5.0,'[001]',color='black',fontsize=SMALL_SIZE)
    
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    m = cm.ScalarMappable(cmap=cm.jet)
    m = cm.ScalarMappable(cmap=cmap)
    mticks = 6;
    m.set_array(np.linspace(vmin,vmax,mticks))
    cb = plt.colorbar(m)
    #cb.set_clim(vmin,vmax)
    cb.set_ticks(np.linspace(vmin,vmax,mticks))
    cb.set_label('Directional Young''s modulus [GPa]')
    plt.show()
    plt.axis('off')
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']); ax.set_box_aspect(np.ptp(limits, axis = 1))


#    ax.set_aspect('equal')
    ax.view_init(45,-45)
    plt.draw()
#    ax.set_aspect('equal')
    
    return fig,ax,surf,cb

def directional_shear_modulus(St, phi1lim=2*np.pi,phi2lim=np.pi,npp=181):
    phi1 = np.linspace(0,phi1lim,npp);
    phi2 = np.linspace(0,phi2lim,npp);
    CHI = np.linspace(0,2*np.pi,73);
    [PHI1,PHI2] = np.meshgrid(phi1,phi2);
    Gmaxx = np.zeros(np.shape(PHI1))
    Gmaxy = np.zeros(np.shape(PHI1))
    Gmaxz = np.zeros(np.shape(PHI1))
    for i in range(0,np.shape(PHI1)[0]):
        print(str(i)+'/'+str(np.shape(PHI1)[0]))
        for j in range(0,np.shape(PHI1)[1]):
            Phi1i = PHI1[i,j];
            Phi2i = PHI2[i,j];
            
            rotX = np.array([[1,0,0],[0,np.cos(Phi1i),np.sin(Phi1i)],[0,-np.sin(Phi1i),np.cos(Phi1i)]]);
            rotZ = np.array([[np.cos(Phi1i),np.sin(Phi1i),0],[-np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
            rotY = np.array([[np.cos(Phi2i),0,-np.sin(Phi2i)],[0,1,0],[np.sin(Phi2i),0,np.cos(Phi2i)]]);
            A=np.matmul(rotY,rotZ);
            Ov = np.dot(np.transpose(A),[0,0,1]);
            S44=[];
            for chi in CHI:
                rotZ2 = np.array([[np.cos(chi),np.sin(chi),0],[-np.sin(chi),np.cos(chi),0],[0,0,1]]);
                A2=np.matmul(rotZ2,A);
                St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A2, A2, A2, A2,St)
                S3333 = St_rot[2,2,2,2];
                S44.append(St_rot[1,2,1,2]);
                
                
                
            S44max=np.mean(S44)
            Gmax=1.0/(4.0*S44max);
            #print(A)
            #S1111 = 0;
            Gmaxx[i,j] = Gmax*Ov[0];
            Gmaxy[i,j] = Gmax*Ov[1];
            Gmaxz[i,j] = Gmax*Ov[2];



    return Gmaxx,Gmaxy,Gmaxz

def directional_shear_modulus_poisson(St, phi1lim=2*np.pi,phi2lim=np.pi,npp=181):
    phi1 = np.linspace(0,phi1lim,npp);
    phi2 = np.linspace(0,phi2lim,npp);
    CHI = np.linspace(0,2*np.pi,73);
    [PHI1,PHI2] = np.meshgrid(phi1,phi2);
    Gmax = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    Gmin = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    Gavg = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    MUmax = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    MUminpos = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    MUminneg = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    MUavg = [np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1)),np.zeros(np.shape(PHI1))];
    for i in range(0,np.shape(PHI1)[0]):
        print(str(i)+'/'+str(np.shape(PHI1)[0]))
        for j in range(0,np.shape(PHI1)[1]):
            Phi1i = PHI1[i,j];
            Phi2i = PHI2[i,j];
            
            rotX = np.array([[1,0,0],[0,np.cos(Phi1i),np.sin(Phi1i)],[0,-np.sin(Phi1i),np.cos(Phi1i)]]);
            rotZ = np.array([[np.cos(Phi1i),np.sin(Phi1i),0],[-np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
            rotY = np.array([[np.cos(Phi2i),0,-np.sin(Phi2i)],[0,1,0],[np.sin(Phi2i),0,np.cos(Phi2i)]]);
            A=np.matmul(rotY,rotZ);
            
            St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,St)
            S3333 = St_rot[2,2,2,2];
            C3333 = 1/S3333;

            Ov = np.dot(np.transpose(A),[0,0,1]);
            S23=[];
            S44=[];
            for chi in CHI:
                rotZ2 = np.array([[np.cos(chi),np.sin(chi),0],[-np.sin(chi),np.cos(chi),0],[0,0,1]]);
                A2=np.matmul(rotZ2,A);
                St_rot2 = np.einsum('ia,jb,kc,ld,abcd->ijkl',A2, A2, A2, A2,St)
                S23.append(St_rot2[1,1,2,2]);
                S44.append(St_rot2[1,2,1,2]);
            S44max=np.mean(S44)
            Gavgi=np.mean([1.0/(4.0*S44i) for S44i in S44]);
            Gmini=min([1.0/(4.0*S44i) for S44i in S44]);
            Gmaxi=max([1.0/(4.0*S44i) for S44i in S44]);
            
            MUavgi = np.mean([-S23i/S3333 for S23i in S23]);
            MUmaxi = np.max([-S23i/S3333 for S23i in S23]);
            MUminnegi = np.min([-S23i/S3333 for S23i in S23]);
            MUminposi = np.min([-S23i/S3333 for S23i in S23 if S23i<0]);
            
            #print(A)
            #S1111 = 0;
            for Ovi,ii in zip(Ov, range(0,3)):
                Gmax[ii][i,j] = Gmaxi*Ovi;
            for Ovi,ii in zip(Ov, range(0,3)):
                Gmin[ii][i,j] = Gmini*Ovi;
            for Ovi,ii in zip(Ov, range(0,3)):
                Gavg[ii][i,j] = Gavgi*Ovi;

            for Ovi,ii in zip(Ov, range(0,3)):
                MUmax[ii][i,j] = MUmaxi*Ovi;
            for Ovi,ii in zip(Ov, range(0,3)):
                MUavg[ii][i,j] = MUavgi*Ovi;
            for Ovi,ii in zip(Ov, range(0,3)):
                MUminneg[ii][i,j] = MUminnegi*Ovi;
            for Ovi,ii in zip(Ov, range(0,3)):
                MUminpos[ii][i,j] = MUminposi*Ovi;




    return Gmax,Gmin,Gavg, MUmax,MUminneg,MUminpos,MUavg

def directional_shear_modulus_poisson_xyplane(St,npp=181):
    CHI = np.linspace(0,2*np.pi,73);        
    S3333 = St[2,2,2,2];

    MU=[];
    MUx=[];
    MUy=[];
    G=[];
    Gx=[];
    Gy=[];
    for chi in CHI:
        rotZ2 = np.array([[np.cos(chi),np.sin(chi),0],[-np.sin(chi),np.cos(chi),0],[0,0,1]]);
        
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',rotZ2, rotZ2, rotZ2, rotZ2,St)
    
        G.append(1.0/(4.0*St_rot[1,2,1,2]));
        Gi=1.0/(4.0*St_rot[1,2,1,2]);
        Gx.append(Gi*np.cos(chi));
        Gy.append(Gi*np.sin(chi));
                       
        MU.append(-St_rot[0,0,2,2]/S3333);
        MUi =-St_rot[0,0,2,2]/S3333;
        MUx.append(MUi*np.cos(chi));
        MUy.append(MUi*np.sin(chi));
            


    return G,Gx,Gy,MU,MUx,MUy,CHI

def directional_shear_modulus_xyplane(St,npp=181):
    phi1 = np.linspace(0,2*np.pi,npp);
    CHI = np.linspace(0,2*np.pi,73);
    Gmax = [];
    Gmax_x = [];
    Gmax_y = [];
    Gmin = [];
    Gmin_x = [];
    Gmin_y = [];
    Gavg = [];
    Gavg_x = [];
    Gavg_y = [];
    Gmindir=[]
    Gmaxdir=[]


    for Phi1i in phi1:
        #matice prechodu do s.s. natoceneho kolem z o uhel Phi(i)
        A = np.array([[np.cos(Phi1i),np.sin(Phi1i),0],[-np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
        
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,St)
        #Souradnice natocene osy x v puvodnim s.s
        Ov = np.dot(np.transpose(A),[1,0,0]);
        S13=[];
        Adir=[]
        G=[]
        for chi in CHI:
            rotX = passive_rotation(chi,'x')
            A2=np.matmul(rotX,A);
            St_rot2 = np.einsum('ia,jb,kc,ld,abcd->ijkl',A2, A2, A2, A2,St)
            S13.append(St_rot2[0,2,0,2]);
            Adir.append(np.transpose(A2).dot([0,1,0]))
            G.append(1.0/(4.0*S13[-1]))
        Gavg.append(np.mean(G));
        Gavg_x.append(Gavg[-1]*Ov[0]);
        Gavg_y.append(Gavg[-1]*Ov[1]);
        
        Gmin.append(min(G));
        Gmin_x.append(min(G)*Ov[0]);
        Gmin_y.append(min(G)*Ov[1]);
        Gmindir.append(Adir[G.index(min(G))])
        Gmax.append(max(G));
        Gmaxdir.append(Adir[G.index(max(G))])
        Gmax_x.append(Gmax[-1]*Ov[0]);
        Gmax_y.append(Gmax[-1]*Ov[1]);
        


    return np.vstack((Gmax_x,Gmax_y)).T,Gmaxdir, np.vstack((Gmin_x,Gmin_y)).T,Gmindir,np.vstack((Gavg_x,Gavg_y)).T

def directional_poisson_xyplane_loading_z(St,npp=181, both=False):
    phi1 = np.linspace(0,2*np.pi,npp);
    
    MU=[]
    MUx=[]
    MUy=[]
    MU2=[]
    MU2x=[]
    MU2y=[]
    #Smykovy modul v rovine kolme na y ve smeru z
    #G_yz=1/4/ST(2,3,2,3)
    #Smykovy modul v rovine kolme na y ve smeru x
    #G_zx=1/4/ST(1,3,1,3)
    #Smerove possonovo cislo modul ve smeru y pro tah ve smeru x
    #mu_y = -ST(3,3,2,2)/ST(2,2,2,2)


    for Phi1i in phi1:
        #matice prechodu do s.s. natoceneho kolem z o uhel Phi(i)
        A = np.array([[np.cos(Phi1i),np.sin(Phi1i),0],[-np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
        #A2 = np.array([[np.cos(Phi1i),-np.sin(Phi1i),0],[np.sin(Phi1i),np.cos(Phi1i),0],[0,0,1]]);
        
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,St)
        #St2_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A2, A2, A2, A2,St)
        #Souradnice natocene osy x v puvodnim s.s
        Ov = np.dot(np.transpose(A),[1,0,0]);
        Ov2 = np.dot(np.transpose(A),[0,1,0]);
        S33=St_rot[2,2,2,2]
        S31=St_rot[2,2,0,0]
        S32=St_rot[2,2,1,1]
        MU.append(-S31/S33)
        MUx.append(MU[-1]*Ov[0])
        MUy.append(MU[-1]*Ov[1])
        MU2.append(-S32/S33)
        MU2x.append(MU2[-1]*Ov2[0])
        MU2y.append(MU2[-1]*Ov2[1])
    if both:
        return MU,np.vstack((MUx,MUy)).T, MU2, np.vstack((MU2x,MU2y)).T
    else:
        return np.vstack((MUx,MUy)).T
def directional_young_modulus_xyplane(St,npp=73):
    CHI = np.linspace(0,2*np.pi,npp);        
    S3333 = St[2,2,2,2];

    E=[];
    Ex=[];
    Ey=[];
    for chi in CHI:
        rotZ2 = np.array([[np.cos(chi),np.sin(chi),0],[-np.sin(chi),np.cos(chi),0],[0,0,1]]);
        
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',rotZ2, rotZ2, rotZ2, rotZ2,St)
    
        E.append(1.0/(St_rot[0,0,0,0]));
        Ei=1.0/(St_rot[0,0,0,0]);
        Ex.append(Ei*np.cos(chi));
        Ey.append(Ei*np.sin(chi));
                                  
    return E,Ex,Ey,CHI



def stiffness_tensor_cubic(CA_11,CA_12,CA_44):
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    C=np.zeros((3,3,3,3))
    C[0,0,0,0] = CA_11;
    C[1,1,1,1] = CA_11;
    C[2,2,2,2] = CA_11;
    
    C[0,0,1,1] = CA_12;
    C[0,0,2,2] = CA_12;
    C[1,1,0,0] = CA_12;
    C[1,1,2,2] = CA_12;
    C[2,2,0,0] = CA_12;
    C[2,2,1,1] = CA_12;
    
    
    C[1,2,1,2] = CA_44;
    C[2,0,2,0] = CA_44;
    C[0,1,0,1] = CA_44;
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if C[i,j,k,l]!=C[k,l,i,j]:
                        if C[i,j,k,l]==0:
                            C[i,j,k,l] = C[k,l,i,j];
                        else:
                            C[k,l,i,j] = C[i,j,k,l];
                    if C[i,j,k,l]!=C[j,i,k,l]:
                        if C[j,i,k,l]==0:
                            C[j,i,k,l] = C[i,j,k,l];
                        else:
                            C[i,j,k,l] = C[j,i,k,l];
                    if C[i,j,k,l]!=C[i,j,l,k]:
                        if C[i,j,l,k] ==0:
                            C[i,j,l,k] = C[i,j,k,l];
                        else:
                            C[i,j,k,l] = C[i,j,l,k];

    
    CA = np.zeros((6,6));
    for i in range(0,6):
        for j in range(0,6):
            Ind1 = IndT.index(i)
            Ind2 = IndT.index(j)
            CA[i,j]=C[Vi[Ind1][0],Vi[Ind1][1],Vi[Ind2][0],Vi[Ind2][1]]
    
        
    return CA, C
def stiffness_tensor_matrix_index_dictionaries():    
    CT = np.ones((3,3,3,3))
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    rg=range(0,3)
    T2M={}
    M2T={}
    for i in rg:
        for j in rg:
            for k in rg:
                for l in rg:
                    Ind1 = Vi.index([i,j])
                    Ind2 = Vi.index([k,l])
                    #print(f'CT[{(i,j,k,l)}=CA[{IndT[Ind1]},{IndT[Ind2]}]')
                    #CT[i,j,k,l]=CA[IndT[Ind1],IndT[Ind2]];
                    T2M[(i,j,k,l)]=(IndT[Ind1],IndT[Ind2])
                    M2T[(IndT[Ind1],IndT[Ind2])]=(i,j,k,l)
    return T2M,M2T
def stiffness_matrix_tensor_const(constants):
    T2M,M2T=stiffness_tensor_matrix_index_dictionaries()
    C_M=np.zeros((6,6))
    for ij in T2M.keys():
        #print(f'{ij}-{T2M[ij]}')
        try:
            C_M[T2M[ij][0],T2M[ij][1]]=constants[ij]
            C_M[T2M[ij][1],T2M[ij][0]]=constants[ij]
        except:
            pass
    return C_M
        
def stiffness_matrix(constants):
#CM_11=223
#CM_12=129;
#CM_13=99;
#CM_15=27;
#CM_22=241;
#CM_23=125;
#CM_25=-9;
#CM_33=200;
#CM_35=4;
#CM_44=76;
#CM_46=-4;
#CM_55=21;
#CM_66=77;
#constants={'11':CM_11,'12':CM_12,'13':CM_13,'15':CM_15,\
#           '22':CM_22,'23':CM_23,'25':CM_25,\
#            '33':CM_33,'35':CM_35,\
#           '44':CM_44,'46':CM_46,\
#           '55':CM_55,'66':CM_66}

    C=np.zeros((6,6))
    
    for key in constants.keys():
        C[int(key.split()[0][0])-1,int(key.split()[0][1])-1]=constants[key]
        C[int(key.split()[0][1])-1,int(key.split()[0][0])-1]=constants[key]
    
    return C

def compliance_tensor_cubic(CA_11,CA_12,CA_44):
    SA = np.zeros((6,6));
    Coeff = np.zeros((6,6))
    SI = np.ones((3,3,3,3))
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    for i in range(0,3):
        for j in range(0,3):
            #pro epsilon(i,j)=>vektorovy index (1-6) IndT(Ind1)
            try:
                Ind1 = Vi.index([i,j])
            except:
                Ind1 = [];    
            for k in range(0,3):
                for l in range(0,3):
                    #prispevky vsech sigma(k,l) pro dane esilon(i,j)
                    for m in range(0,3):
                        for n in range(0,3):
                            if (k==m and l==n):
                                #vyber prispevku od sigma(m,n)=>prispevek
                                #elementu matice poddajnosti
                                #S(IndT(Ind1),IndT(Ind2)), (m,n)=>IndT(Ind2)
                                try:
                                    Ind2 = Vi.index([k,l])
                                except:
                                    Ind2 =[] ;    
                                
                                if type(Ind1)!=list:
                                    if type(Ind2)!=list:
                                        Coeff[IndT[Ind1],IndT[Ind2]] = Coeff[IndT[Ind1],IndT[Ind2]]+SI[i,j,k,l];
    
    Coeff_inv = 1./Coeff;
    
    
    
    SA[0,0]=(CA_11+CA_12)/(CA_11**2+CA_11*CA_12-2*CA_12**2);
    SA[1,1]=SA[0,0];
    SA[2,2]=SA[0,0];
    SA[0,1]=-CA_12/(CA_11**2+CA_11*CA_12-2*CA_12**2);
    SA[1,0]=SA[0,1]
    SA[0,2]=SA[0,1]
    SA[2,0]=SA[0,1]
    SA[1,2]=SA[0,1]
    SA[2,1]=SA[0,1]
    SA[1,2]=SA[0,1]
    SA[3,3]=1/CA_44;
    SA[4,4]=SA[3,3]
    SA[5,5]=SA[3,3]
    S = np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    Ind1 = Vi.index([i,j])
                    Ind2 = Vi.index([k,l])
                    S[i,j,k,l]=Coeff_inv[IndT[Ind1],IndT[Ind2]]*SA[IndT[Ind1],IndT[Ind2]];
    
    
    return SA, S

def stiffness_from_tensor2voight_notation(C):
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    
    CA = np.zeros((6,6));
    for i in range(0,6):
        for j in range(0,6):
            Ind1 = IndT.index(i)
            Ind2 = IndT.index(j)
            CA[i,j]=C[Vi[Ind1][0],Vi[Ind1][1],Vi[Ind2][0],Vi[Ind2][1]]
    
        
    return CA
def stiffness_from_voight_notation2tensor(CA):
    CT = np.ones((3,3,3,3))
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    rg=range(0,3)
    for i in rg:
        for j in rg:
            for k in rg:
                for l in rg:
                    Ind1 = Vi.index([i,j])
                    Ind2 = Vi.index([k,l])
                    CT[i,j,k,l]=CA[IndT[Ind1],IndT[Ind2]];
                    
    return CT                
def compliance_from_voight_notation2tensor(SA):
    Coeff = np.zeros((6,6))
    SI = np.ones((3,3,3,3))
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    for i in range(0,3):
        for j in range(0,3):
            #pro epsilon(i,j)=>vektorovy index (1-6) IndT(Ind1)
            try:
                Ind1 = Vi.index([i,j])
            except:
                Ind1 = [];    
            for k in range(0,3):
                for l in range(0,3):
                    #prispevky vsech sigma(k,l) pro dane esilon(i,j)
                    for m in range(0,3):
                        for n in range(0,3):
                            if (k==m and l==n):
                                #vyber prispevku od sigma(m,n)=>prispevek
                                #elementu matice poddajnosti
                                #S(IndT(Ind1),IndT(Ind2)), (m,n)=>IndT(Ind2)
                                try:
                                    Ind2 = Vi.index([k,l])
                                except:
                                    Ind2 =[] ;    
                                
                                if type(Ind1)!=list:
                                    if type(Ind2)!=list:
                                        Coeff[IndT[Ind1],IndT[Ind2]] = Coeff[IndT[Ind1],IndT[Ind2]]+SI[i,j,k,l];
    
    Coeff_inv = 1./Coeff;
    S = np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    Ind1 = Vi.index([i,j])
                    Ind2 = Vi.index([k,l])
                    S[i,j,k,l]=Coeff_inv[IndT[Ind1],IndT[Ind2]]*SA[IndT[Ind1],IndT[Ind2]];
    return S
    
def compliance_from_tensor2voight_notation(St_rot):
    Coeff = np.zeros((6,6))
    SI = np.ones((3,3,3,3))
    Vi = [[0,0],[1,1],[2,2],[2,1],[1,2],[2,0],[0,2],[0,1],[1,0]];
    IndT =[0,1,2,3,3,4,4,5,5]
    for i in range(0,3):
        for j in range(0,3):
            #pro epsilon(i,j)=>vektorovy index (1-6) IndT(Ind1)
            try:
                Ind1 = Vi.index([i,j])
            except:
                Ind1 = [];    
            for k in range(0,3):
                for l in range(0,3):
                    #prispevky vsech sigma(k,l) pro dane esilon(i,j)
                    for m in range(0,3):
                        for n in range(0,3):
                            if (k==m and l==n):
                                #vyber prispevku od sigma(m,n)=>prispevek
                                #elementu matice poddajnosti
                                #S(IndT(Ind1),IndT(Ind2)), (m,n)=>IndT(Ind2)
                                try:
                                    Ind2 = Vi.index([k,l])
                                except:
                                    Ind2 =[] ;    
                                
                                if type(Ind1)!=list:
                                    if type(Ind2)!=list:
                                        Coeff[IndT[Ind1],IndT[Ind2]] = Coeff[IndT[Ind1],IndT[Ind2]]+SI[i,j,k,l];


    S_rot = np.zeros((6,6));
    for i in range(0,6):
        for j in range(0,6):
            Ind1 = IndT.index(i)
            Ind2 = IndT.index(j)
            S_rot[i,j]=Coeff[i,j]*St_rot[Vi[Ind1][0],Vi[Ind1][1],Vi[Ind2][0],Vi[Ind2][1]]

    return S_rot

def voigt_reuss_variations(Phi1,PHI,Phi2,Vol, C11,C12,C44, meanstrainT, meanstressT):
#    meanstressT=[0,0,meanstress,0,0,0]
#    meanstrainT=[-0.00214,-0.00214,meanstrain,0,0,0]
    C_eff_v = np.zeros((6,6))
    S_eff_r = np.zeros((6,6))
    TotalVol=0;
    CA, C = stiffness_tensor_cubic(C11,C12,C44)
    SA, S = compliance_tensor_cubic(C11,C12,C44)
    ini=True
    for phi1,Fi,phi2,voli in zip(Phi1,PHI,Phi2,Vol):
        TotalVol+=voli
        #rotation matrix
        A =inverse_euler_matrix(phi1, Fi, phi2);
        Ct_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,C)
        C_rot=stiffness_from_tensor2voight_notation(Ct_rot)
        #print(C_rot)
        #C_rot = elasticity.rotate_cubic_elastic_constants(C11, C12, C44, A, tol=1e-6);
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,S)
        S_rot = compliance_from_tensor2voight_notation(St_rot)
        if ini:
            Stress=np.matmul(C_rot,meanstrainT)
            Epsilon=np.matmul(S_rot,meanstressT)
            for i in [3,4,5]:
                Epsilon[i]=Epsilon[i]/2
            ini=False
        else:
            Eps = np.matmul(S_rot,meanstressT)
            for i in [3,4,5]:
                Eps[i]=Eps[i]/2            
            Epsilon=np.vstack((Epsilon,Eps))
            Stress=np.vstack((Stress,np.matmul(C_rot,meanstrainT)))
        C_eff_v+=C_rot*voli
        S_eff_r+=S_rot*voli
        
    C_eff_v/=TotalVol
    S_eff_r/=TotalVol
    weights=[]
    for voli in Vol:
        weights.append(voli/TotalVol)
        
    stdstrain_reuss=[]
    meanstrain_reuss=[]
    stdstress_voigt=[]
    meanstress_voigt=[]
    for eps,stress in zip(Epsilon.T,Stress.T):
        mu = np.average(eps, weights=weights)
        stdstrain_reuss.append(np.sqrt(np.average((eps-mu)**2, weights=weights))*100)
        meanstrain_reuss.append(np.average(eps,weights=weights)*100.)
        meanstress_voigt.append(np.average(stress,weights=weights)*1000)
        mu = np.average(stress, weights=weights)
        stdstress_voigt.append(np.sqrt(np.average((stress-mu)**2, weights=weights))*1000)

    return C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss 


def voigt_reuss_effective_constants(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress):

    C_eff_v = np.zeros((6,6))
    S_eff_r = np.zeros((6,6))
    TotalVol=0;
    CA, C = stiffness_tensor_cubic(C11,C12,C44)
    SA, S = compliance_tensor_cubic(C11,C12,C44)
    Sigma11_vi=[]
    Epsilon11_ri=[]
    Sigma22_vi=[]
    Epsilon22_ri=[]
    Sigma33_vi=[]
    Epsilon33_ri=[]
    Epsilon23_ri=[]
    Epsilon13_ri=[]
    Epsilon12_ri=[]
    meanstress_voigt=np.empty([3])
    stdstress_voigt=np.empty([3])
    meanstrain_reuss=np.empty([3])
    stdstrain_reuss=np.empty([6])
    for phi1,Fi,phi2,voli in zip(Phi1,PHI,Phi2,Volume):
        TotalVol+=voli
        #rotation matrix
        A =inverse_euler_matrix(phi1, Fi, phi2);
        Ct_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,C)
        C_rot=stiffness_from_tensor2voight_notation(Ct_rot)
        #print(C_rot)
        #C_rot = elasticity.rotate_cubic_elastic_constants(C11, C12, C44, A, tol=1e-6);
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,S)
        S_rot = compliance_from_tensor2voight_notation(St_rot)
        Sigma11_vi.append(1/S_rot[0,0]*meanstrain*1000)
        Epsilon11_ri.append(S_rot[0,0]*meanstress*100)
        Sigma22_vi.append(1/S_rot[1,1]*meanstrain*1000)
        Epsilon22_ri.append(S_rot[1,1]*meanstress*100)
        Sigma33_vi.append(1/S_rot[2,2]*meanstrain*1000)
        Epsilon33_ri.append(S_rot[2,2]*meanstress*100)


        C_eff_v+=C_rot*voli
        S_eff_r+=S_rot*voli
        
    C_eff_v/=TotalVol
    S_eff_r/=TotalVol
    
    meanstress_voigt[0]=np.mean(Sigma11_vi)
    stdstress_voigt[0]=np.std(Sigma11_vi)
    meanstress_voigt[1]=np.mean(Sigma22_vi)
    stdstress_voigt[1]=np.std(Sigma22_vi)
    meanstress_voigt[2]=np.mean(Sigma33_vi)
    stdstress_voigt[2]=np.std(Sigma33_vi)
    
    meanstrain_reuss[0]=np.mean(Epsilon11_ri)
    stdstrain_reuss[0]=np.std(Epsilon11_ri)
    meanstrain_reuss[1]=np.mean(Epsilon22_ri)
    stdstrain_reuss[1]=np.std(Epsilon22_ri)
    meanstrain_reuss[2]=np.mean(Epsilon33_ri)
    stdstrain_reuss[2]=np.std(Epsilon33_ri)
    stdstrain_reuss[3]=np.std(Epsilon23_ri)
    stdstrain_reuss[4]=np.std(Epsilon13_ri)
    stdstrain_reuss[5]=np.std(Epsilon12_ri)

    return C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss 



def voigt_reuss_effective_constants02Cyl(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress,COORDS):
    
    C_eff_v = np.zeros((6,6))
    S_eff_r = np.zeros((6,6))
    TotalVol=0;
    CA, C = stiffness_tensor_cubic(C11,C12,C44)
    SA, S = compliance_tensor_cubic(C11,C12,C44)
    Sigma11_vi=[]
    Epsilon11_ri=[]
    Sigma22_vi=[]
    Epsilon22_ri=[]
    Sigma33_vi=[]
    Epsilon33_ri=[]
    Epsilon23_ri=[]
    Epsilon13_ri=[]
    Epsilon12_ri=[]
    meanstress_voigt=np.empty([3])
    stdstress_voigt=np.empty([3])
    meanstrain_reuss=np.empty([3])
    stdstrain_reuss=np.empty([6])
    
    strain_r=[]
    stress_r=[]
    stress_v=[]
    strain_v=[]
    for phi1,Fi,phi2,voli,coords in zip(Phi1,PHI,Phi2,Volume,COORDS):
        TotalVol+=voli
        #rotation matrix
        A =inverse_euler_matrix(phi1, Fi, phi2);
        Ct_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,C)
        C_rot=stiffness_from_tensor2voight_notation(Ct_rot)
        #print(C_rot)
        #C_rot = elasticity.rotate_cubic_elastic_constants(C11, C12, C44, A, tol=1e-6);
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,S)
        S_rot = compliance_from_tensor2voight_notation(St_rot)
        
#        redS_rot = np.zeros((len(IndMS),len(IndMS)))
#        
#        for i,ii in zip(IndMS,range(0,len(IndMS))):
#            for j,jj in zip(IndMS,range(0,len(IndMS))):
#                redS_rot[ii,jj]=S_rot[i,j]
#                
        #stress_vired = np.matmul(np.linalg.inv(S_rot[np.ix_(IndMS,IndMS)]),meanstrain[IndMS])
        #stress_vi = np.array([0.,0.,0.,0.,0.,0.])
        #stress_vi[IndMS]=stress_vired
        #strain_vi=np.matmul(S_rot,stress_vi)
        stress=np.matmul(np.linalg.inv(S_rot),meanstrain)
        stress_tensor = [[stress[0],stress[3],stress[4]],
                 [stress[3],stress[1],stress[5]],
                 [stress[4],stress[5],stress[2]]];

        strain = np.matmul(S_rot,meanstress)
        strain_tensor = [[strain[0],strain[3]/2.,strain[4]/2.],
                         [strain[3]/2.,strain[1],strain[5]/2.],
                         [strain[4]/2.,strain[5]/2.,strain[2]]];

        stress_tensor_cyl, strain_tensor_cyl = tranform2cylindrical(stress_tensor,strain_tensor,coords[0],coords[1])                   

        stress_vi=[stress_tensor_cyl[0][0],stress_tensor_cyl[1][1],stress_tensor_cyl[2][2],
                              stress_tensor_cyl[0][1],stress_tensor_cyl[0][2],stress_tensor_cyl[1][2]];
        strain_ri=[strain_tensor_cyl[0][0],strain_tensor_cyl[1][1],strain_tensor_cyl[2][2],
                              strain_tensor_cyl[0][1],strain_tensor_cyl[0][2],strain_tensor_cyl[1][2]];
        
        stress_v.append(stress_vi)
        strain_r.append(strain_ri)
        

        C_eff_v+=C_rot*voli
        S_eff_r+=S_rot*voli
        
    C_eff_v/=TotalVol
    S_eff_r/=TotalVol
    
    mean_stress_v=[]
    mean_strain_r=[]
    std_stress_v=[]
    std_strain_r=[]
    for i in range(0,len(stress_v[0])):
        mean_stress_v.append(np.mean([s[i] for s in stress_v]))
        mean_strain_r.append(np.mean([s[i]*100 for s in strain_r]))
        std_stress_v.append(np.std([s[i] for s in stress_v]))
        std_strain_r.append(np.std([s[i]*100 for s in strain_r]))
    

    return C_eff_v,S_eff_r,mean_stress_v, mean_strain_r,std_stress_v,std_strain_r
 



def voigt_reuss_effective_constants02(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress):
#BASEPATH = "/home/lheller/media/lheller/Data/msc/model123/original_C"
#BASENAME = "complete_model_sym_stress_exp_job"
#Grains3dxrdFileName = BASEPATH+'/'+BASENAME+'_Grains3dxrd.pkl'
#GrainResults3dxrdFileName = BASEPATH+'/'+BASENAME+'_GrainResults3dxrd.pkl'
#GrainElasticConstantsFileName = BASEPATH+'/'+BASENAME+'_elastic_constants.pkl'
#
#HeaderGrains3dxrd, Grains3dxrd = read_pckl(Grains3dxrdFileName) ;
#HeaderGrainResults3dxrd, GrainResults3dxrd = read_pckl(GrainResults3dxrdFileName) ;
#HeaderGrainElasticConstants, GrainElasticConstants = read_pckl(GrainElasticConstantsFileName) ;
##single crystal directional modulus
#C11=GrainElasticConstants[0][HeaderGrainElasticConstants.index('C11')]
#C12=GrainElasticConstants[0][HeaderGrainElasticConstants.index('C12')]
#C44=GrainElasticConstants[0][HeaderGrainElasticConstants.index('C44')]
#
#
#
#IndPhi1 = HeaderGrains3dxrd.index('fi1 aligned [rad]');
#IndPhi2 = HeaderGrains3dxrd.index('fi2 aligned [rad]');
#IndPhi = HeaderGrains3dxrd.index('Fi aligned [rad]');
#IndVol = HeaderGrains3dxrd.index('Volume [um^3]');
#
#IndEps11 = HeaderGrainResults3dxrd.index('eps11_s [-]')
#IndSig11 = HeaderGrainResults3dxrd.index('sig11_s [MPa]')
#IndEps22 = HeaderGrainResults3dxrd.index('eps22_s [-]')
#IndSig22 = HeaderGrainResults3dxrd.index('sig22_s [MPa]')
#IndEps33 = HeaderGrainResults3dxrd.index('eps33_s [-]')
#IndSig33 = HeaderGrainResults3dxrd.index('sig33_s [MPa]')
#
#IndEps12 = HeaderGrainResults3dxrd.index('eps12_s [-]')
#IndSig12 = HeaderGrainResults3dxrd.index('sig12_s [MPa]')
#IndEps13 = HeaderGrainResults3dxrd.index('eps13_s [-]')
#IndSig13 = HeaderGrainResults3dxrd.index('sig13_s [MPa]')
#IndEps23 = HeaderGrainResults3dxrd.index('eps23_s [-]')
#IndSig23 = HeaderGrainResults3dxrd.index('sig23_s [MPa]')
#
#
#Phi1=[grain[IndPhi1] for grain in Grains3dxrd]
#Phi2=[grain[IndPhi2] for grain in Grains3dxrd]
#PHI=[grain[IndPhi] for grain in Grains3dxrd]
#Volume=[grain[IndVol] for grain in Grains3dxrd]
#
#Eps11 = [grain[IndEps11] for grain in GrainResults3dxrd]
#Sig11 = [grain[IndSig11] for grain in GrainResults3dxrd]
#Eps22 = [grain[IndEps22] for grain in GrainResults3dxrd]
#Sig22 = [grain[IndSig22] for grain in GrainResults3dxrd]
#Eps33 = [grain[IndEps33] for grain in GrainResults3dxrd]
#Sig33 = [grain[IndSig33] for grain in GrainResults3dxrd]
#
#Eps12 = [grain[IndEps12] for grain in GrainResults3dxrd]
#Sig12 = [grain[IndSig12] for grain in GrainResults3dxrd]
#Eps13 = [grain[IndEps13] for grain in GrainResults3dxrd]
#Sig13 = [grain[IndSig13] for grain in GrainResults3dxrd]
#Eps23 = [grain[IndEps23] for grain in GrainResults3dxrd]
#Sig23 = [grain[IndSig23] for grain in GrainResults3dxrd]
#
#meanstrain = np.array([np.mean(Eps11),np.mean(Eps22),np.mean(Eps33),
#                       2*np.mean(Eps23),2*np.mean(Eps13),2*np.mean(Eps12)])
#meanstress = np.array([np.mean(Sig11),np.mean(Sig22),np.mean(Sig33),
#                       np.mean(Sig23),np.mean(Sig13),np.mean(Sig12)])

#IndMS = []
#for ss,i in  zip(meanstrain, range(0,meanstrain.shape[0])):
#    if ss!=0:
#        IndMS.append(i)
#        
#redmeanstrain=np.zeros((len(IndMS)))
#    
#Mask = np.zeros((6,6))  
#for i,ii in zip(IndMS,range(0,len(IndMS))): 
#    redmeanstrain[ii]=meanstrain[i]
#    for j in IndMS: 
#        Mask[j,i]=1 
    
    C_eff_v = np.zeros((6,6))
    S_eff_r = np.zeros((6,6))
    TotalVol=0;
    CA, C = stiffness_tensor_cubic(C11,C12,C44)
    SA, S = compliance_tensor_cubic(C11,C12,C44)
    Sigma11_vi=[]
    Epsilon11_ri=[]
    Sigma22_vi=[]
    Epsilon22_ri=[]
    Sigma33_vi=[]
    Epsilon33_ri=[]
    Epsilon23_ri=[]
    Epsilon13_ri=[]
    Epsilon12_ri=[]
    meanstress_voigt=np.empty([3])
    stdstress_voigt=np.empty([3])
    meanstrain_reuss=np.empty([3])
    stdstrain_reuss=np.empty([6])
    
    strain_r=[]
    stress_r=[]
    stress_v=[]
    strain_v=[]
    for phi1,Fi,phi2,voli in zip(Phi1,PHI,Phi2,Volume):
        TotalVol+=voli
        #rotation matrix
        A =inverse_euler_matrix(phi1, Fi, phi2);
        Ct_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,C)
        C_rot=stiffness_from_tensor2voight_notation(Ct_rot)
        #print(C_rot)
        #C_rot = elasticity.rotate_cubic_elastic_constants(C11, C12, C44, A, tol=1e-6);
        St_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,S)
        S_rot = compliance_from_tensor2voight_notation(St_rot)
        
#        redS_rot = np.zeros((len(IndMS),len(IndMS)))
#        
#        for i,ii in zip(IndMS,range(0,len(IndMS))):
#            for j,jj in zip(IndMS,range(0,len(IndMS))):
#                redS_rot[ii,jj]=S_rot[i,j]
#                
        #stress_vired = np.matmul(np.linalg.inv(S_rot[np.ix_(IndMS,IndMS)]),meanstrain[IndMS])
        #stress_vi = np.array([0.,0.,0.,0.,0.,0.])
        #stress_vi[IndMS]=stress_vired
        #strain_vi=np.matmul(S_rot,stress_vi)
        stress_vi=np.matmul(np.linalg.inv(S_rot),meanstrain)
        stress_v.append(stress_vi)
        strain_ri = np.matmul(S_rot,meanstress)
        strain_r.append(strain_ri)
        

        C_eff_v+=C_rot*voli
        S_eff_r+=S_rot*voli
        
    C_eff_v/=TotalVol
    S_eff_r/=TotalVol
    
    mean_stress_v=[]
    mean_strain_r=[]
    std_stress_v=[]
    std_strain_r=[]
    for i in range(0,len(stress_v[0])):
        if i>2:
            fac=0.5
        else:
            fac=1.0
        mean_stress_v.append(np.mean([s[i] for s in stress_v]))
        mean_strain_r.append(np.mean([s[i]*100*fac for s in strain_r]))
        std_stress_v.append(np.std([s[i] for s in stress_v]))
        std_strain_r.append(np.std([s[i]*100*fac for s in strain_r]))
    

    return C_eff_v,S_eff_r,mean_stress_v, mean_strain_r,std_stress_v,std_strain_r
 


def stress_from_strain_3dxrd(Phi1,PHI,Phi2,C11,C12,C44, Strain,stresslim=[0.250,0.650]):

    TotalVol=0;
    CA, C = stiffness_tensor_cubic(C11,C12,C44)
    meanstress=[]
    stdstress=[]
    Stress=[]
    counter=-1    
    for phi1,Fi,phi2,strain in zip(Phi1,PHI,Phi2,Strain):
        counter+=1;
        A =inverse_euler_matrix(phi1, Fi, phi2);
        Ct_rot = np.einsum('ia,jb,kc,ld,abcd->ijkl',A, A, A, A,C)
        C_rot=stiffness_from_tensor2voight_notation(Ct_rot)
        stress = np.matmul(C_rot,strain);
        if stress[2]>=stresslim[0] and stress[2]<=stresslim[1]: 
            Stress.append(stress);




    for i in range(0,6):
        meanstress.append(np.mean([stress[i]*1000 for stress in Stress]))
        stdstress.append(np.std([stress[i]*1000 for stress in Stress]))
        
        
    return  meanstress,stdstress


def voigt_reuss_effective_constants_mapping(Phi1,PHI,Phi2,Volume, C11span,C12span,C44span, meanstrain, meanstress, Strain):
    E11_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E22_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E33_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    A = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    E11_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E22_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E33_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstress11_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress11_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstress22_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress22_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstress33_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress33_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstrain11_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain11_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstrain22_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain22_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstrain33_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain33_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstress3dxrd=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0],6]);
    stdstress3dxrd=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0],6]);

    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
    
    for k in range(0,C44span.shape[0]):
        
        C44=C44span[k]
        for j in range(0,C12span.shape[0]):
            C12=C12span[j]
            for i in range(0,C11span.shape[0]):
                print(str(k)+'/'+str(C44span.shape[0])+','+str(j)+'/'+str(C12span.shape[0])+','+str(i)+'/'+str(C11span.shape[0]))
                C11=C11span[i]
                C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss =\
                voigt_reuss_effective_constants(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress)
                
                meanstress3dxrdi,stdstress3dxrdi = stress_from_strain_3dxrd(Phi1,PHI,Phi2,C11,C12,C44, Strain,stresslim=[0.250,0.650])
                
                meanstress3dxrd[i,j,k,:] = meanstress3dxrdi;
                stdstress3dxrd[i,j,k,:] = stdstress3dxrdi;
                
                S_eff_v = np.linalg.inv(C_eff_v)
                E11_eff_v[i,j,k] = 1/S_eff_v[0,0]
                E11_eff_r[i,j,k] = 1/S_eff_r[0,0]
                E22_eff_v[i,j,k] = 1/S_eff_v[1,1]
                E22_eff_r[i,j,k] = 1/S_eff_r[1,1]    
                E33_eff_v[i,j,k] = 1/S_eff_v[2,2]
                E33_eff_r[i,j,k] = 1/S_eff_r[2,2]
                Xc[i,j] = C11;
                Yc[i,j] = C12;
                A[i,j,k]=2*C44/(C11-C12);    
                meanstress11_v[i,j,k] = meanstress_voigt[0];
                meanstress22_v[i,j,k] = meanstress_voigt[1];
                meanstress33_v[i,j,k] = meanstress_voigt[2];
                
                stdstress11_v[i,j,k] = stdstress_voigt[0];
                stdstress22_v[i,j,k] = stdstress_voigt[1];
                stdstress33_v[i,j,k] = stdstress_voigt[2];
                
                meanstrain11_r[i,j,k] = meanstrain_reuss[0];
                meanstrain22_r[i,j,k] = meanstrain_reuss[1];
                meanstrain33_r[i,j,k] = meanstrain_reuss[2];
    
                stdstrain11_r[i,j,k] = stdstrain_reuss[0];
                stdstrain22_r[i,j,k] = stdstrain_reuss[1];
                stdstrain33_r[i,j,k] = stdstrain_reuss[2];

    return Xc,Yc,A,E11_eff_v,E11_eff_r,E22_eff_v,E22_eff_r,E33_eff_v,E33_eff_r,\
meanstress11_v,meanstress22_v,meanstress33_v,stdstress11_v,stdstress22_v,stdstress33_v,\
meanstrain11_r,meanstrain22_r,meanstrain33_r,stdstrain11_r,stdstrain22_r,stdstrain33_r,meanstress3dxrd, stdstress3dxrd

def voigt_reuss_effective_constants_mapping2(Phi1,PHI,Phi2,Volume, C11span,C12span,C44span, meanstrain, meanstress, Strain):
    E11_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E22_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E33_eff_v = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    A = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    E11_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E22_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    E33_eff_r = np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstress11_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress11_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstress22_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress22_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstress33_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstress33_v=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstrain11_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain11_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstrain22_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain22_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    meanstrain33_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);
    stdstrain33_r=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0]]);

    meanstress3dxrd=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0],6]);
    stdstress3dxrd=np.empty([C11span.shape[0],C12span.shape[0],C44span.shape[0],6]);

    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
    
    for k,j,i in zip(range(0,C44span.shape[0]),range(0,C12span.shape[0]),range(0,C11span.shape[0])):
        
        C44=C44span[k]
        C12=C12span[j]
        print(str(k)+'/'+str(C44span.shape[0]))
        C11=C11span[i]
        C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss =\
        voigt_reuss_effective_constants(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress)
        
        meanstress3dxrdi,stdstress3dxrdi = stress_from_strain_3dxrd(Phi1,PHI,Phi2,C11,C12,C44, Strain,stresslim=[0.250,0.650])
        
        meanstress3dxrd[i,j,k,:] = meanstress3dxrdi;
        stdstress3dxrd[i,j,k,:] = stdstress3dxrdi;
        
        S_eff_v = np.linalg.inv(C_eff_v)
        E11_eff_v[i,j,k] = 1/S_eff_v[0,0]
        E11_eff_r[i,j,k] = 1/S_eff_r[0,0]
        E22_eff_v[i,j,k] = 1/S_eff_v[1,1]
        E22_eff_r[i,j,k] = 1/S_eff_r[1,1]    
        E33_eff_v[i,j,k] = 1/S_eff_v[2,2]
        E33_eff_r[i,j,k] = 1/S_eff_r[2,2]
        Xc[i,j] = C11;
        Yc[i,j] = C12;
        A[i,j,k]=2*C44/(C11-C12);    
        meanstress11_v[i,j,k] = meanstress_voigt[0];
        meanstress22_v[i,j,k] = meanstress_voigt[1];
        meanstress33_v[i,j,k] = meanstress_voigt[2];
        
        stdstress11_v[i,j,k] = stdstress_voigt[0];
        stdstress22_v[i,j,k] = stdstress_voigt[1];
        stdstress33_v[i,j,k] = stdstress_voigt[2];
        
        meanstrain11_r[i,j,k] = meanstrain_reuss[0];
        meanstrain22_r[i,j,k] = meanstrain_reuss[1];
        meanstrain33_r[i,j,k] = meanstrain_reuss[2];

        stdstrain11_r[i,j,k] = stdstrain_reuss[0];
        stdstrain22_r[i,j,k] = stdstrain_reuss[1];
        stdstrain33_r[i,j,k] = stdstrain_reuss[2];

    return Xc,Yc,A,E11_eff_v,E11_eff_r,E22_eff_v,E22_eff_r,E33_eff_v,E33_eff_r,\
meanstress11_v,meanstress22_v,meanstress33_v,stdstress11_v,stdstress22_v,stdstress33_v,\
meanstrain11_r,meanstrain22_r,meanstrain33_r,stdstrain11_r,stdstrain22_r,stdstrain33_r,meanstress3dxrd, stdstress3dxrd
    
def voigt_reuss_effective_constants_mapping3(Phi1,PHI,Phi2,Volume, C11span,C12span,C44span, meanstrain, meanstress, Strain):
    E11_eff_v = np.empty(C11span.shape[0]);
    E22_eff_v = np.empty(C11span.shape[0]);
    E33_eff_v = np.empty(C11span.shape[0]);

    A = np.empty(C11span.shape[0]);

    E11_eff_r = np.empty(C11span.shape[0]);
    E22_eff_r = np.empty(C11span.shape[0]);
    E33_eff_r = np.empty(C11span.shape[0]);

    meanstress11_v=np.empty(C11span.shape[0]);
    stdstress11_v=np.empty(C11span.shape[0]);
    meanstress22_v=np.empty(C11span.shape[0]);
    stdstress22_v=np.empty(C11span.shape[0]);
    meanstress33_v=np.empty(C11span.shape[0]);
    stdstress33_v=np.empty(C11span.shape[0]);

    meanstrain11_r=np.empty(C11span.shape[0]);
    stdstrain11_r=np.empty(C11span.shape[0]);
    meanstrain22_r=np.empty(C11span.shape[0]);
    stdstrain22_r=np.empty(C11span.shape[0]);
    meanstrain33_r=np.empty(C11span.shape[0]);
    stdstrain33_r=np.empty(C11span.shape[0]);

    meanstress3dxrd=np.empty([C11span.shape[0],6]);
    stdstress3dxrd=np.empty([C11span.shape[0],6]);

   
    for i in range(0,C44span.shape[0]):
        
        C44=C44span[i]
        C12=C12span[i]
        print(str(i)+'/'+str(C44span.shape[0]))
        C11=C11span[i]
        C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss =\
        voigt_reuss_effective_constants(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress)
        
        meanstress3dxrdi,stdstress3dxrdi = stress_from_strain_3dxrd(Phi1,PHI,Phi2,C11,C12,C44, Strain,stresslim=[0.250,0.650])
        
        meanstress3dxrd[i,:] = meanstress3dxrdi;
        stdstress3dxrd[i,:] = stdstress3dxrdi;
        
        S_eff_v = np.linalg.inv(C_eff_v)
        E11_eff_v[i] = 1/S_eff_v[0,0]
        E11_eff_r[i] = 1/S_eff_r[0,0]
        E22_eff_v[i] = 1/S_eff_v[1,1]
        E22_eff_r[i] = 1/S_eff_r[1,1]    
        E33_eff_v[i] = 1/S_eff_v[2,2]
        E33_eff_r[i] = 1/S_eff_r[2,2]
        A[i]=2*C44/(C11-C12);    
        meanstress11_v[i] = meanstress_voigt[0];
        meanstress22_v[i] = meanstress_voigt[1];
        meanstress33_v[i] = meanstress_voigt[2];
        
        stdstress11_v[i] = stdstress_voigt[0];
        stdstress22_v[i] = stdstress_voigt[1];
        stdstress33_v[i] = stdstress_voigt[2];
        
        meanstrain11_r[i] = meanstrain_reuss[0];
        meanstrain22_r[i] = meanstrain_reuss[1];
        meanstrain33_r[i] = meanstrain_reuss[2];

        stdstrain11_r[i] = stdstrain_reuss[0];
        stdstrain22_r[i] = stdstrain_reuss[1];
        stdstrain33_r[i] = stdstrain_reuss[2];

    return A,E11_eff_v,E11_eff_r,E22_eff_v,E22_eff_r,E33_eff_v,E33_eff_r,\
meanstress11_v,meanstress22_v,meanstress33_v,stdstress11_v,stdstress22_v,stdstress33_v,\
meanstrain11_r,meanstrain22_r,meanstrain33_r,stdstrain11_r,stdstrain22_r,stdstrain33_r,meanstress3dxrd, stdstress3dxrd
    

    
def plot_surface(Xc,Yc,Zc,vmin,vmax,xlabel,ylabel,zlabel,colormap=cm.jet,addplot='off',ax=[],s=[],fig=[],cb=[],alpha=1.):
    
    if addplot=='off':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d',proj_type = 'ortho')
        s=[]

    my_col = cm.jet((Zc-vmin)/(np.amax(Zc)-vmin))    
    
    s.append(ax.plot_surface(Xc,Yc,Zc, linewidth=0.5, edgecolors='black',cmap=colormap,antialiased=True,alpha=alpha))#,facecolors = my_col)
    
    mynorm = s[0].norm
    mynorm.vmax = vmax
    mynorm.vmin = vmin 
    
    for i in range(1,len(s)):
        s[i].set_norm(mynorm)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    m = cm.ScalarMappable(cmap=colormap)
    mticks = 9;
    m.set_array(np.linspace(vmin,vmax,mticks))
    if addplot=='off':
        cb = fig.colorbar(s[0],ax=ax,cmap=m)
    cb.set_clim(vmin,vmax)
    cb.set_ticks(np.linspace(vmin,vmax,mticks))
    cb.set_label(zlabel)
    plt.show()
    
    ax.set_aspect('equal', 'box')
    
    
    ax.set_xlim([Xc.min(),Xc.max()])
    ax.set_ylim([Yc.min(),Yc.max()])
    ax.set_zlim([vmin,vmax])
    #ax.set_zlim([Zc.min,Zc.max])
    nticks = 9;
    ax.set_xticks(np.linspace(Xc.min(),Xc.max(),nticks))
    ax.set_yticks(np.linspace(Yc.min(),Yc.max(),nticks))
    ax.set_zticks(np.linspace(vmin,vmax,nticks))
        
    return ax,s,fig, cb
    
def voigt_reuss_effective_constants_mapping_oneE(Phi1,PHI,Phi2,Volume, C11span,C12span, dirvin, Edirin, meanstrain, meanstress):
    E11_eff_v = np.empty([C11span.shape[0],C12span.shape[0]]);
    E22_eff_v = np.empty([C11span.shape[0],C12span.shape[0]]);
    E33_eff_v = np.empty([C11span.shape[0],C12span.shape[0]]);

    E11_eff_r = np.empty([C11span.shape[0],C12span.shape[0]]);
    E22_eff_r = np.empty([C11span.shape[0],C12span.shape[0]]);
    E33_eff_r = np.empty([C11span.shape[0],C12span.shape[0]]);

    meanstress11_v=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstress11_v=np.empty([C11span.shape[0],C12span.shape[0]]);
    meanstress22_v=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstress22_v=np.empty([C11span.shape[0],C12span.shape[0]]);
    meanstress33_v=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstress33_v=np.empty([C11span.shape[0],C12span.shape[0]]);

    meanstrain11_r=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstrain11_r=np.empty([C11span.shape[0],C12span.shape[0]]);
    meanstrain22_r=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstrain22_r=np.empty([C11span.shape[0],C12span.shape[0]]);
    meanstrain33_r=np.empty([C11span.shape[0],C12span.shape[0]]);
    stdstrain33_r=np.empty([C11span.shape[0],C12span.shape[0]]);

    C44span = np.empty([C11span.shape[0],C12span.shape[0]]);

    Xc = np.empty([C11span.shape[0],C12span.shape[0]]);
    Yc = np.empty([C11span.shape[0],C12span.shape[0]]);
    dirv=[0.,0.,0.]
    #compliance constant
    dirv[0] = dirvin[0]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[1] = dirvin[1]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    dirv[2] = dirvin[2]/np.sqrt(dirvin[0]**2+dirvin[1]**2+dirvin[2]**2);
    
    alpha = dirv[0]
    beta = dirv[1]
    gama = dirv[2]
    
    for j in range(0,C12span.shape[0]):
        C12=C12span[j]
        for i in range(0,C11span.shape[0]):
            print(str(j)+'/'+str(C12span.shape[0])+','+str(i)+'/'+str(C11span.shape[0]))
            C11=C11span[i]
            
            
            S11 = (C11+C12)/((C11+2*C12)*(C11-C12))
            S12 = -C12/((C11+2*C12)*(C11-C12))
           
            C44 = 1./(2.*(S11-S12)+(1./Edirin-S11)/((alpha**2)*(beta**2)+(alpha**2)*(gama**2)+(beta**2)*(gama**2)));
            
            C44span[i,j] = C44;            
            Xc[i,j] = C11;
            Yc[i,j] = C12;

            C_eff_v,S_eff_r,meanstress_voigt,stdstress_voigt,meanstrain_reuss,stdstrain_reuss =\
            voigt_reuss_effective_constants(Phi1,PHI,Phi2,Volume, C11,C12,C44, meanstrain, meanstress)
            S_eff_v = np.linalg.inv(C_eff_v)
            E11_eff_v[i,j] = 1/S_eff_v[0,0]
            E11_eff_r[i,j] = 1/S_eff_r[0,0]
            E22_eff_v[i,j] = 1/S_eff_v[1,1]
            E22_eff_r[i,j] = 1/S_eff_r[1,1]    
            E33_eff_v[i,j] = 1/S_eff_v[2,2]
            E33_eff_r[i,j] = 1/S_eff_r[2,2]
            
            meanstress11_v[i,j] = meanstress_voigt[0];
            meanstress22_v[i,j] = meanstress_voigt[1];
            meanstress33_v[i,j] = meanstress_voigt[2];
            
            stdstress11_v[i,j] = stdstress_voigt[0];
            stdstress22_v[i,j] = stdstress_voigt[1];
            stdstress33_v[i,j] = stdstress_voigt[2];
            
            meanstrain11_r[i,j] = meanstrain_reuss[0];
            meanstrain22_r[i,j] = meanstrain_reuss[1];
            meanstrain33_r[i,j] = meanstrain_reuss[2];

            stdstrain11_r[i,j] = stdstrain_reuss[0];
            stdstrain22_r[i,j] = stdstrain_reuss[1];
            stdstrain33_r[i,j] = stdstrain_reuss[2];

    return Xc,Yc,C44span, E11_eff_v,E11_eff_r,E22_eff_v,E22_eff_r,E33_eff_v,E33_eff_r,\
meanstress11_v,meanstress22_v,meanstress33_v,stdstress11_v,stdstress22_v,stdstress33_v,\
meanstrain11_r,meanstrain22_r,meanstrain33_r,stdstrain11_r,stdstrain22_r,stdstrain33_r

   
