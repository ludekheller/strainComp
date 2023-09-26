#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:06:48 2020

@author: lheller
"""
import numpy as np

def generate_hkls(hklmax, syms,hkls=[]):
    #hklmax=3
    if len(hkls)==0:
        hkl_range = range(-hklmax,hklmax+1,1)
    else:
        hkl_range = hkls
    hkls = []
    hkls2 = {}
    
    
    for h in hkl_range :
        for k in hkl_range :
            for l in hkl_range :
                if h==0 and k==0 and l==0 : continue
            
                mhkl = (h,k,l)
                eq_els = []
                for sym in syms:
                    idxs=np.where(abs(sym)<1e-10)
                    sym[idxs[0],idxs[1]]=0.0
                    eq_els.append(tuple(sym.dot(mhkl)))
                isin=False
                if len(hkls)==0:
                    hkls.append(tuple(eq_els[0]))
                    
                unique_eq_el=[eq_els[0]]  
                for eq_el in eq_els:
                    isin2=False
                    for ueq_el in unique_eq_el:
                        if ueq_el==eq_el:
                            isin2=True
                    if not isin2:
                        unique_eq_el.append(eq_el)
                    for mplane in  hkls:
                        if (mplane==eq_el):                                                                
                            isin=True
                            #print('ok')
                            break
                if len(hkls2)==0:
                    hkls2[tuple(eq_els[0])]=unique_eq_el
                if not isin:
                    hkls.append(eq_el)
                    hkls2[eq_el]=unique_eq_el
    fam = {}
    for key in list(hkls2.keys()):
        fam.update(get_unique_families(hkls2[key]))

    return hkls,hkls2,fam

def generate_hkls02(hklmax, syms,G ,hkls=[]):
    #hklmax=3
    if len(hkls)==0:
        hkl_range = range(-hklmax,hklmax+1,1)
    else:
        hkl_range = hkls
    hkls = []
    hkls2 = {}
    
    Gi=np.linalg.inv(G)
    for h in hkl_range :
        for k in hkl_range :
            for l in hkl_range :
                if h==0 and k==0 and l==0 : continue
                
                mhkl = (h,k,l)
                eq_els = []
                for sym in syms:
                    idxs=np.where(abs(sym)<1e-10)
                    sym[idxs[0],idxs[1]]=0.0
                    eq_els.append(tuple(Gi.dot(sym.dot(G.dot(mhkl)))))
                isin=False
                if len(hkls)==0:
                    hkls.append(tuple(eq_els[0]))
                    
                unique_eq_el=[eq_els[0]]  
                for eq_el in eq_els:
                    isin2=False
                    for ueq_el in unique_eq_el:
                        if ueq_el==eq_el:
                            isin2=True
                    if not isin2:
                        unique_eq_el.append(eq_el)
                    for mplane in  hkls:
                        if (mplane==eq_el):                                                                
                            isin=True
                            # if mhkl == (1,0,2):
                            #     print(mplane)
                            #     print(eq_el)
                            #     print('ok')
                            break
                # if mhkl == (-1,0,2):
                #     print(mhkl)
                #     print(eq_el)
                #     print(unique_eq_el)
                #     print(isin)
                # if mhkl == (1,0,2):
                #     print(mhkl)
                #     print(eq_el)
                #     print(unique_eq_el)
                #     print(isin)
                # if len(hkls2)==0:
                    hkls2[tuple(eq_els[0])]=unique_eq_el
                if not isin:
                    hkls.append(eq_el)
                    hkls2[eq_el]=unique_eq_el
    fam = {}
    for key in list(hkls2.keys()):
        fam.update(get_unique_families(hkls2[key]))

    return hkls,hkls2,fam

def get_unique_families(hkls):
    import collections
    """
    Returns unique families of Miller indices. Families must be permutations
    of each other.
    Args:
        hkls ([h, k, l]): List of Miller indices.
    Returns:
        {hkl: multiplicity}: A dict with unique hkl and multiplicity.
    """

    # TODO: Definitely can be sped up.
    def is_perm(hkl1, hkl2):
        h1 = np.abs(hkl1)
        h2 = np.abs(hkl2)
        return all([i == j for i, j in zip(sorted(h1), sorted(h2))])

    unique = collections.defaultdict(list)
    for hkl1 in hkls:
        #print(hkl1)
        found = False
        for hkl2 in unique.keys():
            if is_perm(hkl1, hkl2):
                found = True
                unique[hkl2].append(hkl1)
                break
        if not found:
            unique[hkl1].append(hkl1)

    pretty_unique = {}
    for k, v in unique.items():
        pretty_unique[sorted(v)[-1]] = len(v)

    return pretty_unique


def lattice_vec(lattice_param):
    if lattice_param['type'].lower()=='cubic':
        a= lattice_param['a'];
        V = a*np.eye(3)
    elif lattice_param['type'].lower()=='tetragonal':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([0,b,0])
        V[:,2] = np.array([0,0,c])
        
    elif lattice_param['type'].lower()=='monoclinic':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        beta= lattice_param['beta'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([0,b,0])
        V[:,2] = np.array([c*np.cos(beta),0,c*np.sin(beta)])
    elif lattice_param['type'].lower()=='triclinic':
        a= lattice_param['a'];
        b= lattice_param['b'];
        c= lattice_param['c'];
        alpha= lattice_param['alpha'];
        beta= lattice_param['beta'];
        gamma= lattice_param['gamma'];
        V = np.zeros((3,3))
        V[:,0] = np.array([a,0.,0])
        V[:,1] = np.array([b*np.cos(gamma),b*np.sin(gamma),0])
        cx=c*np.cos(beta)
        cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        cz=np.sqrt(c**2-cx**2-cy**2)
        V[:,2] = np.array([cx,cy,cz])
    elif lattice_param['type'].lower()=='trigonal':
        a= lattice_param['a'];
        c= lattice_param['c'];
        V = np.zeros((3,3))
        V[:,0] = np.array([1./2.*a,-np.sqrt(3)/2.*a,0])
        V[:,1] = np.array([1./2.*a,np.sqrt(3)/2.*a,0])
        V[:,2] = np.array([0,0,c])
        
        

        
    return V[:,0], V[:,1], V[:,2]

def reciprocal_basis(a1,a2,a3):
    
    b1 = np.cross(a2,a3)/np.dot(a1,np.cross(a2,a3))
    b2 = np.cross(a3,a1)/np.dot(a2,np.cross(a3,a1))
    b3 = np.cross(a1,a2)/np.dot(a3,np.cross(a1,a2))
    return b1,b2,b3
