#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:39:06 2023

@author: lheller
"""
from projlib import  *
from crystallography_functions import *
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
class plotter:  
    def __init__(self):  
        #colorbarAttributes
        self.setAttributes(cbartitle="",vmbar=None,cbarh=0.04, cbarwfac=0.75,cbarhshift=-0.15)    
        #figureAttributes
        self.setAttributes(fig=None,suptitle=" ",datadeviders=None)
        #colormapAttributes
        self.setAttributes(colmapdata=None ,oris=None,plotmap=True, contourcol='k',nump=1001,oris2=None)#TSTR,oris,
        #projectionAttributes
        self.setAttributes(sphere='full',R2Proj=np.eye(3),stereogrid=False,stereoresolution=None,stereomesh=False)#ProjType
        #axisAttributes
        self.setAttributes(ax=None,ticks=None, ticklabels=None, cmap='jet',vm=None,contourfontsize=9,labelpad=None)
        #self.crystalAttributes(LA=np.eye(3),LrA=np.eye(3), symops=np.eye(3), recsymops=np.eye(3))
        self.setAttributes(symops=[np.eye(3)], recsymops=[np.eye(3)],printPhase2First=False)
        #self.crystalAttributes()
        #saveAttributes
        self.setAttributes(SAVE=False,fname=".\EA.png")
        #crystdirsnormsAttributes
        self.setAttributes(plotdirsnorms=True,dirs=[],dirtexthifts=[],norms=[],normtexthifts=[],phase='A',phase2='M',printcorrespasfamily=False,correspdelim='||',
                                      printasfamily=True,printcorrespascubicfamily=False,printascubicfamily=False,printcorrespondent=False,Cd=np.eye(3),Cp=np.eye(3),
                                       dy1=None,dx1=None,dy2=None,dx2=None) 
        #hbpAttributes
        self.setAttributes(scatterplot=False, scatterdata=None,scattercolscale=None,scattercolscalevm=None, scattersizescale=None,scat=[],scatteridxs=None,
                           scatterzorder=None,scattercolscaleticks=None,scatterproj=[],scattereqhkl=[],scatteroris=None,scatterLr=None,ax2annot=None,ax2annotcompres=None,
                          ax2annottension=None,scatteredgecolors='None',scatterlinewidth=0)
        #onmovetextAttributes
        self.setAttributes(annot=False,onmovetext=None,showdataasplotted=True, showdatanames=None,showdata=None)

        #save figure
        self.setAttributes(crop=False,imformats=None,tight_layout=False,
                           figparam={'dpi':300, 'facecolor':'none', 'edgecolor':'none','orientation':'portrait','transparent':'False', 'bbox_inches':'None', 'pad_inches':0.1})

    def setAttributes(self,**kwargs):    
        self.__dict__.update(kwargs)
    def plotProj(self,**kwargs):
        self.setAttributes(**kwargs)
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)
        if self.ProjType=='equalarea':
            self.equalarea=True
        else:
            self.equalarea=False

        if self.equalarea:
            if self.sphere=='half':
                self.fig,self.ax=schmidtnet_half(ax=self.ax,basedirs=False,facecolor='None')
            elif self.sphere=='triangle':                
                self.fig,self.ax=stereotriangle(ax=self.ax,basedirs=False,equalarea=self.equalarea,grid=self.stereogrid,resolution=self.stereoresolution,mesh=self.stereomesh)
            else:
                self.fig,self.ax=schmidtnet(ax=self.ax,basedirs=False,facecolor='None')
        else:    
            if self.sphere=='half':
                self.fig,self.ax=wulffnet_half(ax=self.ax,basedirs=False)
            elif self.sphere=='triangle':
                self.fig,self.ax=stereotriangle(ax=self.ax,basedirs=False,equalarea=self.equalarea,grid=self.stereogrid,resolution=self.stereoresolution,mesh=self.stereomesh)
            else:
                self.fig,self.ax=wulffnet(basedirs=False)
                
        if self.sphere=='full':    
            self.dirslabel=['[',']']
            self.normslabel=['(',')']
            if self.equalarea:
                if self.dy1 is None: self.dy1=-0.12
                if self.dx1 is None: self.dx1=-0.08
                if self.dy2 is None: self.dy2=0.02
                if self.dx2 is None: self.dx2=-0.05
            else:
                if self.dy1 is None: self.dy1=-0.1
                if self.dx1 is None: self.dx1=-0.08
                if self.dy2 is None: self.dy2=0.01
                if self.dx2 is None: self.dx2=-0.02

        else:
            self.dirslabel=['[',']']
            self.normslabel=['(',')']
            if self.sphere=='triangle':
                if self.dy1 is None:
                    if self.equalarea:
                        self.dy1=-0.04
                    else:
                        self.dy1=-0.025
                if self.dx1 is None: self.dx1=0.0
                if self.dy2 is None: self.dy2=0.01
                if self.dx2 is None: self.dx2=0.01
            elif self.sphere=='half':            
                if self.dy1 is None: self.dy1=-0.14
                if self.dx1 is None: self.dx1=-0.08
                if self.dy2 is None: self.dy2=0.02
                if self.dx2 is None: self.dx2=-0.05

        
    def plotDirsNorms(self,**kwargs):
        self.setAttributes(**kwargs)
        if self.sphere=='full':
            self.textlim=-10    
        else:
            self.textlim=0 
        dirs,normals=gen_dirs_norms(self.LA, self.LrA,self.dirs,self.norms,R2Proj=self.R2Proj, recsymops=self.recsymops,symops=self.symops)
        uvwkeys=[f"{int(d['uvw'][0])}{int(d['uvw'][1])}{int(d['uvw'][2])}" for d in dirs]
        hklkeys=[f"{int(d['hkl'][0])}{int(d['hkl'][1])}{int(d['hkl'][2])}" for d in normals]
        self.defaultdirtexthifts={key:[0,0] for key in uvwkeys}
        self.defaultnormtexthifts={key:[0,0] for key in hklkeys}
        if len(self.dirtexthifts)!=0:
            for key in self.dirtexthifts.keys():
                for i in [0,1]:
                    self.defaultdirtexthifts[key][i]+=self.dirtexthifts[key][i]
        if len(self.normtexthifts)!=0:
            for key in self.normtexthifts.keys():
                for i in [0,1]:
                    self.defaultnormtexthifts[key][i]+=self.normtexthifts[key][i]
        for key,toplot,textshiftkeys,textshifts,CdCp in zip(['uvw','hkl'],[dirs,normals],[uvwkeys,hklkeys],[self.defaultdirtexthifts,self.defaultnormtexthifts],[self.Cd,self.Cp]):
            #print(toplot)
            if self.printasfamily and not self.sphere=='full':
                brackL='\{'
                brackR='\}'
                if key=='uvw':
                    brackL='\\langle'
                    brackR='\\rangle'
            else:
                brackL='('
                brackR=')'
                if key=='uvw':
                    brackL='['
                    brackR=']'
            if self.printcorrespasfamily and not self.sphere=='full':
                brackCL='\{'
                brackCR='\}'
                if key=='uvw':
                    brackCL='\\langle'
                    brackCR='\\rangle'
            else:
                brackCL='('
                brackCR=')'
                if key=='uvw':
                    brackCL='['
                    brackCR=']'

            d2plot=[(d[self.ProjType][0:2],d[key],d['textshift'],textshiftkey) for d,textshiftkey in zip(toplot,textshiftkeys) if d['vector'][2]>=0 or np.abs(d['vector'][2])<1e-5]#np.array([d['equalarea'][0:2] for d in dirs if d['vector'][2]>=0 or np.abs(d['vector'][2])<1e-5])
            for d2p in d2plot:               
                if d2p[0][1]>=self.textlim:
                    if d2p[0][1]==0:
                        dy=self.dy1
                        dx=self.dx1
                    else:
                        dy=self.dy2
                        dx=-self.dx2
                        
                    self.ax.scatter(d2p[0][0],d2p[0][1],c='k',s=50,edgecolors='w',alpha=1,linewidths=1, zorder=5000)
                    if self.printascubicfamily:
                        d2pf=np.sort(np.abs(d2p[1]))[::-1]
                        textPh1=f'${brackL}{{{int(d2pf[0])}}}{{{int(d2pf[1])}}}{{{int(d2pf[2])}}}{brackR}^{{{self.phase}}}$'
                    else:
                        textPh1=f'${brackL}{{{int(d2p[1][0])}}}{{{int(d2p[1][1])}}}{{{int(d2p[1][2])}}}{brackR}^{{{self.phase}}}$'
                    if self.printcorrespondent:
                        d2pc=vector2miller(CdCp.dot(d2p[1]))
                        if d2pc[2]<0:
                            d2pc=-1*d2pc 
                        if self.printcorrespascubicfamily:
                            d2pc=np.sort(np.abs(d2pc))[::-1]
                            
                        textPh2=f'${brackCL}{{{int(d2pc[0])}}}{{{int(d2pc[1])}}}{{{int(d2pc[2])}}}{brackCR}^{{{self.phase2}}}$'
                        if self.printPhase2First:
                            text=f'{textPh2}{self.correspdelim}{textPh1}'
                        else:
                            text=f'{textPh1}{self.correspdelim}{textPh2}'
                    else:
                        text=f'{textPh1}'
                    text=text.replace('{-','\\overline{')#.replace('phase',phase).replace('phs2',phase2)
                    tt=self.ax.text(d2p[0][0]+dx+textshifts[d2p[3]][0],d2p[0][1]+dy+textshifts[d2p[3]][1],text,color='k', zorder=5000)
                    tt.set_bbox(dict(boxstyle='square,pad=-0.',facecolor='white', alpha=0.5, edgecolor='None'))
    def plotColormap(self,**kwargs):
        self.setAttributes(**kwargs)
        if self.oris is not None:
            if self.ProjType=='equalarea':
                equalarea=True
                if self.oris2 is not None:
                    self.poris2 = equalarea_directions(self.oris2)
                self.poris = equalarea_directions(self.oris)
            else:
                equalarea=False
                if self.oris2 is not None:
                    self.poris2 = stereoprojection_directions(self.oris2)
                self.poris = stereoprojection_directions(self.oris)
        if self.plotmap:
            #Grid data
            if not self.colmapdata is None and self.plotmap:
                if self.datadeviders is None:
                    datadeviders=[[0,self.oris.shape[1]]]
                else:
                    datadeviders=self.datadeviders
                for datadevider in datadeviders:    
                    gx,gy, gz, nummask, mask=genprojgrid(self.oris,#self.oris[:,datadevider[0]:datadevider[1]],
                                                         gdata=self.colmapdata[datadevider[0]:datadevider[1]],
                                                         nump=self.nump,proj=self.ProjType)
                    if not self.vm is None:
                        self.colormap=self.ax.pcolor(gx,gy,gz,cmap=self.cmap,vmin=self.vm[0],vmax=self.vm[1])
                    else:
                        self.colormap=self.ax.pcolor(gx,gy,gz,cmap=self.cmap)
                    if self.ticks is not None:
                        levels=self.ticks
                    else:
                        levels=9
                    if not self.vm is None:    
                        self.contours=self.ax.contour(gx,gy,gz,levels=levels,colors=self.contourcol,vmin=self.vm[0],vmax=self.vm[1])
                    else:
                        self.contours=self.ax.contour(gx,gy,gz,levels=levels,colors=self.contourcol)
    
                    self.ax.clabel(self.contours, fontsize=self.contourfontsize, inline=1)
                
        #self.ax.set_xlim([(1)*self.ax.get_xlim()[0]-0.05,1.05*self.ax.get_xlim()[1]])
        if abs(self.ax.get_xlim()[0])<0.01:
            self.ax.set_xlim([-.05+self.ax.get_xlim()[0],1.05*self.ax.get_xlim()[1]])
        else:
            self.ax.set_xlim([1.05*self.ax.get_xlim()[0],1.05*self.ax.get_xlim()[1]])
        
        if abs(self.ax.get_ylim()[0])<0.01:
            self.ax.set_ylim([-.05+self.ax.get_ylim()[0],1.05*self.ax.get_ylim()[1]])
        else:
            self.ax.set_ylim([1.05*self.ax.get_ylim()[0],1.05*self.ax.get_ylim()[1]])

    def processScatterData(self,**kwargs):  
        self.setAttributes(**kwargs)            
        if self.scatterdata is not None:
            if self.scatteroris is None:
                self.scatteroris=self.scatterdata
            self.scattereqhkl=[]
            self.scatterproj=[]
            for scatterdata in self.scatterdata:
                if self.ProjType=='equalarea':
                    equalarea=True
                    if self.sphere=='triangle':
                        scatterproj,scattereq=equalarea_intotriangle(scatterdata,geteqdirs=True)
                        scattereq=scattereq['eqdirs']
                    else:
                        scatterproj=equalarea_directions(scatterdata)
                        scattereq=scatterdata
                else:   
                    equalarea=False
                    if self.sphere=='triangle':
                        scatterproj,scattereq=stereoprojection_intotriangle(scatterdata,geteqdirs=True)
                        scattereq=scattereq['eqdirs']
                    else:
                        scatterproj=stereoprojection_directions(scatterdata)
                        scattereq=scatterdata
                if self.scatterLr is None:
                    self.scattereqhkl.append(vectors2miller(self.LrA.dot(scattereq)))
                else:
                    self.scattereqhkl.append(vectors2miller(self.scatterLr.dot(scattereq)))
                self.scatterproj.append(scatterproj)
            
    def plotScatter(self,**kwargs):  
        self.setAttributes(**kwargs)
        if self.scatterdata is None:
            self.scatterplot=False
        else:
            if self.scatteroris is None:
                self.scatteroris=self.scatterdata
            if self.scatteroris is None:
                self.processScatterData()
            if self.scatterzorder is None:
                self.scatterzorder=list(range(self.scatterproj[0].shape[1]))
            if self.scattercolscale is not None:
                c=self.scattercolscale[self.scatterzorder]
            else:
                c='k'
            if self.scattersizescale is not None:
                s=self.scattersizescale[self.scatterzorder]
            else:
                s=5
            self.scat=[]
            for scatterproj in self.scatterproj:
                if self.scattercolscalevm is not None:
                    self.scat.append(self.ax.scatter(scatterproj[0,self.scatterzorder],scatterproj[1,self.scatterzorder],s=s,c=c,linewidth=self.scatterlinewidth,edgecolors=self.scatteredgecolors, cmap=self.cmap,vmin=self.scattercolscalevm[0],vmax=self.scattercolscalevm[1]))
                else:
                    self.scat.append(self.ax.scatter(scatterproj[0,self.scatterzorder],scatterproj[1,self.scatterzorder],s=s,c=c,linewidth=self.scatterlinewidth, edgecolors=self.scatteredgecolors,cmap=self.cmap))

    def plotColorbar(self,**kwargs):
        self.setAttributes(**kwargs)
        try:
            if  self.scatterplot:
                scbar=self.scat[0]
                plotbar=True
                vmcolbar=self.vmbar#scattercolscalevm
                vmcolbarticks=self.scattercolscaleticks
            else:
                scbar=self.colormap
                plotbar=True
                vmcolbar=self.vmbar
                vmcolbarticks=self.ticks
            pos = self.ax.get_position()
            self.cbar_ax = self.fig.add_axes([pos.width*(1-self.cbarwfac)/2+pos.x0, pos.y0+self.cbarh+self.cbarhshift, pos.width*self.cbarwfac,self.cbarh])
            self.cbar = self.fig.colorbar(scbar, cax=self.cbar_ax, orientation='horizontal')  
            if self.labelpad is None:
                self.cbar.ax.set_xlabel(self.cbartitle)
            else:
                self.cbar.ax.set_xlabel(self.cbartitle,labelpad=self.labelpad)
            if not vmcolbar is None:
                self.cbar.ax.set_xlim(vmcolbar)
            if not vmcolbarticks is None:
                self.cbar.ax.set_xticks(vmcolbarticks)
            if self.ticklabels is not None:
                self.cbar.ax.set_xticklabels(self.ticklabels)
            self.cbar_ax.xaxis.set_ticks_position('bottom')
            self.cbar_ax.xaxis.set_label_position('bottom')
        except:
            pass
    
    def format_coord(self,x, y,**kwargs):
        self.setAttributes(**kwargs)
        #Conversion to hkl, uvw
        HKL=inv(self.LrA)
        UVW=inv(self.LA)
        if self.scatteridxs is None:
            scatteridxs=list(np.where(~np.isnan(self.colmapdata))[0])     
        else:
            scatteridxs=self.scatteridxs
        #on pointer motion vizualization defnition
        dp=np.sqrt((self.poris[0,:]-x)**2+(self.poris[1,:]-y)**2)
        idx=np.where(dp==min(dp))[0][0]
        if min(dp)<0.01:
            #print(idx)
            tol=100
            hkl=HKL.dot(self.R2Proj.T.dot(self.oris[:,idx]))
            #npwhere
            hkl=np.round(hkl/min(hkl[np.abs(hkl)>1/tol])*tol)/tol
            uvw=UVW.dot((self.R2Proj.T.dot(self.oris[:,idx])))
            uvw=np.round(uvw/min(uvw[np.abs(uvw)>1/tol])*tol)/tol
            basictext=f"(h,k,l)$^{{{self.phase}}}$={str(hkl).replace('[','(').replace(']',')')}$^{{{self.phase}}}$, [u,v,w]$^{{{self.phase}}}$={str(uvw)}$^{{{self.phase}}}$\n"
            if self.printcorrespondent:
                d2pcs=[]
                for CdCp,d2p in zip([self.Cp,self.Cd],[hkl,uvw]):
                    d2pc=vector2miller(CdCp.dot(d2p))
                    if d2pc[2]<0:
                        d2pc=-1*d2pc 
                    if self.printcorrespascubicfamily:
                        d2pc=np.sort(np.abs(d2pc))[::-1]
                    d2pcs.append(d2pc)
                basictext+=f"(h,k,l)$^{{{self.phase2}}}$={str(d2pcs[0]).replace('[','(').replace(']',')')}$^{{{self.phase2}}}$, [u,v,w]$^{{{self.phase2}}}$={str(d2pcs[1])}$^{{{self.phase2}}}$\n"
            if self.scatterdata is not None:
                for scdi,scattereqhkl in enumerate(self.scattereqhkl):
                    try:
                        scatteridx=scatteridxs.index(idx)
                        scatter1hkl=scattereqhkl[:,scatteridx]
                        #scatter2hkl=self.scatter2eqhkl[:,scatteridx]
                        scatter1hkl=np.sort(np.abs(scatter1hkl))[::-1]
                        try:
                            hpbpoint1.remove()
                        except:
                            pass
                    except:
                        scatter1hkl='No habit plane'  
                        #scatter2hkl='No habit plane'  
                    #basictext+=f", lenidx={len(scatteridxs)}, idx={idx}"
                    if self.ax not in self.ax2annot: 
                        basictext+=f"HBP{int(scdi+1)}={scatter1hkl}$^A$,".replace("[","(").replace("]",")")
                basictext=basictext[:-1]
                basictext+="\n"
            if self.showdatanames is not None:
                for name,data in zip(self.showdatanames,self.showdata):                    
                    basictext+=f"{name}:{np.around(data[idx],decimals=3)}\n"
            elif self.showdataasplotted:
                basictext+=f"{self.cbartitle}:{np.around(self.colmapdata[idx],decimals=3)}"
            if self.annot:    
                self.fig.texts[1].set_text(basictext)
            return basictext
    def dataShow(self,**kwargs):
        self.setAttributes(**kwargs)
        self.ax.format_coord = self.format_coord
    def onmove(self,event):  
        if event.inaxes:
            #print(dir(event))
            #self.fig.texts[1].set_text(self.format_coord(event.xdata,event.ydata))
            self.fig.texts[1].set_visible(True)
        else:
            self.fig.texts[1].set_visible(False)

    def dataAnnot(self,**kwargs):
        self.setAttributes(**kwargs)
        self.annot=True
        if self.onmovetext is None:
            self.onmovetext=self.fig.text(0.1,0.85,'',fontsize=12)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)

    def scatterDataAnnot(self,**kwargs):
        self.setAttributes(**kwargs)
        if self.scatteridxs is None:
            self.scatteridxs=list(np.where(~np.isnan(self.colmapdata))[0])     
        else:
            scatteridxs=self.scatteridxs

        #self.scatteridxs=list(np.where(~np.isnan(self.colmapdata))[0])  
        self.ANNOTS=[]
        for scatterproj in self.scattereqhkl:
            self.ANNOTS.append(self.ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w",alpha=1),zorder=100000))
            self.ANNOTS[-1].set_visible(False)
        #if self.ax2annot is None:
            #self.ax2annot=[True]*len(self.ANNOTS)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick);


    def onclick(self,event):
        if event.dblclick:
            if self.ax2annot is None or event.inaxes in self.ax2annot:
                for scat in self.scat:
                    scat.set_visible(not scat.get_visible())
            else:
                for curvei in self.ax.get_lines():
                    if curvei.get_gid()==50000:
                        curvei.remove()
                for annot in self.ANNOTS:
                    annot.set_visible(False)
                

        else:
            x=event.xdata
            y=event.ydata
            if self.ax2annot is None or event.inaxes not in self.ax2annot:
                ttt=""
                if self.oris2 is not None:
                    dp=np.sqrt((self.poris2[0,:]-x)**2+(self.poris2[1,:]-y)**2)
                    #ttt="oris2"
                else:
                    dp=np.sqrt((self.poris[0,:]-x)**2+(self.poris[1,:]-y)**2)
                    #ttt="oris"
                
                if min(dp)<0.05 and True:
                    idx=np.where(dp==min(dp))[0][0]
                #if True:
                    #for annot in self.ANNOTS:
                        #if self.ax2annot is None or self.ax in self.ax2annot:
                        #annot.xy = (0,0)
                        #annot.set_text(f'{ttt}:{min(dp)},idx:{idx},projlib:{self.scatterproj[0].shape}')
                        #annot.set_visible(True)
                if min(dp)<0.05 and True:
                    idx=np.where(dp==min(dp))[0][0]
                            
                    try:
                        for curvei in self.ax.get_lines():
                            if curvei.get_gid()==50000:
                                curvei.remove()    
                        scatteridx=self.scatteridxs.index(idx)
                        for scattereqhkl,scatterproj,annot in zip(self.scattereqhkl,self.scatterproj,self.ANNOTS):
                            scatter1hkl=scattereqhkl[:,scatteridx]
                            if self.ax2annot is None or self.ax in self.ax2annot:
                                showannot=False
                                if self.ax2annotcompres is not None and self.scattercolscale[scatteridx]<0:
                                    if self.ax in self.ax2annotcompres:
                                        showannot=True
                                if self.ax2annottension is not None and self.scattercolscale[scatteridx]>0:
                                    if self.ax in self.ax2annottension:
                                        showannot=True
                                if showannot:
                                    self.ax.plot(scatterproj[0,scatteridx],scatterproj[1,scatteridx],marker="o",markersize=8,markeredgecolor='k',markerfacecolor='k',gid=50000)
                                    annot.xy = (scatterproj[0,scatteridx],scatterproj[1,scatteridx])
                                    annot.set_text(f'Strain:{np.around(self.scattercolscale[scatteridx],decimals=1)},'+str(np.sort(np.abs(scatter1hkl))[::-1]).replace("[","(").replace("]",")")+'$^A$')
                                    annot.set_visible(True)
                                    #for scat in self.scat:
                                    #    scat.set_visible(False)
                    except:
                        pass
        
    def figsave(self,**kwargs):
        self.setAttributes(**kwargs)
        if type(self.fname)!=list:
            if self.imformats is not None:
                fnames=[]
                for imformat in self.imformats:
                    fnames.append(self.fname.replace(self.fname[self.fname.index('.')+1:],f'{imformat}'))
                self.fname=fnames
            else:
                self.fname=[self.fname]        
        for fnamei in self.fname:
            IMformat=fnamei[fnamei.index('.')+1:]
            self.figparam['format']=IMformat
            self.figsaveproc(fnamei)    
            if self.crop:
                if IMformat=='png':
                    image = Image(filename =fnamei)
                    image.trim(fuzz=0)
                    image.save(filename=fnamei)
                    image.close()
    def figsaveproc(self,fname,**kwargs):
        self.setAttributes(**kwargs)
        try:
            pp=self.figparam['format']
        except:
            self.figparam['format']=fname[fname.index('.')+1:]
        #print(self.fname)
        if self.tight_layout:
            self.fig.tight_layout()
        self.fig.savefig(fname,bbox_inches='tight',
        dpi=self.figparam['dpi'], facecolor=self.figparam['facecolor'], edgecolor=self.figparam['edgecolor'],
        orientation=self.figparam['orientation'],  format =self.figparam['format'],
        transparent=self.figparam['transparent'],  
        pad_inches= self.figparam['pad_inches']);


#colormap definition using gradients between n colors given
def get_cmap(colors,nbins=1000, name='my_cmap'):
    from matplotlib.colors import LinearSegmentedColormap
    
    #colors = [(1, 1, 1),(1, 0, 0)]  # R -> G -> B
    #n_bin = 1000  # Discretizes the interpolation into bins
    #cmap_name = 'my_list'
    cmap = LinearSegmentedColormap.from_list(name, colors, N=nbins)
    #cmap='jet'
    return cmap

def plotcolmaps(fname=None):
    if fname is not None:
        import pickle as pickle
        data1 = pickle.load(open(fname, 'rb'))
        for key in data1.keys():
            #print(key)
            exec(f'{key}=data1["{key}"]')
    fig, AX = plt.subplots(2,2)
    PP=[]
    contourZero=data1['contourZero']  
    for data,attribs in zip(data1['data2plot'],data1['attributes2use']):
        PP.append(plotter())
        for attrib in attribs:
            PP[-1].__dict__.update(attrib)
        #plotter1.__dict__.update(HBPDict)
        PP[-1].__dict__.update(data)
        PP[-1].__dict__.update({'ax2annot':[AX.flatten()[2],AX.flatten()[3]]})
        PP[-1].__dict__.update({'ax2annotcompres':[AX.flatten()[2]]})
        PP[-1].__dict__.update({'ax2annottension':[AX.flatten()[3]]})
        try:
            PP[-1].plotProj(fig=fig,ax=AX.flatten()[len(PP)-1])
        except:
            break

        PP[-1].plotDirsNorms()
        
        PP[-1].plotColormap(nump=301)
        PP[-1].processScatterData()
        if PP[-1].scatterplot:
            PP[-1].plotScatter()#cmap=cmap,vmbar=vmbar,vm=vm,ticks=ticks,scattercolscalevm=vm)    
        PP[-1].plotColorbar()
        if PP[-1].sphere=="half":
            tt2=PP[-1].ax.text(np.mean(contourZero[:,0])-0.05,np.max(contourZero[:,1])+0.,'Trans.\nstrain=0',color='k', zorder=50000)
            PP[-1].ax.plot(contourZero[:,0],contourZero[:,1],'k')
            PP[-1].__dict__.update({'showdhalf':data1['showdhalf']})
        PP[-1].dataShow()
        PP[-1].scatterDataAnnot()
        PP[-1].dataAnnot()
    for idxi in [-1,-2]:
        PP[idxi].scatterdata=PP[0].scatterdata    
        PP[idxi].processScatterData()
        PP[idxi].scattercolscale=PP[0].scattercolscale
        #PP[idxi].scattereqhkl=PP[0].scattereqhkl    
