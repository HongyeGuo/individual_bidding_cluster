# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:31:32 2019

@author: Hongye Guo

"""

# this is a demo

from pairwiseDist_2 import caldist
from pairwiseDist_2 import clip
from cluster import adaptiveKMedoids
from processBids import generateCurves

import datetime as dt
import pandas as pd 
import numpy as np
import os
import time


def demo(ID, start, end, kmin, kmax, cores, curvePara, CapabilityCap, threshold=0.05, iterations=100, version=1, tolerance=0, pmin=-100, pmax=500, part=None, distanceMethod='Wasserstein'):
    t0 = time.time()
    save_path = generateCurves(ID, start, end, cores, CapabilityCap)
    curves = pd.read_csv(save_path, index_col=0, header=0)
    
    if part:
        curves = curves[0:part]
    # 添加价格帽
    if pmin and pmax:
        curves1 = clip(curves.copy(), pmin, pmax)
    
    D, curveAvg, curveLen = caldist(curves1, cores, curvePara, distanceMethod, pmin)
    #D, curves1, curveAvg, curveLen = caldist(save_path, cores, curvePara, -100, 500)
    J, M, C, k, vC, numC, probC, valid_MDinds = adaptiveKMedoids(D, kmin, kmax, curveAvg, curveLen, threshold, cores=cores, iterations=iterations, tolerance=tolerance)
    # identify the type
    curves['labels'] = J
    # identify if it is clustering centers
    curves['centers'] = np.zeros(len(J))
    curves.loc[curves.index[M],'centers'] = 1  
    
    numC1 = np.zeros((len(curves),1))
    numC1[M,0] = numC
    curves['numC'] = numC1
    
    probC1 = np.zeros((len(curves),1))
    probC1[M,0] = probC
    curves['probC'] = probC1

    # generate result path
    save_dir = '../result/Cluster/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + ID  +'_'+start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+'-Ver'+ str(version)+'-Thr' + str(threshold) +'-Cap'+ str(CapabilityCap) + '-vldNum' + str(valid_MDinds) + '-k' + str(k)+'.csv'
    curves.to_csv(save_path)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: demo4 is done: '+str(time.time()-t0))
    return save_path,J, M, C, k, vC, numC, probC, curveAvg, curves, curves1, valid_MDinds


if __name__ == '__main__':
    
    # %%
    # read DUID data
    # ID = 'BW01','BARRON-1','CALL_B_1','DDPS1','ER01','GSTONE1','KPP_1','LD01','LYA1','LYA2','OSB-AG','SHGEN','STAN-1','SWAN_E','WHOE#1'
    # ID special: HVGTS样本过少，KPP_1自适应聚类需要算很久
    # IDtotal = ['BW01','BARRON-1','CALL_B_1','DDPS1','ER01','GSTONE1','KPP_1','LD01','LYA1','LYA2','OSB-AG','SHGEN','STAN-1','SWAN_E','WHOE#1']
    
    # IDtotal = ['AGLHAL', 'AGLSOM', 'ANGAST1', 'BALDHWF1', 'BARRON-1', 'BARRON-2', 'BASTYAN', 'BBTHREE1', 'BBTHREE2', 'BBTHREE3', 'BDL01', 'BDL02']
    # IDtotal = ['BLUFF1', 'BRAEMAR1', 'BRAEMAR2', 'BRAEMAR3', 'BRAEMAR5', 'BRAEMAR6', 'BRAEMAR7', 'BROKENH1', 'BW01', 'BW02', 'BW03', 'BW04']
    # IDtotal = ['CALL_B_1', 'CALL_B_2', 'CETHANA', 'CG1', 'CG2', 'CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G']
    # IDtotal = ['CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G', 'DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER']
    # IDtotal = ['DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER']
    # IDtotal = ['GORDON', 'GSTONE1', 'GSTONE2', 'GSTONE3', 'GSTONE4', 'GSTONE5', 'GSTONE6', 'GUTHEGA', 'HALLWF1', 'HALLWF2', 'HVGTS', 'JBUTTERS']
    # IDtotal = ['JLA01', 'JLA02', 'JLA03', 'JLA04', 'JLB01', 'JLB02', 'JLB03', 'KAREEYA1', 'KAREEYA2', 'KAREEYA3', 'KAREEYA4', 'KPP_1', 'LADBROK1']
    # IDtotal = ['LADBROK2', 'LD01', 'LD02', 'LD03', 'LD04', 'LEM_WIL', 'LI_WY_CA', 'LKBONNY2', 'LKBONNY3', 'LK_ECHO', 'LNGS1', 'LNGS2', 'LONSDALE']
    # IDtotal = ['LOYYB1', 'LOYYB2', 'LYA1', 'LYA2', 'LYA3', 'LYA4', 'MACARTH1', 'MACKAYGT', 'MACKNTSH', 'MCKAY1', 'MEADOWBK', 'MINTARO', 'MORTLK11']
    # IDtotal = ['MORTLK12', 'MP1', 'MP2', 'MPP_1', 'MPP_2', 'MSTUART1', 'MSTUART2', 'MSTUART3', 'MURRAY', 'NBHWF1', 'NPS', 'NYNGAN1', 'OAKEY1', 'OAKEY2']
    # IDtotal = ['OAKLAND1', 'OSB-AG', 'POAT110', 'POAT220', 'POR01', 'POR03', 'PPCCGT', 'PTSTAN1', 'PUMP1', 'PUMP2', 'QPS1', 'QPS2', 'QPS3', 'QPS4', 'QPS5']
    # IDtotal = ['REECE1', 'REECE2', 'ROMA_7', 'ROMA_8', 'SHGEN', 'SHPUMP', 'SNUG1', 'STAN-1', 'STAN-2', 'STAN-3', 'STAN-4', 'SWAN_E', 'TALWA1']
    # IDtotal = ['TARONG#1', 'TARONG#2', 'TARONG#3', 'TARONG#4', 'TARRALEA', 'TNPS1', 'TORRA1', 'TORRA2', 'TORRA3', 'TORRA4', 'TORRB1', 'TORRB2', 'TORRB3', 'TORRB4']
    # IDtotal = ['TREVALLN', 'TRIBUTE', 'TUMUT3', 'TUNGATIN', 'TVCC201', 'TVPP104', 'UPPTUMUT', 'URANQ11', 'URANQ12', 'URANQ13', 'URANQ14', 'VP5', 'VP6']
    # IDtotal = ['VPGS1', 'VPGS2', 'VPGS3', 'VPGS4', 'VPGS5', 'VPGS6', 'WHOE#1', 'WHOE#2', 'WKIEWA1', 'WKIEWA2', 'WOODLWN1', 'YABULU', 'YABULU2']
    # IDtotal = ['YWPS1', 'YWPS2', 'YWPS3', 'YWPS4']

    

    IDtotal = ['HALLWF1','GSTONE4','AGLSOM','CETHANA','TARONG#3','STAN-2','VPGS6','ER03','TRIBUTE','BW02','BBTHREE2','KAREEYA1']

    # set calculation time
    start = dt.datetime(2017,10,18,4,30)
    end = dt.datetime(2018,10,31,4,0) 
    
    # set parameters for clustering
    kmin = 2
    kmax = 30
    threshold = 0.01   #
    curvePara = 10000  # referenced values for clustering
    CapabilityCap = 0.5  # effective capacity limits
    tolerance = 0.001  # tolerance parameter for clustering
    pmin = -100      # price min cap
    pmax = 500       # price max cap
    part = None
    distanceMethod = 'Manhattan'  #distance calculation method: 'Wasserstein', 'Euclidean', 'Manhattan'
    
    # set parameters for clustering calculation
    iterations = 1000
    cores = 10
    version = 8  
    
    NumID = len(IDtotal)

    # %%
    for i in range(NumID):     
        ID = IDtotal[i]
        # run demo1    
        
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'ID: '+str(ID)+' starts!')
        t0 = time.time()
        save_path,J, M, C, k, vC, numC, probC, curveAvg, curves, curves1, valid_MDinds = demo(ID, start, end, kmin, kmax, cores, curvePara, CapabilityCap, threshold, iterations, version, tolerance, pmin, pmax, part, distanceMethod)



    