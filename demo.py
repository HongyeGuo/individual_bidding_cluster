# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:31:32 2019

@author: Hongye Guo & Yuxuan Gu

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
    
    # read DUID data
    IDtotal = ['AGLSOM', 'BBTHREE2', 'BW02']  # test

    # IDtotal = ['AGLSOM', 'BBTHREE2', 'BW02', 'CETHANA', 'ER03', 'GSTONE4', 'HALLWF1', 'KAREEYA1', 'STAN-2', 'TARONG#3',
    #            'TRIBUTE', 'UTANQ14', 'VPGS6']  # total

    # set calculation time
    start = dt.datetime(2017, 10, 18, 4, 30)
    end = dt.datetime(2018, 10, 31, 4, 0)
    
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



    