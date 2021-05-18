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

    # find centers
    curves_centers = dict(list(curves.groupby(['centers'])))[1]
    curves_labels = curves.loc[:, ['labels']]

    # generate result path
    save_dir = './data/Cluster/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+ str(version)+'-Thr' + str(threshold) +'-Cap'+ str(CapabilityCap) + '-vldNum' + str(valid_MDinds) + '-k' + str(k)+'.csv'
    save_path_centers = save_dir + ID  +'-centers_' + save_path
    save_path_labels = save_dir + ID + '-labels_' + save_path

    curves_centers.to_csv(save_path_centers)
    curves_labels.to_csv(save_path_labels)

    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: demo is done: '+str(time.time()-t0))
    return save_path, J, M, C, k, vC, numC, probC, curveAvg, curves, curves_centers, curves_labels, valid_MDinds


if __name__ == '__main__':
    
    # read DUID data
    IDtotal = ['AGLSOM']  # test

    # IDtotal = ['AGLSOM', 'BBTHREE2', 'BW02', 'CETHANA', 'ER03', 'GSTONE4', 'HALLWF1', 'KAREEYA1', 'STAN-2', 'TARONG#3',
    #            'TRIBUTE', 'UTANQ14', 'VPGS6']  # total

    # set calculation time
    start = dt.datetime(2017, 11, 1, 4, 30)  # starts from 20171018
    end = dt.datetime(2017, 11, 30, 4, 0)   # ends until 20181031
    
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
        save_path,J, M, C, k, vC, numC, probC, curveAvg, curves, curves_centers, curves_labels, valid_MDinds = demo(ID, start, end, kmin, kmax, cores, curvePara, CapabilityCap, threshold, iterations, version, tolerance, pmin, pmax, part, distanceMethod)



    