# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:33:39 2019

@author: Hongye Guo & Yuxuan Gu

revised on 20191010.2038, 增加了基于correlation计算距离矩阵的代码，对源代码没有调整；

"""

# 计算报价曲线之间的相似度

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from multiprocessing import Pool
from functools import partial
import time

def integration(x, y):
    '''
        本函数返回两条报价曲线之间的wasserstein距离
    '''
    pmin = min(np.min(x),np.min(y))
    x = x - pmin
    y = y - pmin
    sx = np.sum(x)
    sy = np.sum(y)
    if sx == 0 and sy == 0:
        d = 0
    else:
        d = np.abs(sx-sy)/np.sum(sx+sy)   
    return d

def pool_wasserstein(i,n,curves):
    dist = np.zeros((1,n))
    for j in range(n):
        d = wasserstein_distance(curves[i].copy(), curves[j].copy())
        dist[0,j] = d
    return dist

def pool_EuclideanD(i,n,curves):
    dist = np.zeros((1,n))
    for j in range(n):
        curves_temp = np.vstack([curves[i].copy(),curves[j].copy()])
        d = pdist(curves_temp)
        dist[0,j] = d
    return dist 

def pool_ManhatanD(i,n,curves):
    dist = np.zeros((1,n))
    for j in range(n):
        curves_temp = np.vstack([curves[i].copy(),curves[j].copy()])
        d = pdist(curves_temp, 'cityblock')
        dist[0,j] = d
    return dist 
    

def pairwise_distance(curves, cores):
    '''
       本函数返回价格曲线集距离矩阵，使用Wasserstein方法计算
    '''
    # 初始化距离矩阵  
    curves = curves.values
    n = curves.shape[0]
    dist = np.zeros((n,n))
    t0 = time.time()
    func = partial(pool_wasserstein, n=n, curves=curves)
    poolD = Pool(processes=cores)
    #print('poolD._processes',poolD._processes)
    ret = poolD.map(func, range(n))
    poolD.close()   #关闭进程池，不再接受新的进程
    #print(poolD._processes)
    poolD.join()    #主进程阻塞等待子进程的退出
    #pool.terminate()
    #print(time.time())
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: pairwise_distance is done: ' +str(time.time()-t0))
    dist = np.concatenate(ret, 0)
    return dist

def pairwise_distance_Tcor(curves, cores):
    '''
       本函数返回价格曲线集距离矩阵，使用correlation的方法进行计算
    '''
    t0 = time.time()
    curves0 = pd.DataFrame(curves.values.T, index=curves.columns, columns=curves.index)
    dist0 = curves0.corr(method='spearman').values
    dist = np.ones((dist0.shape[0],dist0.shape[0])) - dist0;
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: pairwise_distance is done: ' +str(time.time()-t0))
    return dist

def pairwise_distance_TEuc(curves, cores):
    '''
       本函数返回价格曲线集距离矩阵，使用Euclidean或Manhattan Distance的方法进行计算
    '''
    curves = curves.values
    n = curves.shape[0]
    dist = np.zeros((n,n))
    t0 = time.time()
    func = partial(pool_EuclideanD, n=n, curves=curves)
    poolD = Pool(processes=cores)
    #print('poolD._processes',poolD._processes)
    ret = poolD.map(func, range(n))
    poolD.close()   #关闭进程池，不再接受新的进程
    #print(poolD._processes)
    poolD.join()    #主进程阻塞等待子进程的退出
    #pool.terminate()
    #print(time.time())
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: pairwise_distance is done: ' +str(time.time()-t0))
    dist = np.concatenate(ret, 0)
    return dist

def pairwise_distance_TMan(curves, cores):
    '''
       本函数返回价格曲线集距离矩阵，使用Euclidean或Manhattan Distance的方法进行计算
    '''
    curves = curves.values
    n = curves.shape[0]
    dist = np.zeros((n,n))
    t0 = time.time()
    func = partial(pool_ManhatanD, n=n, curves=curves)
    poolD = Pool(processes=cores)
    #print('poolD._processes',poolD._processes)
    ret = poolD.map(func, range(n))
    poolD.close()   #关闭进程池，不再接受新的进程
    #print(poolD._processes)
    poolD.join()    #主进程阻塞等待子进程的退出
    #pool.terminate()
    #print(time.time())
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: pairwise_distance is done: ' +str(time.time()-t0))
    dist = np.concatenate(ret, 0)
    return dist

def clip(curves, pmin, pmax):
    '''
       本函数对价格曲线通过加价格帽的方式进行裁剪，使得最低报价和最高报价在给定阈值内
    '''
    # 转为 adarray 格式    
    curves0 = curves.values
    # 添加价格帽 
    curves0[np.where(curves0<pmin)] = pmin
    curves0[np.where(curves0>pmax)] = pmax
    # 以dataframe格式返回    
    return pd.DataFrame(curves0, index=curves.index, columns=curves.columns)

def caldist(path, cores=18, curvePara=100, pmin=None, pmax=None, part=None):
    # 读取报价曲线文件
    t0 = time.time()
    curves = pd.read_csv(path, index_col=0, header=0)
    #print('load curves done.')
    if part:
        curves = curves[0:part]
    # 添加价格帽
    if pmin and pmax:
        curves = clip(curves,pmin,pmax)
    #print('clip done')
    
    curves1 = np.array(curves-pmin)
    #curveAvg = np.mean(curves1)
    curveAvg = curvePara
    curveLen = curves1.shape[1]
    # 形成距离矩阵
    dist = pairwise_distance(curves.copy(), cores)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: caldist is done: ' +str(time.time()-t0))
    return dist, curves, curveAvg, curveLen

def caldist_Tcor(path, cores=18, curvePara=100, pmin=None, pmax=None, part=None):
    # 读取报价曲线文件
    t0 = time.time()
    curves = pd.read_csv(path, index_col=0, header=0)
    #print('load curves done.')
    if part:
        curves = curves[0:part]
    # 添加价格帽
    if pmin and pmax:
        curves = clip(curves,pmin,pmax)
    #print('clip done')
    
    curves1 = np.array(curves-pmin)
    #curveAvg = np.mean(curves1)
    curveAvg = curvePara
    curveLen = curves1.shape[1]
    # 形成距离矩阵
    dist = pairwise_distance_Tcor(curves.copy(), cores)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: caldist is done: ' +str(time.time()-t0))
    return dist, curves, curveAvg, curveLen


def caldist_TEuc(path, cores=18, curvePara=100, pmin=None, pmax=None, part=None):
    # 读取报价曲线文件
    t0 = time.time()
    curves = pd.read_csv(path, index_col=0, header=0)
    #print('load curves done.')
    if part:
        curves = curves[0:part]
    # 添加价格帽
    if pmin and pmax:
        curves = clip(curves,pmin,pmax)
    #print('clip done')
    
    curves1 = np.array(curves-pmin)
    #curveAvg = np.mean(curves1)
    curveAvg = curvePara
    curveLen = curves1.shape[1]
    # 形成距离矩阵
    dist = pairwise_distance_TEuc(curves.copy(), cores)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: caldist is done: ' +str(time.time()-t0))
    return dist, curves, curveAvg, curveLen

def caldist_TMan(path, cores=18, curvePara=100, pmin=None, pmax=None, part=None):
    # 读取报价曲线文件
    t0 = time.time()
    curves = pd.read_csv(path, index_col=0, header=0)
    #print('load curves done.')
    if part:
        curves = curves[0:part]
    # 添加价格帽
    if pmin and pmax:
        curves = clip(curves,pmin,pmax)
    #print('clip done')
    
    curves1 = np.array(curves-pmin)
    #curveAvg = np.mean(curves1)
    curveAvg = curvePara
    curveLen = curves1.shape[1]
    # 形成距离矩阵
    dist = pairwise_distance_TMan(curves.copy(), cores)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: caldist is done: ' +str(time.time()-t0))
    return dist, curves, curveAvg, curveLen

def DisMatGen(DUname):
    t0 = time.time()
    filename = DUname + '_20171018_20181031.csv'
    filename1 = DUname + '_20171018_20181031.xlsx'
    path = '../data/Curves/' + filename    
    distpath = '../data/PairwiseDistance/Distance_' + filename
    curvepath = '../data/PairwiseDistance/Curve_' + filename1
    curves = pd.read_csv(path, index_col=0, header=0)
    dist, curves0, curveAvg, curveLen = caldist(path, 18, 100, -100, 500, 1000) 
    distDF = pd.DataFrame(dist)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist calculation is done: ' +str(time.time()-t0))
    
    curves.to_excel(curvepath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Curve Output is done: ' +str(time.time()-t0))
    
    distDF.to_csv(distpath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist Output is done: ' +str(time.time()-t0))
    
    #return distDF, curves
    return t0

def DisMatGen_Tcor(DUname):
    t0 = time.time()
    filename = DUname + '_20171018_20181031.csv'
    filename1 = DUname + '_20171018_20181031.xlsx'
    path = '../data/Curves/' + filename    
    distpath = '../data/PairwiseDistance/Distance_' + filename
    curvepath = '../data/PairwiseDistance/Curve_' + filename1
    curves = pd.read_csv(path, index_col=0, header=0)
    dist, curves0, curveAvg, curveLen = caldist_Tcor(path, 18, 100, -100, 500, 1000) 
    distDF = pd.DataFrame(dist)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist calculation is done: ' +str(time.time()-t0))
    
    curves.to_excel(curvepath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Curve Output is done: ' +str(time.time()-t0))
    
    distDF.to_csv(distpath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist Output is done: ' +str(time.time()-t0))
    
    #return distDF, curves
    return t0, distDF

def DisMatGen_TEuc(DUname):
    t0 = time.time()
    filename = DUname + '_20171018_20181031.csv'
    filename1 = DUname + '_20171018_20181031.xlsx'
    path = '../data/Curves/' + filename    
    distpath = '../data/PairwiseDistance/Distance_' + filename
    curvepath = '../data/PairwiseDistance/Curve_' + filename1
    curves = pd.read_csv(path, index_col=0, header=0)
    dist, curves0, curveAvg, curveLen = caldist_TEuc(path, 18, 100, -100, 500) 
    distDF = pd.DataFrame(dist)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist calculation is done: ' +str(time.time()-t0))
    
    curves.to_excel(curvepath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Curve Output is done: ' +str(time.time()-t0))
    
    distDF.to_csv(distpath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist Output is done: ' +str(time.time()-t0))
    
    #return distDF, curves
    return t0, distDF

def DisMatGen_TMan(DUname):
    t0 = time.time()
    filename = DUname + '_20171018_20181031.csv'
    filename1 = DUname + '_20171018_20181031.xlsx'
    path = '../data/Curves/' + filename    
    distpath = '../data/PairwiseDistance/Distance_' + filename
    curvepath = '../data/PairwiseDistance/Curve_' + filename1
    curves = pd.read_csv(path, index_col=0, header=0)
    dist, curves0, curveAvg, curveLen = caldist_TMan(path, 18, 100, -100, 500) 
    distDF = pd.DataFrame(dist)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist calculation is done: ' +str(time.time()-t0))
    
    curves.to_excel(curvepath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Curve Output is done: ' +str(time.time()-t0))
    
    distDF.to_csv(distpath)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Dist Output is done: ' +str(time.time()-t0))
    
    #return distDF, curves
    return t0, distDF
    
if __name__ == '__main__':
    
#    IDtotal = ['AGLHAL', 'AGLSOM', 'ANGAST1', 'BALDHWF1', 'BARRON-1', 'BARRON-2', 'BASTYAN', 'BBTHREE1', 'BBTHREE2', 'BBTHREE3', 'BDL01', 'BDL02',
#               'BLUFF1', 'BRAEMAR1', 'BRAEMAR2', 'BRAEMAR3', 'BRAEMAR5', 'BRAEMAR6', 'BRAEMAR7', 'BROKENH1', 'BW01', 'BW02', 'BW03', 'BW04',
#               'CALL_B_1', 'CALL_B_2', 'CETHANA', 'CG1', 'CG2', 'CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G',
#               'CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G', 'DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER',
#               'DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER',
#               'GORDON', 'GSTONE1', 'GSTONE2', 'GSTONE3', 'GSTONE4', 'GSTONE5', 'GSTONE6', 'GUTHEGA', 'HALLWF1', 'HALLWF2', 'HVGTS', 'JBUTTERS',
#               'JLA01', 'JLA02', 'JLA03', 'JLA04', 'JLB01', 'JLB02', 'JLB03', 'KAREEYA1', 'KAREEYA2', 'KAREEYA3', 'KAREEYA4', 'KPP_1', 'LADBROK1',
#               'LADBROK2', 'LD01', 'LD02', 'LD03', 'LD04', 'LEM_WIL', 'LI_WY_CA', 'LKBONNY2', 'LKBONNY3', 'LK_ECHO', 'LNGS1', 'LNGS2', 'LONSDALE',
#               'LOYYB1', 'LOYYB2', 'LYA1', 'LYA2', 'LYA3', 'LYA4', 'MACARTH1', 'MACKAYGT', 'MACKNTSH', 'MCKAY1', 'MEADOWBK', 'MINTARO', 'MORTLK11',
#               'MORTLK12', 'MP1', 'MP2', 'MPP_1', 'MPP_2', 'MSTUART1', 'MSTUART2', 'MSTUART3', 'MURRAY', 'NBHWF1', 'NPS', 'NYNGAN1', 'OAKEY1', 'OAKEY2',
#               'OAKLAND1', 'OSB-AG', 'POAT110', 'POAT220', 'POR01', 'POR03', 'PPCCGT', 'PTSTAN1', 'PUMP1', 'PUMP2', 'QPS1', 'QPS2', 'QPS3', 'QPS4', 'QPS5',
#               'REECE1', 'REECE2', 'ROMA_7', 'ROMA_8', 'SHGEN', 'SHPUMP', 'SNUG1', 'STAN-1', 'STAN-2', 'STAN-3', 'STAN-4', 'SWAN_E', 'TALWA1',
#               'TARONG#1', 'TARONG#2', 'TARONG#3', 'TARONG#4', 'TARRALEA', 'TNPS1', 'TORRA1', 'TORRA2', 'TORRA3', 'TORRA4', 'TORRB1', 'TORRB2', 'TORRB3', 'TORRB4',
#               'TREVALLN', 'TRIBUTE', 'TUMUT3', 'TUNGATIN', 'TVCC201', 'TVPP104', 'UPPTUMUT', 'URANQ11', 'URANQ12', 'URANQ13', 'URANQ14', 'VP5', 'VP6',
#               'VPGS1', 'VPGS2', 'VPGS3', 'VPGS4', 'VPGS5', 'VPGS6', 'WHOE#1', 'WHOE#2', 'WKIEWA1', 'WKIEWA2', 'WOODLWN1', 'YABULU', 'YABULU2',
#               'YWPS1', 'YWPS2', 'YWPS3', 'YWPS4']
    
    IDtotal = ['GSTONE4']
    
    # 读取数据
    # DUname = 'BW01'
    NumID = len (IDtotal)
    
    for i in range(NumID):     
        DUname = IDtotal[i]
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'ID: '+str(DUname)+' starts!')
        t0, distDF = DisMatGen_TMan(DUname)
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'ID: '+str(DUname)+' done!')
    





