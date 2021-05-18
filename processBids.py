# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:09:41 2019

@author: Hongye Guo & Yuxuan Gu

"""

# 原始报价数据为18864*20的矩阵，对应于2017年10月18日4时30分到2018年11月15日4时
#
# 每半小时形成一条报价数据，分别为10段容量数据和10段价格数据。
#
# 首先剔除停机时段，获得机组出力的时变曲线，估计概率分布函数。 
#
# 然后剔除出力未达到总容量90%的时段数据，对剩余时段采样获得长度为100的分段报价曲线。


import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from multiprocessing import Pool
from functools import partial
import time

# 设置绘图字体类型
plt.rc("font", family="Times New Roman")

def sampling(price, cap, interval):
    
    curve = []
    for i in range(1,10):
        cap[i] = cap[i-1] + cap[i]
    tap = 0
    for j in range(len(interval)):
        for k in range(tap, 10):
            if interval[j] <= cap[k]:
                curve.append(price[k])
                tap = k
                break
    # 延长报价曲线
    if len(curve) < len(interval):
        maxP = max(price)
        for i in range(len(interval)-len(curve)):
            curve.append(maxP)
                
    return np.asarray(curve).reshape(1,-1)

def pltCap(ID, start, end):
    # 设置相对路径
    data_dir = '../data/Bid/'
    # 读取报价数据
    bid = pd.read_csv(data_dir + ID + '.txt', index_col=0, header=0)
    timestamp = pd.date_range(start='2017-10-18 04:30:00', end='2018-11-15 04:00:00', freq='30min')
    bid.index = timestamp
    # 截取给定时段
    span = pd.date_range(start, end, freq='30min')
    bid = bid.loc[span]
    bid['Cap.'] = bid.loc[:,bid.columns[0:10]].sum(axis=1) 
    MaxCap = max(bid['Cap.'])
    # 获得出力
    Cap = bid['Cap.']    
    # 生成保存路径
    save_dir = '../result/AvailCap/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 绘图
    plt.figure(figsize=(12,6))
    plt.plot(Cap, linestyle='--', marker='o', color='b')
    plt.ylim(0, MaxCap+50)
    plt.xlabel('Time Point', fontsize=16)
    plt.ylabel('Available Capacity', fontsize=18)
    plt.title('Available Capacity of %s from %s to %s' % (ID, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    plt.tick_params(labelsize=18)
    plt.grid(linestyle='--')
    save_path = save_dir+ID+'_'+start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+'.png'
    plt.savefig(save_path)
    plt.show()
    # 返回图片保存位置
    return save_path  
    
def pltHist(ID, start, end, bins=None):
    '''
    sns.distplot参数说明
        bins:矩形图数量
        hist:是否显示直方图
        kde:是否显示核函数估计图 
        Attention!! Drawing a KDE is more computationally involved than drawing a histogram
        rug:控制是否显示观察的边际毛毯
        fit:控制拟合的参数分布图形 
        不同分布对应的参数 https://blog.csdn.net/u011702002/article/details/78245804
        vertical:显示正交控制         
    '''
    # 设置相对路径
    data_dir = '../data/Bid/'
    # 读取报价数据
    bid = pd.read_csv(data_dir + ID + '.txt', index_col=0, header=0)
    timestamp = pd.date_range(start='2017-10-18 04:30:00', end='2018-11-15 04:00:00', freq='30min')
    bid.index = timestamp
    # 截取给定时段
    span = pd.date_range(start, end, freq='30min')
    bid = bid.loc[span]
    bid['Cap.'] = bid.loc[:,bid.columns[0:10]].sum(axis=1)
    # 获得出力
    Cap = bid['Cap.']    
    # 生成保存路径
    save_dir = '../result/AvailCapPDF/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 绘图
    if not bins:
        bins = 20
    plt.figure(figsize=(12,6))
    sns.distplot(Cap, bins=bins, kde=True, rug=False)
    plt.xlabel('Available Capacity', fontsize=16)
    plt.ylabel('Probability Density', fontsize=18)
    plt.title('Probability Density Function of Available Capacity of \n %s from %s to %s' % (ID, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    plt.tick_params(labelsize=18)
    plt.grid(linestyle='--')
    save_path = save_dir+ID+'_'+start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+'.png'
    plt.savefig(save_path)
    plt.show()
    return save_path

def pool_sample(i, bid, stepSeq):
    row = bid[i]
    cap = row[0:10]
    price = row[10:20]
    curve = sampling(price.copy(), cap.copy(), stepSeq)
    return curve
    
def generateCurves(ID, start, end, cores, CapabilityCap=0.9, num=100):
    t0 = time.time()
    # 设置相对路径
    data_dir = './data/Bid/'
    # 读取报价数据
    bid = pd.read_csv(data_dir + ID + '.txt', index_col=0, header=0)
    timestamp = pd.date_range(start='2017-10-18 04:30:00', end='2018-11-15 04:00:00', freq='30min')
    bid.index = timestamp
    # 设置分析时段
    span = pd.date_range(start, end, freq='30min')
    # 求时段出力
    bid = bid.loc[span]
    bid['Cap.'] = bid.loc[:,bid.columns[0:10]].sum(axis=1)   
    bidC = bid['Cap.']
    # 剔除停机时段
    bid = bid.loc[bid['Cap.'] != 0]
     # 获得最大出力
    MaxCap = max(bid['Cap.'])
    # 剔除容量未达到最大出力CapabilityCap的时段
    bid = bid.loc[bid['Cap.'] >= CapabilityCap*MaxCap]
    # 生成容量步幅序列
    stepSeq = np.linspace(0, MaxCap, num=num)
    func = partial(pool_sample, bid=bid.values, stepSeq=stepSeq)
    poolG = Pool(processes=cores)
    ret = poolG.map(func, range(len(bid.values)))
    poolG.close()  #关闭进程池，不再接受新的进程
    #print(poolG._processes)
    poolG.join()   #主进程阻塞等待子进程的退出
    #pool.terminate()
    ret = np.concatenate(ret,0)
    curves = pd.DataFrame(ret, index=bid.index, columns=range(1,num+1))
    # 生成保存路径
    save_dir = './data/Curves/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir+ID+'_'+start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+'.csv'
    save_path2 = './result/Cluster/'+ID+'_'+start.strftime("%Y%m%d")+'_'+end.strftime("%Y%m%d")+'-Capacity.csv'
    curves.to_csv(save_path)
    # bidC.to_csv(save_path2)
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: generateCurves is done: '+ str(time.time()-t0))
    return save_path
    
if __name__ == '__main__':
    
    #IDtotal = ['BW01','BARRON-1','CALL_B_1','DDPS1','ER01','GSTONE1','KPP_1','LD01','LYA1','LYA2','OSB-AG','SHGEN','STAN-1','SWAN_E','WHOE#1']
    
    IDtotal = ['AGLHAL', 'AGLSOM', 'ANGAST1', 'BALDHWF1', 'BARRON-1', 'BARRON-2', 'BASTYAN', 'BBTHREE1', 'BBTHREE2', 'BBTHREE3', 'BDL01', 'BDL02',
               'BLUFF1', 'BRAEMAR1', 'BRAEMAR2', 'BRAEMAR3', 'BRAEMAR5', 'BRAEMAR6', 'BRAEMAR7', 'BROKENH1', 'BW01', 'BW02', 'BW03', 'BW04',
               'CALL_B_1', 'CALL_B_2', 'CETHANA', 'CG1', 'CG2', 'CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G',
               'CG3', 'CG4', 'CPP_3', 'CPP_4', 'CPSA', 'DARTM1', 'DDPS1', 'DEVILS_G', 'DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER',
               'DRYCGT1', 'DRYCGT2', 'DRYCGT3', 'EILDON1', 'EILDON2', 'ER01', 'ER02', 'ER03', 'ER04', 'FISHER',
               'GORDON', 'GSTONE1', 'GSTONE2', 'GSTONE3', 'GSTONE4', 'GSTONE5', 'GSTONE6', 'GUTHEGA', 'HALLWF1', 'HALLWF2', 'HVGTS', 'JBUTTERS',
               'JLA01', 'JLA02', 'JLA03', 'JLA04', 'JLB01', 'JLB02', 'JLB03', 'KAREEYA1', 'KAREEYA2', 'KAREEYA3', 'KAREEYA4', 'KPP_1', 'LADBROK1',
               'LADBROK2', 'LD01', 'LD02', 'LD03', 'LD04', 'LEM_WIL', 'LI_WY_CA', 'LKBONNY2', 'LKBONNY3', 'LK_ECHO', 'LNGS1', 'LNGS2', 'LONSDALE',
               'LOYYB1', 'LOYYB2', 'LYA1', 'LYA2', 'LYA3', 'LYA4', 'MACARTH1', 'MACKAYGT', 'MACKNTSH', 'MCKAY1', 'MEADOWBK', 'MINTARO', 'MORTLK11',
               'MORTLK12', 'MP1', 'MP2', 'MPP_1', 'MPP_2', 'MSTUART1', 'MSTUART2', 'MSTUART3', 'MURRAY', 'NBHWF1', 'NPS', 'NYNGAN1', 'OAKEY1', 'OAKEY2',
               'OAKLAND1', 'OSB-AG', 'POAT110', 'POAT220', 'POR01', 'POR03', 'PPCCGT', 'PTSTAN1', 'PUMP1', 'PUMP2', 'QPS1', 'QPS2', 'QPS3', 'QPS4', 'QPS5',
               'REECE1', 'REECE2', 'ROMA_7', 'ROMA_8', 'SHGEN', 'SHPUMP', 'SNUG1', 'STAN-1', 'STAN-2', 'STAN-3', 'STAN-4', 'SWAN_E', 'TALWA1',
               'TARONG#1', 'TARONG#2', 'TARONG#3', 'TARONG#4', 'TARRALEA', 'TNPS1', 'TORRA1', 'TORRA2', 'TORRA3', 'TORRA4', 'TORRB1', 'TORRB2', 'TORRB3', 'TORRB4',
               'TREVALLN', 'TRIBUTE', 'TUMUT3', 'TUNGATIN', 'TVCC201', 'TVPP104', 'UPPTUMUT', 'URANQ11', 'URANQ12', 'URANQ13', 'URANQ14', 'VP5', 'VP6',
               'VPGS1', 'VPGS2', 'VPGS3', 'VPGS4', 'VPGS5', 'VPGS6', 'WHOE#1', 'WHOE#2', 'WKIEWA1', 'WKIEWA2', 'WOODLWN1', 'YABULU', 'YABULU2',
               'YWPS1', 'YWPS2', 'YWPS3', 'YWPS4']
    
    # 读取数据
    #ID = 'BW01'
    NumID = len (IDtotal)
    # 设置分析时段
    start = dt.datetime(2017,10,18,4,30)
    end = dt.datetime(2018,10,31,4,0)    
    
    for i in range(NumID):     
        ID = IDtotal[i]
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'ID: '+str(ID)+' starts!')
        # 绘制出力曲线
        # save_path1 = pltCap(ID, start, end)
        # 绘制出力的概率分布
        # save_path2 = pltHist(ID, start, end, bins=100)
        # 生成报价曲线
        save_path = generateCurves(ID, start, end, cores=18, CapabilityCap=0.5, num=100)
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'ID: '+str(ID)+' done!')
