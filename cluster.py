# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 07:01:27 2019

@author: Hongye Guo & Yuxuan Gu

"""

# 本脚本中包含自适应聚类方法的函数，因为注释写的比较清楚，这儿就不废话了

import numpy as np
from pairwiseDist import caldist
from multiprocessing import Pool
from functools import partial
import time

def kMedoids(D, k, iterations):
    #print('k=',k)
    # 统计样本总量
    n = D.shape[0]
    # 若类别数 > 总数， 异常    
    if k > n:
        raise Exception('too many medoids')

    # 寻找 k 个初始聚类中心
    # 定义可用中心
    valid_medoid_inds = set(range(n))
    # 定义不可用中心
    invalid_medoid_inds = set([])
    # 定义重合点
    rs,cs = np.where(D==0)
    # 打乱重合点顺序
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    # 将重合点中的第一个点保留入可用中心
    for r,c in zip(rs,cs):
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)
    #print('k=',k)
    len_valid_medoid_inds = len(valid_medoid_inds)
    if k == 2:
        print('valid_medoid_inds=',len_valid_medoid_inds)
    # 如果可用的初始中心比类别数少即报错
    if k > len_valid_medoid_inds:
        print('k=',k)
        print('valid_medoid_inds=',valid_medoid_inds)
        raise Exception('too many medoids (after removing {} duplicate points)'.format(   len(invalid_medoid_inds)))
        
    # 随机选取 k 个聚类中心
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])
    Mnew = np.copy(M)

    # 初始化保存聚类中心的集合 C
    C = {}
    for t in range(iterations):
        #print('t=',t)
        # 根据距离最小原则确定每个样本所属类
        J = np.argmin(D[:,M], axis=1)
        # 获得每个类对应的样本
        for i in range(k):
            C[i] = np.where(J==i)[0]
        # 更新聚类中心
        for i in range(k):
            J = np.mean(D[np.ix_(C[i],C[i])],axis=1)
            j = np.argmin(J)
            Mnew[i] = C[i][j]
        # 如果前后两次迭代的中心不发生改变，则判断为收敛
        if np.array_equal(M, Mnew):
            break
        # 保存新的聚类中心
        M = np.copy(Mnew)
    # 获得每个样本所对应的类别
    #print('聚类完成')
    J = np.argmin(D[:,M], axis=1)
    for i in range(k):
        C[i] = np.where(J==i)[0]
     
    # 返回结果
    return J, M, C, len_valid_medoid_inds

def pool_threshold(i,D, C,M,curvePara, threshold, tolerance):
    vC = []
    d = D[C[i], M[i]]/curvePara
    if len(np.where(d > threshold)[0]) > tolerance*D.shape[0]:
    #if len(np.where(d > threshold)[0]) > tolerance*curves.shape[0]:
       vC.append(i)
    return vC

    
def adaptiveKMedoids(D, kmin, kmax, curveAvg, curveLen, threshold,cores, iterations=100, tolerance=0):
    
    vC = [1]
    k = kmin - 1
    while (len(vC) > 0)&(k < kmax) : 
        k = k + 1
        J, M, C, valid_MDinds = kMedoids(D, k, iterations)
        #print('JMC is done')
        curvePara = curveAvg*curveLen
        # 寻找包含异常样本的类别vC    
        vC = []
        t0 = time.time()
        func = partial(pool_threshold, D=D, C=C, M=M, curvePara=curvePara, threshold=threshold, tolerance=tolerance)
        #print('func')
        poolC = Pool(processes=cores)
        #print('poolC._processes',poolC._processes)
        #print('poolC')
        ret = poolC.map(func, range(k))
        poolC.close()#关闭进程池，不再接受新的进程
        poolC.join()#主进程阻塞等待子进程的退出
        #pool.terminate()
        print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: adaptiveKMedoids (k = '+str(k) + ') is done: ' +str(time.time()-t0) )
        vC = np.concatenate(ret, 0)          
            
# =============================================================================
#     if len(vC) > 0:
#         if kmin + len(vC) > kmax:
#             raise Exception('failure to converge')
#         # 在异常类内部进行第二次聚类，k'=2
#         # j是需要在原来的聚类结果后增补新类的索引
#         j = 0
#         for i in vC:            
#             # 保留原始索引
#             origin_ix = C[i]
#             # 提取类内元素准备二次聚类
#             d = D[np.ix_(C[i],C[i])]
#             tJ, tM, tC = kMedoids(d, 2, iterations)
#             # 增补新类，修改旧类            
#             C[i] = origin_ix[tC[0]]
#             C[kmin+j] = origin_ix[tC[1]]
#             M[i] = origin_ix[tM[0]]
#             M = np.append(M, origin_ix[tM[1]])
#             # J在这里更新还是在最后更新结果是有轻微差异的，因为新的类中心被引入使得原来别的类的
#             # 样本可能会被分到新产生的类中去，可以再讨论下怎么处理， anyway差别也不是特别大
#             J[origin_ix[tJ==0]] = i
#             J[origin_ix[tJ==1]] = kmin+j
#             # 更新索引
#             j = j + 1
# =============================================================================
    
    #计算每个类的比例        
    numC = np.zeros((k,))
    probC = np.zeros((k,))
    for i in range(k):
        numC[i] = len(C[i])
        probC[i] = numC[i]/len(D)
    
    print(str(time.strftime("%Y%m%d %X", time.localtime()) )+' '+'Function: adaptiveKMedoids is done')
    #J =  np.argmin(D[:,M], axis=1)         
    # 返回结果
    return J, M, C, k, vC, numC, probC, valid_MDinds


if __name__ == '__main__':
    
    path = './data/Curves/' + 'BW01_20171018_20180418.csv'
    D, curves, curveAvg , curveLen= caldist(path, 18, 100, -100, 500, 70)    
    J, M, C, k, vC, numC, probC, valid_MDinds = adaptiveKMedoids(D, 5, 30, curveAvg , curveLen, 0.01,18)    

