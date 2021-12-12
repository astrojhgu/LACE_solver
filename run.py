#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pylab as plt
import dipole_ant
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr,cg,bicg,lgmres,gmres

#用于求解的天空角度划分
angles_sol=np.linspace(1e-6, np.pi, 40)
#用于模拟的天空角度划分
angles_sim=np.linspace(1e-6, np.pi, 1000)
#频率通道
freqs=np.linspace(50,199,1024)*1e6

f0=100e6

#偶极天线长度
ant_len=1.0

#用f0归一化的频率
normalized_f=freqs/f0

#银河系幂律谱指数
alpha0=-2.7


#频率f0上的天空模型
sky_model=1000.0*np.exp(-(angles_sim-np.pi/4)**2/(2*np.radians(20.0)**2))


def calc_f_matrix(f, n):
    result=np.zeros([len(f), n])
    for i in range(n):
        result[:, i]=np.log(f)**(i+1)
    return result


#根据coeff计算计算频谱轮廓
def calc_spec_profile(f, coeff):
    n=len(coeff)
    return np.exp((calc_f_matrix(f, n)@np.atleast_2d(coeff).T)[:,0])

#p_sim=np.zeros((len(freqs), len(angles_sim)))
#for i in range(0, p_sim.shape[0]):
#    p_sim[i,:]=dipole_ant.normalized_dipole_beam(angles_sim, freqs[i], ant_len)*np.sin(angles_sim)

#天线的频率、方向响应
p_sim=dipole_ant.normalized_dipole_beam(angles_sim, freqs, ant_len) #用于模拟的
p_sol=dipole_ant.normalized_dipole_beam(angles_sol,freqs, ant_len)  #用于求解的

#p_sol=np.zeros((len(freqs), len(angles_sol)))
#for i in range(0, p_sol.shape[0]):
#    b=dipole_ant.normalized_dipole_beam(angles_sol, freqs[i], ant_len)*np.sin#(angles_sol)+np.random.normal(scale=0.00001, size=len(angles_sol))**2
#    b/=np.sum(b)
#    p_sol[i,:]=b


#Teor=-0.01*np.exp(-(freqs-100e6)**2/(2*10e6**2))

#模拟观测得到的24小时平均天线温度谱
Ta=(p_sim@sky_model)*calc_spec_profile(normalized_f, [alpha0, 0])

#alpha=alpha0+0.5
#待求解alpha的迭代初始值
alpha=alpha0-1e-4
#beta=0e-6
coeff=[alpha,0]

print(alpha0, np.mean(sky_model))

coeff_opt=None #迭代过程中对应于残差最小的频谱参数
solution_opt=None #迭代过程中对应于残差最小的天空解
resid_min=1e99 #迭代过程中当前最小的残差值
for i in range(10000):
    #当前频谱轮廓
    spec_profile=calc_spec_profile(normalized_f, coeff)
    A=np.diag(spec_profile)@p_sol
    #solution,*_=lsqr(A, Ta, atol=1e-25)
    #天空解，用线性方程迭代算法求解
    solution,*_=gmres(A.T@A, A.T@Ta, tol=1e-15,atol=1e-15)

    #用多项式拟合算法求出频谱参数解
    ab_solution=lsqr(calc_f_matrix(normalized_f, len(coeff)),  np.atleast_2d(np.log(Ta/(p_sol@solution))).T)
    coeff=ab_solution[0]

    #计算当前残差
    resid=np.sum(Ta-(p_sol@solution)*calc_spec_profile(normalized_f, coeff))**2
    if resid_min>resid:
        #如果当前残差比之前最小的残差要小，就更新resid_min, solution_opt, coeff_opt
        resid_min=resid
        coeff_opt=coeff
        solution_opt=solution
    if i%100==0:
        print(coeff, np.mean(solution), resid, resid_min)


#打印结果、画图
plt.plot(angles_sol, solution_opt)
plt.plot(angles_sim, sky_model)
plt.show()
print(np.sum(Ta-(p_sol@solution_opt)*calc_spec_profile(normalized_f, coeff_opt))**2)
plt.plot(freqs/1e6, (Ta-(p_sol@solution_opt)*calc_spec_profile(normalized_f, coeff_opt))*1e3)
plt.xlabel("freq (MHz)")
plt.ylabel("resid (mK)")
plt.show()
