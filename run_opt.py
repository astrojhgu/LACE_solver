#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import numpy as np
import matplotlib.pylab as plt
import dipole_ant
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr,cg,bicg,lgmres,gmres
from scipy.interpolate import interp1d
from scipy.optimize import fmin_powell

nsky=10
#用于求解的天空角度划分
angles_sol=np.linspace(1e-6, np.pi, nsky)
#用于模拟的天空角度划分
angles_sim=np.linspace(1e-6, np.pi, 1000)
#angles_sim=angles_sol
#频率通道
freqs=np.linspace(50,199,1024)*1e6

df=freqs[1]-freqs[0]
dt=10.0*24*3600.0


f0=100e6

#偶极天线长度
ant_len=1.0

#用f0归一化的频率
normalized_f=freqs/f0

#银河系幂律谱指数
alpha0=-2.7


#频率f0上的天空模型
sky_model=1000.0*np.exp(-(angles_sim-np.pi/4)**2/(2*np.radians(20.0)**2))

sky_func=interp1d(angles_sim, sky_model)

#Teor
#Teor=-0.15*np.exp(-(freqs-90e6)**2/(2*10e6**2))
Teor=0


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

print(p_sim)
#print(((p_sol-p_sim)==0).all())
#sys.exit()

#p_sol=np.zeros((len(freqs), len(angles_sol)))
#for i in range(0, p_sol.shape[0]):
#    b=dipole_ant.normalized_dipole_beam(angles_sol, freqs[i], ant_len)*np.sin#(angles_sol)+np.random.normal(scale=0.00001, size=len(angles_sol))**2
#    b/=np.sum(b)
#    p_sol[i,:]=b



#模拟观测得到的24小时平均天线温度谱
Ta=(p_sim@sky_model)*calc_spec_profile(normalized_f, [alpha0, 0])+Teor

def resid(sky, spec_param , Ta, beam, normalized_f):
    T_model=(beam@sky)*calc_spec_profile(normalized_f, spec_param)
    err=T_model/(dt*df)**0.5
    result=(T_model-Ta)/err
    return result

def split_params(x, beam):
    nsky=beam.shape[1]
    sky=x[:nsky]
    spec_param=x[nsky:]
    return sky, spec_param

def fobj(x, *args):
    Ta, beam, normalized_f=args
    sky, spec_param=split_params(x, beam)
    #这个prior只是为了确保不出现非物理的天空亮温度值，去掉之后似乎不影响求解
    prior=np.sum(((np.array(sky)<0).astype(float)*sky)**2)
    result=np.sum(resid(sky, spec_param, Ta, beam, normalized_f)**2)+prior
    print(result, spec_param)
    return result


#print((p_sim@sky_model)*calc_spec_profile(normalized_f, [alpha0, 0])-Ta)
#print(resid(sky0, [alpha0, 0], Ta, p_sol, normalized_f))


#sky0=[sky_func(a) for a in angles_sol]
sky0=sky_func(angles_sol)
print(sky0)

x0=list(sky0)+[alpha0, 0, 0, 0]
print(x0)
print(fobj(x0, Ta, p_sol, normalized_f))


solution=fmin_powell(fobj, x0, args=(Ta, p_sol, normalized_f), ftol=1e-15, maxiter=(1<<32))
sky, spec_param=split_params(solution, p_sol)

plt.plot(freqs/1e6, (Ta-(p_sol@sky)*calc_spec_profile(normalized_f, spec_param))*1e3)
plt.xlabel("freq (MHz)")
plt.ylabel("resid (mK)")

plt.show()
plt.plot(sky)
plt.show()
