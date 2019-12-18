# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:40:45 2018

@author: Renato
"""
# =============================================================================
# 
# =============================================================================
import numpy as np
import scipy.io as sci
import math as m
import matplotlib.pyplot as plt

def xdot_t1(x,u,param):

    p = x[0]
    r = x[1]
    
    da = u[0]
    dr = u[1]
    beta = u[2]
    
    Lp = param[0]
    Lr = param[1]
    Lda = param[2]
    Ldr = param[3]
    Lbeta = param[4]
    
    Np = param[5]
    Nr = param[6]
    Nda = param[7]
    Ndr = param[8]
    Nbeta = param[9]
    
    b_xpdot = param[15]
    b_xrdot = param[16]
    
    
    pdot = Lp*p + Lr*r + Lda*da + Ldr*dr + Lbeta*beta + b_xpdot    
    rdot = Np*p + Nr*r + Nda*da + Ndr*dr + Nbeta*beta + b_xrdot
    
    xdot = np.concatenate((pdot,rdot))
    
    return xdot

def y_t1(x,u,param):

    p = x[0]
    r = x[1]
    
    da = u[0]
    dr = u[1]
    beta = u[2]
    
    Lp = param[0]
    Lr = param[1]
    Lda = param[2]
    Ldr = param[3]
    Lbeta = param[4]
    
    Np = param[5]
    Nr = param[6]
    Nda = param[7]
    Ndr = param[8]
    Nbeta = param[9]
    
    Yp = param[10]
    Yr = param[11]
    Yda = param[12]
    Ydr = param[13]
    Ybeta = param[14]
    
    b_ypdot = param[17]
    b_yrdot = param[18]
    b_yay = param[19]
    b_yp = param[20]
    b_yr = param[21]
        
    pdot = Lp*p + Lr*r + Lda*da + Ldr*dr + Lbeta*beta + b_ypdot
    rdot = Np*p + Nr*r + Nda*da + Ndr*dr + Nbeta*beta + b_yrdot
    ay = Yp*p + Yr*r + Yda*da + Ydr*dr + Ybeta*beta + b_yay
    p = p + b_yp
    r = r + b_yr
    
    ydot = np.concatenate((pdot,rdot,ay,p,r))

    return ydot

def ruku4_t1(x, u1, u2, param, dt):

    # PRIMEIRO TERMO
    f1 = xdot_t1(x , u1, param)
    
    # TERMOS INTERMEDIÁRIOS
    ubarra = (u1 + u2)/2
    f2 = xdot_t1(x + f1*(dt/2), ubarra, param)
    
    f3 = xdot_t1(x + f2*(dt/2), ubarra, param)
    
    # TERMO FINAL
    f4 = xdot_t1(x + f3*dt, u2, param)
    
    # CALCULA TERMO SEGUINTE DE X
    x = x + (f1 + 2*(f2 + f3) + f4)*(dt/6)
    
    return x   

def oe(param, Z, U, x0, dt, perturbacao):
    
    N,Ny = Z.shape
    Nq = len(param)
    k = 0
    
    Y = np.zeros((N,Ny))
    while k < N:
        
        if k == 0:
            x = x0.copy()
            
        u1 = U[k,:]
        Y[k,:] = y_t1(x,u1,param)
        
        if k < (N-1):
            u2 = U[k+1,:]
            x = ruku4_t1(x, u1, u2, param, dt)
            
        k += 1
    
    R = np.zeros((Ny,Ny))
    for i in range(0,N):
        erro = Z[i,:] - Y[i,:]
        R = R + np.outer(erro,erro)
    
    R = np.diag(np.diag(R))/N    
    RI = np.linalg.inv(R)
    
    # Custo
    J = np.linalg.det(R)
    
    # Simula perturbação no pistema
    delta_p = perturbacao*abs(param)
    delta_p[delta_p < perturbacao] = perturbacao
    
    # INICIALIZA MATRIZ VARIAÇAO, VETOR GRADIENTE E MATRIZ HESSIANA
    matriz_var = np.zeros((Ny,Nq))
    Grad = np.zeros((Nq,1))
    Hessian = np.zeros((Nq,Nq))

    k = 0
    
    while k < N:
        
        if k == 0:
            xp = np.repeat(x0.reshape(2,1),Nq,axis=1)
        
        
        for i in range(Nq):
            param_p = param.copy()
            param_p[i] = param[i] + delta_p[i]
            u1 = U[k,:]
            yp = y_t1(xp[:,i],u1,param_p)
            
            deltaY = yp - Y[k,:]
            matriz_var[:,i] = deltaY/delta_p[i]
            
            if k < (N-1):
                u2 = U[k+1,:]
                xp[:,i] = ruku4_t1(xp[:,i],u1,u2,param_p,dt)
            
        erro = Z[k,:] - Y[k,:]
        erro = erro[:,None]
        Grad -= (matriz_var.T @ RI @ erro)
        Hessian += matriz_var.T @ RI @ matriz_var;
        
        k += 1
    
    return J,Grad,Hessian
            
   
    
if __name__ == "__main__":
    
    plt.close('all')
    
    Nx         = 2                         # Número de estados. 
    Ny         = 5                         # Número de variáveis observadas.
    Nu         = 3                         # Número de variáveis de input.
    NparSys    = 15                        # Numero de parâmetros do sistema.
    Nparam     = NparSys + Nx + Ny         # Número total de parâmetros considerando o 
                                           # desvio de medida e na variavel de estado.
    dt         = 0.04                      # Tempo da amostra.  
    
    print("Ex.: 4.20.1")
    print("Movimento Lateral-Direcional, Nx=2,Ny=5,Nu=3,ATTAS")
    
    # Variáveis uteis

    r2d = 180/m.pi
    d2r = m.pi/180
    
    ## Carrega Dados
    
    dic = sci.loadmat('fAttasAilRud1.mat')
    data = dic['fAttasAilRud1']
    Npontos,Nvar = np.shape(data)
    
    t = np.arange(Npontos)*dt
    
    # Variáveis de estado
    Z = np.zeros((Npontos,Ny))
    Z[:,0] = data[:,16]*d2r # pdot
    Z[:,1] = data[:,18]*d2r # rdot
    Z[:,2] = data[:,2]      # ay
    Z[:,3] = data[:,6]*d2r  # p
    Z[:,4] = data[:,8]*d2r  # r
    
    # Variáveis de controle
    U = np.zeros((Npontos,Nu))
    U[:,0] = (data[:,28] - data[:,27])*0.5*d2r    # Aileron
    U[:,1] = data[:,29]*d2r                       # Leme
    U[:,2] = data[:,13]*d2r                       # Beta
    
    # Chute Iniciais
    param0 = np.array([[-3.53744],[4.75534e-1],[-4.93279],[0.73334],[-4.34076],
                       [-0.86939e-1],[-1.69526e-1],[-1.92049e-1],[-0.74855],[1.55155],
                       [0.0364],[2.130],[0.740],[2.279],[-5.193],
                       [0.02],[0.02],
                       [0.02],[0.02],[0.02],[0.02],[0.02]])
    
    x0 = np.array([0.0,0.0])
    
    # Plota parametros
    plt.figure(1)
    plt.suptitle('Medições: variáveis de estado')
    plt.subplot(511)
    plt.plot(t,Z[:,0]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$p_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(512)
    plt.plot(t,Z[:,1]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$r_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(513)
    plt.plot(t,Z[:,2],':')
    plt.xlim((0,60))
    plt.ylabel('$a_{y}$ [$m/s^2$]')
    plt.grid(True)

    plt.subplot(514)
    plt.plot(t,Z[:,3]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$p$ [$\u00b0/s$]')
    plt.grid(True)   
    
    plt.subplot(515)
    plt.plot(t,Z[:,4]*r2d,':')
    plt.xlim((0,60))
    plt.xlabel('Tempo [s]')
    plt.ylabel('$r$ [$\u00b0/s$]')
    plt.grid(True) 
    
    plt.figure(2)
    plt.suptitle('Medições: variáveis de controle')
    plt.subplot(311)
    plt.plot(t,U[:,0]*r2d,linewidth=1.5)
    plt.xlim((0,60))
    plt.ylabel('$\u03BE$ [\u00b0]')        
    plt.grid(True)  

    plt.subplot(312)
    plt.plot(t,U[:,1]*r2d,linewidth=1.5)
    plt.xlim((0,60))
    plt.ylabel('$\u03C2$ [\u00b0]')        
    plt.grid(True)

    plt.subplot(313)
    plt.plot(t,U[:,2]*r2d,linewidth=1.5)
    plt.xlim((0,60))
    plt.xlabel('Tempo [s]')
    plt.ylabel('$\u03B2$ [\u00b0]')        
    plt.grid(True)
    
#   Calcula vetor saída do modelo com parametros iniciais
    y_inicial = np.zeros((Npontos,Ny))
    k = 0
        
    while k < Npontos:
        if k == 0:
            x = x0
            
        u1 = U[k,:]
        y_inicial[k,:] = y_t1(x,u1,param0)
        
        if k < (Npontos-1):
            u2 = U[k+1,:]
            x = ruku4_t1(x, u1, u2, param0, dt)
            
        k += 1    
    
    plt.figure(3)
    #plt.suptitle('Medições: variáveis de estado')
    plt.subplot(511)
    plt.plot(t,y_inicial[:,0]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$p_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(512)
    plt.plot(t,y_inicial[:,1]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$r_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(513)
    plt.plot(t,y_inicial[:,2],':')
    plt.xlim((0,60))
    plt.ylabel('$a_{y}$ [$m/s^2$]')
    plt.grid(True)

    plt.subplot(514)
    plt.plot(t,y_inicial[:,3]*r2d,':')
    plt.xlim((0,60))
    plt.ylabel('$p$ [$\u00b0/s$]')
    plt.grid(True)   
    
    plt.subplot(515)
    plt.plot(t,y_inicial[:,4]*r2d,':')
    plt.xlim((0,60))
    plt.xlabel('Tempo [s]')
    plt.ylabel('$r$ [$\u00b0/s$]')
    plt.grid(True)    
    
#   Parâmetros da otimização
    perturbacao = 1e-6      # Perturbacao no valor dos parametros
    tolR        = 1e-4      # Tolerância para convergência da função custo
    niter_max   = 50;       # Número de iteração maximo
    
#   Inicia processo de otmização
    itera = 0                # Contador de iterações
    custo_antigo = 1
    param = param0.copy()
    
    while itera < niter_max:
        custo_atual,Gr,Hess = oe(param, Z, U, x0, dt, perturbacao)
        dif_rel_custo = abs((custo_atual-custo_antigo)/custo_antigo)
        
        print('Iteration = {}'.format(itera+1))
        print('')
        print('Cost function value = {}'.format(custo_atual))
        print('Grad norm = {}'.format(np.linalg.norm(Gr)))
        print('_____________________________________________')
        
        if dif_rel_custo < tolR:
            print('Convergência alcançada')
            print('')
            print('==============================================')
            paramopt = param.copy()
            break
        else:
            del_param = np.linalg.solve(Hess,-Gr)
            param += del_param
            custo_antigo = custo_atual.copy()                    
        
        itera += 1
    
#   Calcula vetor saída do modelo com parametros otimizados
    Y = np.zeros((Npontos,Ny))
    k = 0
        
    while k < Npontos:
        if k == 0:
            x = x0
            
        u1 = U[k,:]
        Y[k,:] = y_t1(x,u1,paramopt)
        
        if k < (Npontos-1):
            u2 = U[k+1,:]
            x = ruku4_t1(x, u1, u2, paramopt, dt)
            
        k += 1
        
#    Plota saída
    plt.figure(4)
    #plt.suptitle('Resultado modelo')
    plt.subplot(511)
    plt.plot(t,Y[:,0]*r2d)
    plt.plot(t,Z[:,0]*r2d,':r')
    plt.xlim((0,60))
    plt.ylabel('$p_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(512)
    plt.plot(t,Y[:,1]*r2d)
    plt.plot(t,Z[:,1]*r2d,':r')
    plt.xlim((0,60))
    plt.ylabel('$r_{dot}$ [$\u00b0/s^2$]')
    plt.grid(True)
    
    plt.subplot(513)
    plt.plot(t,Y[:,2])
    plt.plot(t,Z[:,2],':r')
    plt.xlim((0,60))
    plt.ylabel('$a_{y}$ [$m/s^2$]')
    plt.grid(True)

    plt.subplot(514)
    plt.plot(t,Y[:,3]*r2d)
    plt.plot(t,Z[:,3]*r2d,':r')
    plt.xlim((0,60))
    plt.ylabel('$p$ [$\u00b0/s$]')
    plt.grid(True)   
    
    plt.subplot(515)
    plt.plot(t,Y[:,4]*r2d)
    plt.plot(t,Z[:,4]*r2d,':r')
    plt.xlim((0,60))
    plt.xlabel('Tempo [s]')
    plt.ylabel('$r$ [$\u00b0/s$]')
    plt.grid(True) 
        