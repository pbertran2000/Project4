import numpy as np
import matplotlib.pyplot as plt

def OrnsteinUhlenbeck(dt, T_t, D, tau):

    
    for j in range(0,N,1):
        

        etax=np.random.normal(0,1)
        etay=np.random.normal(0,1)
        
        
        x_new=1/gamma*(F+epsilonx[j])*dt+x[j]
        y_new=1/gamma*(F+epsilony[j])*dt+y[j]
        
        
        epsilonx_new=epsilonx[j]*(1-dt/tau[k])+np.sqrt(2*D*dt/tau[k]**2)*etax
        epsilony_new=epsilony[j]*(1-dt/tau[k])+np.sqrt(2*D*dt/tau[k]**2)*etay
        
        if x_new>L/2:
                       
            x_new=x_new-L
                        
                        
        if x_new<-L/2:
                        
            x_new=x_new+L
                            
                        
        if y_new>L/2:
                        
            y_new=y_new-L
                            
                        
        if y_new<-L/2:
                        
            y_new=y_new+L
            
        mod=x_new**2+y_new**2
        
        
        
        r2.append(mod)
        
        x[j]=x_new
        y[j]=y_new
        
        epsilonx[j]=epsilonx_new
        epsilony[j]=epsilony_new
        
        if j==50:
            
            Xtraj.append(x[j])
            Ytraj.append(y[j])
    
        
            
        
N=10000
tau=[1,10,100]
L=1000
D=10
F=0
gamma=1  
T_t=0.01*1000
dt=0.01          
Nsteps=1000


for k in range(0,len(tau),1):
    
    x=np.zeros(N)
    y=np.zeros(N)
    
    Xtraj=[0]
    Ytraj=[0]
        
    epsilonx=np.zeros(N)
    
    for p in range(0,len(epsilonx),1):
        
        epsilonx[p]=np.sqrt(D/tau[k])
        
    epsilony=np.zeros(N)
    
    for o in range(0,len(epsilony),1):
        
        epsilony[o]=np.sqrt(D/tau[k])
        
    t=0
        
    MSD_list=[]
    Delta_list=[]
    CorrFunc_list=[]
    CorrFuncTeo_list=[]
        
    for i in range(0,Nsteps,1):
        
        t=t+dt
        
        r2=[]
        
        OrnsteinUhlenbeck(dt, T_t, D, tau)
        
        Delta=4*D*(t+tau[k]*(np.exp(-t/tau[k])-1))
        Delta_list.append(Delta)

        P=sum(r2)
        
        CorrFunc=1/N*np.sum(epsilonx*np.sqrt(D/tau[k]))
        
        CorrFunc_list.append(CorrFunc)
        
        CorrFuncTeo=D/tau[k]*np.exp(-t/tau[k])
        
        CorrFuncTeo_list.append(CorrFuncTeo)

        MSD=1/N*P

        MSD_list.append(MSD)
        
    temps=np.linspace(0,T_t,Nsteps)
    


    plt.plot(temps,MSD_list, color='blue', label='Experimental')
    plt.plot(temps,Delta_list, color='red', label='Theoretical')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('MSD')
    plt.show()
    
    plt.plot(Xtraj,Ytraj)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.show()
    
    plt.plot(temps,CorrFunc_list, color='blue', label='Experimental')
    plt.plot(temps,CorrFuncTeo_list, color='red', label='Theoretical')
    plt.xlabel('t')
    plt.ylabel('Correlation function')
    plt.show()        
        
        
    
    