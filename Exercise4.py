import numpy as np
import matplotlib.pyplot as plt

def OrnsteinUhlenbeck(dt, T_t, D, tau):

    
    for j in range(0,N,1):
        

        etax=np.random.normal(0,1)
        x_new=1/gamma*(randomnum[j]*F[k]+epsilonx[j])*dt+x[j]
        epsilonx_new=epsilonx[j]*(1-dt/tau)+np.sqrt(2*D*dt/tau**2)*etax
        
        if x_new>L/2:
                       
            x_new=x_new-L
                        
                        
        if x_new<-L/2:
                        
            x_new=x_new+L
        
        r2.append(x_new**2)
        
        r.append(randomnum[j]*x_new)
        
        x[j]=x_new
        
        epsilonx[j]=epsilonx_new
    
        
            
        
N=1000
tau=10
L=1000
D=10
F=[0.0,0.001,0.01,0.1,1.0]
gamma=1  
T_t=0.01*20000
dt=0.01          
Nsteps=20000
MSD_list=[]
Displ_list=[]
temps=np.linspace(0,T_t,Nsteps)
    
for k in range(0,len(F),1):
    
    epsilonx=np.zeros(N)
    
    randomnum=np.random.choice([1,-1],N)
    

    for p in range(0,len(epsilonx),1):
        
        epsilonx[p]=np.sqrt(D/tau)
    
    x=np.zeros(N)
        
    for i in range(0,Nsteps,1):
        
        
        r2=[]
            
        r=[]
        
        
        
        OrnsteinUhlenbeck(dt, T_t, D, tau)

        MSD=1/N*sum(r2)
        Displ=1/N*abs(sum(r))
        
            
            
        Displ_list.append(Displ)
        MSD_list.append(MSD)
        


temps=np.linspace(0,T_t,Nsteps)


for l in range(0,len(MSD_list),20000):
    
    MSD_plot=MSD_list[l:l+20000]
    
    print(MSD_plot)

    plt.plot(temps,MSD_plot)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('MSD')
        
plt.legend(['F=0','F=0.001','F=0.01', 'F=0.1', 'F=1'])
plt.show()
        
for m in range(0,len(Displ_list),20000):
    
    Displ_plot=Displ_list[m:m+20000]
    
    plt.plot(temps,Displ_plot)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('dx')

plt.legend(['F=0','F=0.001','F=0.01', 'F=0.1', 'F=1'])
plt.show()
    
        
        
        
        
        



        
    