#!/usr/bin/python3

""" This program calculates the time derivative and space derivative with method CTCS,
 except for the first timestep in which FTCS is used. """

# define function
def init_condition(q, x, t):

    c = 1
    if q == 1 or q == 2:
        # function 1
        f = 0.5*(1-np.cos(4*np.pi*(x-c*t)))
    else:
        # function 2
        f = 1
              
    return f

# main program
import numpy as np
import matplotlib.pyplot as plt

# constant values
c = 1
N = 40 
dx = 1/N

# initialise arrays (0:N+1)
phi0 = np.zeros(N+2)        # n-1 timestep
phi1 = np.zeros(N+2)        # n timestep
phi2 = np.zeros(N+2)        # n+1 timestep
real = np.zeros(N+2)        # real solution
error = np.zeros(N+2)       # error = approximation - real


# for 3 questions
for q in [1,2,3] :

    if q == 1 or q == 3:
        dt = 0.1*dx/c
    else:
        dt = 2*dx/c

    # initial conditions t=0
    for j in range(N+2):     # for 0 until N+1 points
        x = j*dx
        t = 0
        if x<=0.5 and x>=0:
            phi1[j] = init_condition(q, x, t)
        else:
            phi1[j] = 0
            
    # for 1 until 200 timesteps
    for n in range(1,251):
        # for 1 until N points
        for j in range(1,N+1):  
           
            # approximation
            if n==1:      # 1st timestep
                # FTCS
                phi2[j] = phi1[j]-(c*dt*(phi1[j+1]-phi1[j-1]))/(2*dx)
                
            else:
                # CTCS
                phi2[j] = phi0[j]-(c*dt/dx)*(phi1[j+1]-phi1[j-1])

        # boundary conditions
        phi2[0] = phi2[N]
        phi2[N+1] = phi2[1]     
   
        # real solution
        for k in range(0,N+2):
            x = k*dx
            t = n*dt

            # set window of function
            xstart = 0 + c*t
            xend = 0.5 + c*t
            
            if xend >= 1:
                xend = xend - int(xend/1)
            if xstart >= 1:
                xstart = xstart - int(xstart/1)

            if x>=xstart and x<=xend:
                real[k] = init_condition(q,x,t)
            elif x<=xend and x<xstart and xend<xstart:
                real[k] = init_condition(q,x,t)
            elif x>xend and x>=xstart and xend<xstart:
                real[k] = init_condition(q,x,t)
            elif x<xstart and x<xend and xstart<xend:
                real[k]=0
            elif x>xend and x<xstart and xend<xstart:
                real[k]=0
            elif x>xend and x>xstart and xstart<xend:
                real[k]=0


        # calculate error
        error = phi2-real
    
        # save old values
        phi0[:] = phi1[:]
        phi1[:] = phi2[:]
               
        # plot function
        x = np.linspace(0,1,N)
        plt.clf()
        plt.plot(x, phi1[0:N], color='blue', label='approx')
        plt.plot(x, real[0:N], color='red', label='real')
        plt.ylabel('f(x)')
        plt.xlabel('x')
        
        if q == 1 or q == 2:
            plt.ylim(-0.2,1.2)
            plt.title('Function 1')
        else:
            plt.ylim(-0.8,1.8)
            plt.title('Function 2')
            
        plt.legend(loc='upper right')
        plt.savefig('fun_'+str(q)+'/function'+str(q)+'_'+str(n)+'.jpg')
        
        # plot error
        x = np.linspace(0,1,N+2)
        plt.clf()
        plt.plot(x,error)
        plt.ylabel('error')
        plt.xlabel('x')
        
        if q == 1 or q == 2:
            plt.ylim(-1.2,1.2)
            plt.title('Error 1')
        else:
            plt.ylim(-2,2)
            plt.title('Error 2')
            
        plt.savefig('err_'+str(q)+'/error'+str(q)+'_'+str(n)+'.jpg')


# turn into gif            
# os.system('convert -delay 5 -loop 0 `ls -v` pulse.gif')

