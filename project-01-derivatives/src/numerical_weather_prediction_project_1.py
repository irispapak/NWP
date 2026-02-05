#!/usr/bin/python3

""" This program calculates the first derivative of exp(x) and cos(x) at x=1
 for Dx=1,1/2,1/3,1/4,...,1/100 with the approximations of 1st, 2nd and 4th order. """


# function: calculation of 1st, 2nd and 4th order approximations of first derivative
def derivative(f, x, order):

    # dx values
    dx = [1/i for i in range(1,101)]
    
    # NaN array for derivatives
    der = np.nan*np.zeros(len(dx))
    
    for j in range(len(dx)):
        
        # 1st order
        if order==1:
            der[j] = (f(x+dx[j])-f(x))/dx[j]
                        
        # 2nd order
        elif order==2:
            der[j] = (f(x+dx[j])-f(x-dx[j]))/(2*dx[j])
            
        # 4th order
        elif order==4:
            der[j] = (1/3)*(2*(f(x+dx[j])-f(x-dx[j]))/dx[j] - (f(x+2*dx[j])-f(x-2*dx[j]))/(4*dx[j]) )
        
    return der

# -----------------------------------------------------------------
# main program

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

orders = [1,2,4]

dx = ['1/'+str(i) for i in range(1,101) ]

# NaN arrays for results
exp_results = np.nan*np.zeros((len(dx),3))
cos_results = np.nan*np.zeros((len(dx),3))


for i in range(len(orders)):
    
    # call function derivative
    exp_results[:,i] = derivative(np.exp, 1, orders[i])
    cos_results[:,i] = derivative(np.cos, 1, orders[i])


# -------------------------------------------------
# EXP(X)

exp_results_df = pd.DataFrame(data=exp_results, index=dx, columns=['1st order', '2nd order', '4th order'])
exp_results_df.columns.name = 'Dx'
exp_results_df['real'] = np.exp(1)

# print results
print('----------------------------------------')
print('Approximations - exp')
print('----------------------------------------')
print(exp_results_df.to_string())

# write results to file
f1 = open('exp_results.txt', 'w')
f1.write(exp_results_df.to_string())
f1.close()

# difference from real value of derivative exp(x) at x=1
exp_diff = exp_results_df-np.exp(1)
exp_diff = exp_diff.drop(labels='real', axis=1)

# print results
print('----------------------------------------')
print('Differences - exp')
print('----------------------------------------')
print(exp_diff.to_string())

# write difference to file
f2 = open('exp_diff.txt', 'w')
f2.write(exp_diff.to_string())
f2.close()


# absolute errors < 0.1, 0.01, 0.001, ...
error=[]

for i in range(1,21):
    error.append(1/(10**i))

    
exp_diff = abs(exp_diff)

errors_dx = pd.DataFrame(data=np.nan, index=error, columns=['Dx 1st order', 'Dx 2nd order', 'Dx 4th order'])
errors_dx.columns.name = 'error'        

# find which Dx correspond to absolute error < 0.1, 0.01, 0.001, ...
for i in range(len(orders)):
    
    for s in range(len(error)):
        
        try:
            # find the difference which is smaller than error[s]
            a = exp_diff[exp_diff.iloc[:,i] < error[s]]
            # put the first Dx that satisfies the condition a
            errors_dx.iloc[s,i] = a.index[0]
        except:
            IndexError
            break
        
print('--------------------------------------')
print(errors_dx.to_string())

# write errors to file
f3 = open('errors_exp.txt', 'w')
f3.write(errors_dx.to_string())
f3.close()


# plot 20 values

exp_diff = abs(exp_diff)

fig = plt.figure(1,figsize=(10,6))
x = exp_diff.index[0:20]
y1 = exp_diff.iloc[0:20, 0] #'1st order']
y2 = exp_diff.iloc[0:20, 1] #'2nd order']
y4 = exp_diff.iloc[0:20, 2] #'4th order']
names = list(exp_diff)
xticks = ['1/'+str(i) for i in range(1,21,1) ]

plt.plot(x, y1, label= names[0], marker='o', markersize=4, linewidth=1.2)
plt.plot(x, y2, label= names[1], marker='s', markersize=4, linewidth=1.2)
plt.plot(x, y4, label= names[2], marker='v', markersize=4, linewidth=1.2)

plt.xticks(xticks)
plt.xlabel('Dx', fontsize=12, labelpad=10)
plt.ylabel('Errors', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right')
plt.title('Absolute error of numerical derivative of $e^{x}$ at x=1', fontsize=15, y=1.03)

plt.savefig('plot_ex_dx20.png', dpi=300)
plt.close()


# plot 100 values
fig = plt.figure(2,figsize=(10,6))
x = exp_diff.index[0:100]
y1 = exp_diff.iloc[0:100, 0] #'1st order']
y2 = exp_diff.iloc[0:100, 1] #'2nd order']
y4 = exp_diff.iloc[0:100, 2] #'4th order']
names = list(exp_diff)
xticks = ['1/'+str(i) for i in range(1,101,10) ]

plt.plot(x, y1, label= names[0], marker='o', markersize=4, linewidth=1.2)
plt.plot(x, y2, label= names[1], marker='s', markersize=4, linewidth=1.2)
plt.plot(x, y4, label= names[2], marker='v', markersize=4, linewidth=1.2)


plt.xticks(xticks)
plt.xlabel('Dx', fontsize=12, labelpad=10)
plt.ylabel('Errors', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right')
plt.title('Absolute error of numerical derivative of $e^{x}$ at x=1', fontsize=15, y=1.03)

plt.savefig('plot_ex_dx100.png', dpi=300)
plt.close()



# -------------------------------------------------------------------------------------------------

# COS(X)

cos_results_df = pd.DataFrame(data=cos_results, index=dx, columns=['1st order', '2nd order', '4th order'])
cos_results_df.columns.name = 'Dx'
cos_results_df['real'] = -np.sin(1)

print('----------------------------------------')
print('Approximations - cos')
print('----------------------------------------')
print(cos_results_df.to_string())


# write results to file
f4 = open('cos_results.txt', 'w')
f4.write(cos_results_df.to_string())
f4.close()


# difference from real value of derivative cos(x) at x=1
cos_diff = cos_results_df-(-np.sin(1))
cos_diff = cos_diff.drop(labels='real', axis=1)

# print differences
print('----------------------------------------')
print('Differences - cos')
print('----------------------------------------')
print(cos_diff.to_string())

# write difference to file
f5= open('cos_diff.txt', 'w')
f5.write(cos_diff.to_string())
f5.close()


# absolute errors < 0.1, 0.01, 0.001, ...

error=[]

for i in range(1,21):
    error.append(1/(10**i))

    
cos_diff = abs(cos_diff)

errors_dx = pd.DataFrame(data=np.nan, index=error, columns=['Dx 1st order', 'Dx 2nd order', 'Dx 4th order'])
errors_dx.columns.name = 'error'        

# find which Dx correspond to absolute error < 0.1, 0.01, 0.001, ...
for i in range(len(orders)):
    
    for s in range(len(error)):
        
        try:
            # find the difference which is smaller than error[s]
            a = cos_diff[cos_diff.iloc[:,i] < error[s]]
            # put the first Dx that satisfies the condition a
            errors_dx.iloc[s,i] = a.index[0]
        except:
            IndexError
            break
        
print('--------------------------------------')
print(errors_dx.to_string())

# write errors to file
f6 = open('errors_cos.txt', 'w')
f6.write(errors_dx.to_string())
f6.close()


# plot 20 values
cos_diff = abs(cos_diff)

fig = plt.figure(3,figsize=(10,6))
x = cos_diff.index[0:20]
y1 = cos_diff.iloc[0:20, 0] #'1st order']
y2 = cos_diff.iloc[0:20, 1] #'2nd order']
y4 = cos_diff.iloc[0:20, 2] #'4th order']
names = list(cos_diff)
xticks = ['1/'+str(i) for i in range(1,21,1) ]

plt.plot(x, y1, label= names[0], marker='o', markersize=4, linewidth=1.2)
plt.plot(x, y2, label= names[1], marker='s', markersize=4, linewidth=1.2)
plt.plot(x, y4, label= names[2], marker='v', markersize=4, linewidth=1.2)

plt.xticks(xticks)
plt.xlabel('Dx', fontsize=12, labelpad=10)
plt.ylabel('Errors', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right')
plt.title('Absolute error of numerical derivative of $\\cos x$ at x=1', fontsize=15, y=1.03)

plt.savefig('plot_cosx_dx20.png', dpi=300)
plt.close()


# plot 100 values
cos_diff = abs(cos_diff)

fig = plt.figure(4,figsize=(10,6))
x = cos_diff.index[0:100]
y1 = cos_diff.iloc[0:100, 0] #'1st order']
y2 = cos_diff.iloc[0:100, 1] #'2nd order']
y4 = cos_diff.iloc[0:100, 2] #'4th order']
names = list(cos_diff)
xticks = ['1/'+str(i) for i in range(1,101,10) ]

plt.plot(x, y1, label= names[0], marker='o', markersize=4, linewidth=1.2)
plt.plot(x, y2, label= names[1], marker='s', markersize=4, linewidth=1.2)
plt.plot(x, y4, label= names[2], marker='v', markersize=4, linewidth=1.2)

plt.xticks(xticks)
plt.xlabel('Dx', fontsize=12, labelpad=10)
plt.ylabel('Errors', fontsize=12, labelpad=10)
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right')
plt.title('Absolute error of numerical derivative of $\\cos x$ at x=1', fontsize=15, y=1.03)

plt.savefig('plot_cosx_dx100.png', dpi=300)
plt.close()
