#!/usr/bin/env python
# coding: utf-8

# In[11]:


# generate a vector of random numbers which obeys the given distribution.
#
# n: length of the vector
# mu: mean value
# sigma: standard deviation.
# dist: choices for the distribution, you need to implement at least normal 
#       distribution and uniform distribution.
#
# For normal distribution, you can use ``numpy.random.normal`` to generate.
# For uniform distribution, the interval to sample will be [mu - sigma/sqrt(3), mu + sigma/sqrt(3)].
import numpy as np

def generate_random_numbers(n, mu, sigma, dist="normal"):  
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
       #    return np.random.normal(mu, sigma, n)
    elif dist == "uniform":
        return np.random.uniform(mu - sigma/np.sqrt(3), mu + sigma/np.sqrt(3),n)
    else:
        raise Exception("The distribution {unknown_dist} is not implemented".format(unknown_dist=dist))
        #list.append()
# test your code:
y_test = generate_random_numbers(5, 0, 0.1, "normal")


# In[12]:


y1 = generate_random_numbers(105, 0.5, 1.0, "normal")
y2 = generate_random_numbers(105, 0.5, 1.0, "uniform")


# In[13]:


# IGD, the ordering is permitted to have replacement. 
#
#
def IGD_wr_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    x0=0
    list1=[]
    for k in range(n):
        lr=1/(n+1)
        x1=x0-lr*(x0-y[ordering[k]])
        obj=0
        for i in range(n):
            obj_new=(1/2)*((x0-y[i])**2)
            obj=obj+obj_new
        list1.append(obj)
        x0=x1
    return x0,list1
# IGD, the ordering is not permitted to have replacement.
#
#
def IGD_wo_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
  #  loss = np.zeros((n,1))
    x0=0
    lr=1/(n+1)
    list2=[]
    for k in range(n):
        x1=x0-lr*(x0-y[ordering[k]])
        obj=0
        for i in range(n):
            obj_new=(1/2)*((x0-y[i])**2)
            obj=obj+obj_new
        list2.append(obj)
        x0=x1
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    return x0,list2


# In[21]:


ite=[i for i in range (105)]
list_wr=[]
for i in range (100):
    x0,list1=IGD_wr_task1(y1)
    list_wr.append(list1)
v=0
list_wr1=[]
for i in range(105):
    v=0
    for j in range(100):
        v=v+list_wr[j][i]
    list_wr1.append(v/100)
list_wo=[]
#list_wo=np.zeros(105)
for i in range (100):
    x0,list1=IGD_wo_task1(y1)
    list_wo.append(list1)
#print(list_wr)
v=0
list_wol=[]
for i in range(105):
    v=0
    for j in range(100):
        v=v+list_wo[j][i]
    list_wol.append(v/100)


# In[22]:


from matplotlib import pyplot as plt
ite=[i for i in range (105)]
plt.plot(ite,list_wr1,'r')
plt.plot(ite,list_wol,'b')
#plt.ylim([0,20])
plt.show


# In[16]:


def IGD_wr_task2(b,y):
    n=len(b)
    ordering = np.random.choice(n, n, replace=True)
    x0=0
    list1=[]
    lr=0.05*min(b)
    for k in range(n):
        x1=x0-lr*b[ordering[k]]*(x0-y)
        obj=0
        for i in range(n):
            obj_new=(1/2)*((x0-y)**2)*b[i]
            obj=obj+obj_new
        list1.append(obj)
        x0=x1
    return x0,list1
def IGD_wo_task2(b,y):
    n=len(b)
    ordering = np.random.choice(n, n, replace=False)
    x0=0
    list1=[]
    lr=0.05*min(b)
    for k in range(n):
        x1=x0-lr*b[ordering[k]]*(x0-y)
        obj=0
        for i in range(n):
            obj_new=(1/2)*((x0-y)**2)*b[i]
            obj=obj+obj_new
        list1.append(obj)
        x0=x1
    return x0,list1


# In[17]:


b = np.random.uniform(1,2,20)
b=b.tolist()
y=1
list_wr=[]
for i in range (100):
    x0,list1=IGD_wr_task2(b,y)
    list_wr.append(list1)
v=0
list_wr1=[]
n=len(b)
for i in range(n):
    v=0
    for j in range(100):
        v=v+list_wr[j][i]
    list_wr1.append(v/100)
list_wo=[]
#list_wo=np.zeros(105)
for i in range (100):
    x0,list1=IGD_wo_task2(b,y)
    list_wo.append(list1)
#print(list_wr)
v=0
list_wol=[]
for i in range(n):
    v=0
    for j in range(100):
        v=v+list_wo[j][i]
    list_wol.append(v/100)
#    x1,list2=IGD_wo_task1(y1)
#    list_wo.append(list2)


# In[18]:


n=20
ite=[i for i in range (n)]
from matplotlib import pyplot as plt
plt.plot(ite,list_wr1,'r')
plt.plot(ite,list_wol,'b')

#plt.ylim((0.5, 0.7))
#plt.xlim((17, 19))
plt.show


# In[ ]:




