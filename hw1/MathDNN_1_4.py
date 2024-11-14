# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random as r

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('pdf', 'svg')

np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2)

def GD_plot(alpha,run_time,num_of_sample):
  rt=[i for i in range(1,run_time+1)]
  for _ in range(num_of_sample):
    # randomly generating samples
    x=[r.uniform(-5,20)]
    for j in range(run_time-1):
      x.append(x[-1]-alpha*fprime(x[-1]))
    plt.plot(rt,x)
  plt.show()

def GD_freq(alpha,run_time,num_of_sample):

  num_0,num_10=0,0
  rt=[i for i in range(1,run_time+1)]
  for _ in range(num_of_sample):
    x=[r.uniform(-5,20)]
    for j in range(run_time-1):
      x.append(x[-1]-alpha*fprime(x[-1]))
    if 0.99*f(0)<=f(x[-1]) and f(x[-1])<=1.01*f(0):
      num_0+=1
    elif 0.99*f(10)<=f(x[-1]) and f(x[-1])<=1.01*f(10):
      num_10+=1
  return num_0/num_of_sample,num_10/num_of_sample

x = np.linspace(-5,20,100)
plt.plot(x,f(x), 'k')
plt.show()

GD_plot(0.01,100,100)

GD_plot(0.01,1000,100)

GD_plot(0.3,100,100)

GD_plot(0.3,1000,100)

GD_plot(4,100,100)

GD_plot(4,1000,100)

#alpha=0.01
rt_1_to_0,rt_1_to_10=GD_freq(0.01,1000,300)
#alpha=0.3
rt_2_to_0,rt_2_to_10=GD_freq(0.3,1000,300)
#alpha=4
rt_3_div=1-sum(GD_freq(4,1000,300))
print(f"If rate is 0.01, approximately {round(100*rt_1_to_0,2)} percent of samples converge to 0 and {round(100*rt_1_to_10,2)} percent of samples converge to 10.")
print(f"If rate is 0.3, approximately {round(100*rt_2_to_0,2)} percent of samples converge to 0 and {round(100*rt_2_to_10,2)} percent of samples converge to 10.")
print(f"If rate is 4, approximately {round(100*rt_3_div,2)} percent of samples diverge.")

'''
from google.colab import drive
drive.mount('/content/drive')

!wget -nc https://raw.githubusercontent.com/brpy/colab-pdf/master/colab_pdf.py
from colab_pdf import colab_pdf
colab_pdf('MATHDNN_1_4.ipynb')
'''