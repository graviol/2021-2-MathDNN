# -*- coding: utf-8 -*-

import numpy as np

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('pdf', 'svg')

class Convolution1d :
    def __init__(self, filt) :
        self.__filt = filt
        self.__r = filt.size
        self.T = TransposedConvolution1d(self.__filt)

    def __matmul__(self, vector) :
        r, n = self.__r, vector.size
        
        return np.asarray([sum([self.__filt[k]*vector[k+i] for k in range(r)]) for i in range(n-r+1)])  # IMPLEMENT THIS

class TransposedConvolution1d :
    
    # Transpose of 1-dimensional convolution operator used for the transpose-convolution operation A.T@(...)
  
  def __init__(self, filt) :
    self.__filt = filt
    self.__r = filt.size
    
    # Goal : Implementing matrix multiplication

  def __matmul__(self, vector) :
    r = self.__r
    n = vector.size + r - 1

    return np.asarray([sum([self.__filt[k]*vector[i-k] for k in range(0,i+1) if k<=r-1 and i-k>=0 and i-k<=n-r]) for i in range(n)])  # IMPLEMENT THIS

def huber_loss(x) :
    return np.sum( (1/2)*(x**2)*(np.abs(x)<=1) + (np.sign(x)*x-1/2)*(np.abs(x)>1) )
    
def huber_grad(x) :
    return x*(np.abs(x)<=1) + np.sign(x)*(np.abs(x)>1)

r, n, lam = 3, 20, 0.1

np.random.seed(0)
k = np.random.randn(r) # randn(r) : normalized seeds of r elements
b = np.random.randn(n-r+1)
A = Convolution1d(k)
#from scipy.linalg import circulant
#A = circulant(np.concatenate((np.flip(k),np.zeros(n-r))))[2:,:]


x = np.zeros(n)
alpha = 0.01
for _ in range(100) :
    x = x - alpha*(A.T@(huber_grad(A@x-b))+lam*x)

print(huber_loss(A@x-b)+0.5*lam*np.linalg.norm(x)**2)

'''
from google.colab import drive
drive.mount('/content/drive')

!wget -nc https://raw.githubusercontent.com/brpy/colab-pdf/master/colab_pdf.py
from colab_pdf import colab_pdf
colab_pdf('MATHDNN_1_5.ipynb')
'''