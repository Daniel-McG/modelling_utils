import numpy as np
def user_fn(x,a,b):
    return a*np.sin(x[0])+np.exp(b*x[1])