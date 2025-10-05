import numpy as np
def func(x, a, b):
    return a*np.sin(x[0])+np.exp(-b*x[1])
xdata=[]
xdata.append(np.linspace(0, 10, 500))
xdata.append(np.linspace(10, 0, 500))
xdata=np.array(xdata)
y = func(xdata, .05, 1.3)
print(y)
rng = np.random.default_rng()
print(xdata.shape)
y_noise = 0.2 * rng.normal(size=xdata.shape[1])
ydata = y + y_noise
with open("data.csv","a") as f:
    f.write("x,y,z")
np.savetxt("data.csv", np.c_[(xdata[0],xdata[1],ydata)],delimiter=",")
