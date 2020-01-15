import numpy as np
import matplotlib.pyplot as plt

def solve_euler_explicit(f, x0, dt, t0, tf):
    temps = np.arange(t0, tf, dt)
    x = [x0]
    for t in temps[:-1] :
        x.append(x[-1] + f(t,x[-1])*dt)
    return temps, np.array(x)

def f(t, x):
    return -x

temps, x = solve_euler_explicit(f, 1, 0.01, 0, 5)

abs2 = np.vectorize(abs)

liste = []
listedt = np.arange(0.00001, 0.0001, 0.00001)
for dt in listedt:
    temps, x = solve_euler_explicit(f, 1, dt, 0, 5)
    x1 = abs2(x - np.exp(-temps))
    maxi = max(x1)
    liste.append(maxi)
liste1 = np.array(liste)

#plt.yscale("log")
#plt.xscale("log")
plt.plot(listedt, liste1, label = "diff")
plt.plot(listedt, listedt, label = "id")
plt.legend()
plt.show()

