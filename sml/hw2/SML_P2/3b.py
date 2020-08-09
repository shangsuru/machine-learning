import tools
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dataSets/nonParamTrain.txt")

h = 0.03

xs = np.linspace(-4, 8, 100)


fig, ax = plt.subplots()   
ax.set_title('gaussian kernel density estimation')

ys = []
for x in xs:
    ys.append(tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))

ll = tools.log_likelihood(data, lambda x: tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))
print('Log-Likelihood with h = ' + str(h) + ': ' + str(ll))

ax.fill_between(xs, ys, label='Density estimate with h = ' + str(h), alpha=0.35)

h = 0.2

ys = []
for x in xs:
    ys.append(tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))
ll = tools.log_likelihood(data, lambda x: tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))
print('Log-Likelihood with h = ' + str(h) + ': ' + str(ll))

ax.fill_between(xs, ys, label='Density estimate with h = ' + str(h), alpha=0.35)


h = 0.8

ys = []
for x in xs:
    ys.append(tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))
ll = tools.log_likelihood(data, lambda x: tools.kernel_density_at_x(data, x, tools.gaussian_kernel, h))
print('Log-Likelihood with h = ' + str(h) + ': ' + str(ll))

ax.fill_between(xs, ys, label='Density estimate with h = ' + str(h), alpha=0.35)
plt.legend()
plt.show()  