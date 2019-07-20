import numpy
import matplotlib.pyplot as plt

x = numpy.arange(0,6,0.1)
y = numpy.sin(x)
y2 = numpy.cos(x)
print(x[x>1])

plt.plot(x,y)
plt.plot(x,y2,linestyle = "- -")
plt.show()