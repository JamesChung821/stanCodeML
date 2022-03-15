"""
File: visualize_sigmoid.py
Name: Jerry Liao
-----------------------------
This file visualizes the sigmoid function
(also known as logistic function) 
with respect to different thetas. The higher
the theta, the steeper the sigmoid is.
"""


import matplotlib
import numpy as np
# matplotlib.use("TKAgg")        # If matplotlib crashes your Mac, un-comment this line
import matplotlib.pyplot as plt


def main():
	"""
	This function shows how to use matplotlib to 
	visualiza stuff of your interests.
	"""
	x = np.linspace(-4, 4, 200)
	plt.figure(1)
	plt.subplot2grid((3, 2), (0, 0))
	k1 = x**2+10*x
	plt.scatter(x, 1/(1 + np.exp(-k1)))
	plt.title('k = x**2+10*x')

	plt.subplot2grid((3, 2), (0, 1))
	k2 = x**3+x**2-x
	plt.scatter(x, 1/(1 + np.exp(-k2)))
	plt.title('k = x**3+x**2-x')

	plt.subplot2grid((3, 2), (2, 0))
	k3 = 5*x
	plt.scatter(x, 1/(1 + np.exp(-k3)))
	plt.title('k = 5x')

	plt.subplot2grid((3, 2), (2, 1))
	k4 = 5*x+10
	plt.scatter(x, 1/(1 + np.exp(-k4)))
	plt.title('k = 5x+10')
	plt.show()


if __name__ == '__main__':
	main()
