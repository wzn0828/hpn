from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

import argparse
import tensorflow as tf

import os.path
import time
import sys
import numpy as np
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import math as m
pi = m.pi


def hypersphere_surface_area(n):
# Calculate the surface area of unit radius n sphere
    log_numerator = 0.5*n*tf.log(pi)+tf.log(2.0)
    log_denominator = tf.lgamma(0.5*n)
    log_surface_area = log_numerator-log_denominator
    return tf.exp(log_surface_area)

def main(dim, num_vecs):
    # This algorithm is based on that described in
    # "Uniform distribution of points on a hyper-sphere with applicationsto vector bit-plane encoding"
    matrix = []
    def recursive_algo(n, delta, w = np.array([]), level = 1):
        nonlocal matrix
        if w == np.array([]):
            sin_product = 1
        else:
            sin_product = np.prod(np.sin(w))
        del_w = delta/sin_product
        if n > level+1:
            w_i = del_w/2
            while w_i < pi:
                recursive_algo(n, delta, np.append(w, w_i), level+1)
                w_i += del_w
        elif n == level+1:
            w_i = del_w/2
            while w_i < 2*pi:
                recursive_algo(n, delta, np.append(w, w_i), level+1)
                w_i += del_w
        else:
            x = []
            for i in range(n-1):
                if i == 0:
                    x.append(np.cos(w[i]))
                else:
                    x.append(np.prod(np.sin(w[:i])) * np.cos(w[i]))
            x.append(np.prod(np.sin(w)))
            matrix.append(x)
    check = False
    delta = tf.Session().run(
        tf.pow(hypersphere_surface_area(dim) / num_vecs, 1 / (tf.constant(dim - 1, dtype=tf.float32))))
    while check == False:
        # matrix = np.array([])
        recursive_algo(dim, delta)
        if len(matrix) == num_vecs:
            check = True
        else:
            print(len(matrix))
            delta = delta * np.power(len(matrix)/num_vecs, 1/(dim-1))
            matrix = []
    return np.array(matrix)

def display_embedding(matrix):
    A = np.matmul(matrix, matrix.T)
    A = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)
    print("The minimum cosine distance is ",np.min(A))
    print("The maximum cosine distance is ",np.max(A))
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    # Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    ax.scatter(matrix[:,0], matrix[:,1], matrix[:,2], color="k", s=20)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct hyper points on hyperball'
        +'Uniform distribution of points on a hyper-sphere with applicationsto vector bit-plane encoding'
        +'. Example: python construct_matrix.py --dim 3 --num_vecs 30\n\n')
    parser.add_argument('--dim', type = int, help='Set the dimensions of the embedding space.', required = True)
    parser.add_argument('--num_vecs', type = int, help='Set the number of vectors to embed in the space.', required = True)
    args = parser.parse_args()

    matrix = main(args.dim, args.num_vecs)
    savepath = 'prototypes/GM/%dd-%dc.npy' % (args.dim, args.num_vecs)
    np.save(savepath, matrix)
    # display_embedding(matrix)