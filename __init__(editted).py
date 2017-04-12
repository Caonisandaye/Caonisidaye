"""
Nvision Experiments
author: Nate Strawn
email: nate.strawn@georgetown.edu
website: http://natestrawn.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import numpy.random as rand
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
from scipy import fftpack

import sklearn.manifold as mfd
from sklearn.decomposition import PCA, KernelPCA
import sklearn.preprocessing as pproc
from sklearn import datasets

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def sparse_grid_laplacian(n):
    '''
    Return sparse graph Laplacian for an n by n grid
    :param n: width/height parameter for the grid
    :return: sparse graph Laplacian matrix
    '''
    t = [2] * n
    t[0] = 1
    t[-1] = 1
    m = [-1] * (n-1)
    a = sparse.diags([t, m, m], [0, -1, 1])
    b = sparse.kron(a, sparse.eye(n)) + sparse.kron(sparse.eye(n), a)
    return b


def sparse_discrete_diff(n):
    '''
    Sparse discrete difference matrix on a grid
    :param n: for the n by n grid
    :return: sparse discrete difference matrix
    '''
    d = sparse.diags([-1, 1], [0, 1], shape = (n-1, n))
    id = sparse.identity(n)
    return sparse.vstack([sparse.kron(d,id), sparse.kron(id,d)])

def data_embedding_variation(x, f, res):
    '''
    Objective for the square variation of image embeddings
    :param x: data matrix (data in rows)
    :param f: the data embedding matrix
    :param res: the resolution of the images
    :return: evaluation of the square variation
    '''
    # x is data in rows
    n = f.shape[1]
    omega_f = (np.transpose(x).dot(x) ).dot(f)
    f_L = f .dot( sparse_grid_laplacian(res))
    return np.sum(omega_f * f_L)


def shrink(x,l):
    '''
    Soft thresholding function
    :param x: numpy array to be threshholded
    :param l: threshold parameter
    :return: numpy array the same shape as x
    '''
    y = np.abs(x)-l
    y[y<0] = 0
    return np.sign(x)*y


def matrix_TV(res, x, f, lam, gam, iter = 100):
    '''
    Perform TV minimization for numerous image embedding examples
    :param res: resolution of images
    :param x: numpy array of data examples
    :param f: embedding matrix
    :param lam: sparsity term
    :param gam: fidelity term
    :param iter: number of iterations to employ
    :return: TV minimized image embeddings
    '''
    F = f.T
    X = x.T
    D = sparse_discrete_diff(res)
    A = gam * D.T.dot(D) + F.dot(F.T)
    lam_gam = lam / gam
    phi = F.dot(X)
    q = np.zeros((D.shape[0], phi.shape[1]))
    b = np.zeros(q.shape)
    h = F.dot(X)

    for i in range(iter):
        phi = la.solve(A, h + gam * D.T .dot(q-b))
        q = shrink(b + D.dot(phi), lam_gam)
        b = b + (D.dot(phi)) - q
        err = np.sqrt(np.sum((F.T.dot(phi) - X) ** 2))
        print('Bregman iteration: %d Error %f' % (i, err))

    return phi.T


def laplacian_eigenvalues(res):
    '''
    Compute graph Laplacian eigenvalues
    :param res: resolution of the grid
    :return: numpy array (res, res) with graph Laplacian eigenvalues
    '''
    # Form the 1d graph Laplacian and eigendecomposition
    n = res ** 2
    t = [2] * res
    t[0] = 1
    t[-1] = 1
    m = [-1] * (res - 1)
    a = sparse.diags([t, m, m], [0, -1, 1])
    w, v = sparsela.eigsh(a, k=(res - 1))

    # Do silly things because eigsh won't return all eigenvalues/eigenvectors
    w = np.concatenate((np.array([0]), w))

    # Form the matrix of sums of all pairs
    exp_w = np.exp(w)
    M = np.log(np.outer(exp_w, exp_w))

    return M


def laplacian_pca_TV(res, x, f0, lam, gam, iter = 10):
    '''
    TV version of Laplacian embedding
    :param res: resolution of the grid
    :param x: numpy array of data in rows
    :param f0: initial embedding matrix
    :param lam: sparsity parameter
    :param gam: fidelity parameter
    :param iter: number of iterations to carry out
    :return: returns embedding matrix
    '''
    # f0 is an initial projection
    n = res ** 2
    num_data = x.shape[0]

    D = sparse_discrete_diff(res)
    M = 1/(lam*laplacian_eigenvalues(res).reshape(n)+gam)

    f = f0
    y = x .dot(f)
    z = shrink(y .dot(D.T), lam)

    for i in range(iter):

        # Update z
        z_old = z
        z = shrink(y .dot (D.T), lam)

        # Update f
        f_old = f
        u, s, v = la.svd(x.T .dot (y), full_matrices=False)
        f = u .dot(v)

        # Update y
        y_old = y
        q = lam * z .dot (D) + gam * x .dot(f)
        # print('norm of y before is %f' % np.sum(q ** 2))
        y = fftpack.dct(q.reshape((num_data, res, res)), norm='ortho') # Images unraveled as rows
        y = fftpack.dct(np.swapaxes(y,1,2), norm='ortho') # Swap and apply dct on the other side
        # print('norm of y after is %f' % np.sum(y ** 2))
        y = np.apply_along_axis(lambda v: M * v, 1, y.reshape((num_data, n)))
        y = fftpack.idct(y.reshape((num_data, res, res)), norm='ortho')
        y = fftpack.idct(np.swapaxes(y,1,2), norm='ortho')
        y = y.reshape((num_data, n))

        zres = np.sqrt(np.sum((z - z_old) ** 2))
        znorm = np.sqrt(np.sum((z)**2))
        yres = np.sqrt(np.sum((y - y_old) ** 2))
        ynorm = np.sqrt(np.sum((y)**2))
        fres = np.sqrt(np.sum((f - f_old) ** 2))

        value = np.sum(abs(z)) + 0.5*lam*np.sum((z-y .dot(D.T))**2) + 0.5*gam*np.sum((y- x .dot(f) )**2)
        print('Iter %d Val %f Z norm %f Z res %f Ynorm %f Y res %f F res %f' % (i, value, znorm, zres, ynorm, yres, fres))

    return f


def laplacian_pca(x, res):
    '''
    Compute image embedding matrix
    :param x: numpy array of data in rows
    :param res: resolution of images
    :return: embedding matrix
    '''
    n = res ** 2
    num_examples = x.shape[0]
    dim = x.shape[1]

    if num_examples > 10 ** 4 and dim < 10 ** 5:
        q = np.transpose(x) .dot(x)
        s_x, v_x = la.eigh(q)
        v_x = np.fliplr(v_x)
    else:
        u_x, s_x, v_x = la.svd(x)  # initial x will be v @ stuff

    # Form the 1d graph Laplacian and eigendecomposition
    t = [2] * res
    t[0] = 1
    t[-1] = 1
    m = [-1] * (res - 1)
    a = sparse.diags([t, m, m], [0, -1, 1])
    w, v = sparsela.eigsh(a, k=(res - 1))

    # Do silly things because eigsh won't return all eigenvalues/eigenvectors
    w = np.concatenate((np.array([0]), w))
    v = np.concatenate((np.ones((res, 1)) / np.sqrt(res), v), axis=1)

    # Form the matrix of sums of all pairs
    exp_w = np.exp(w)
    product_laplacian_eigen = np.log(np.outer(exp_w, exp_w))

    a, b = np.unravel_index(np.argsort(product_laplacian_eigen, axis=None)[:dim], (res, res))

    u = np.zeros((dim, n))
    for i in range(0, dim):
        u[i, :] = np.kron(v[:, a[i]], v[:, b[i]])

    # print (v_x @ u)
    return v_x .dot( u)


def weight_variable(shape):
    '''
    Tensorflow initialization of weights
    :param shape: shape of the weights to be intialized
    :return: tensorflow variable for weights
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    '''
    Tensorflow 2d convolution
    :param x: data variable to be convolved
    :param W: weight variable
    :return: convolved data
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''
    Tensorflow max pooling
    :param x: data to be pooled
    :return: max pooling over 2 by 2 blocks
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def bias_variable(shape):
    '''
    Tensorflow bias variable initialization
    :param shape: desired shape
    :return: bias variable
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def logistic_classification(data, targets, num_train, iter=1000):
    '''
    Logistic classfication test
    :param data: numpy array of data in rows
    :param targets: target labels for each row
    :param num_train: number of data points to train on
    :param iter: number of iterations
    :return: accuracy of the test
    '''
    num_data = data.shape[0]
    dim_data = data.shape[1]
    num_labels = len(np.unique(targets))

    # Replace targets with one-hot representatives
    labels = np.zeros((num_data, num_labels))
    for i in range(0, num_data):
        labels[i, int(targets[i])] = 1

    order = rand.choice(num_data, num_data, replace=False)

    x = tf.placeholder(tf.float32, shape=[None, dim_data])
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

    W = tf.Variable(tf.truncated_normal([dim_data, num_labels], stddev=0.001))
    b = tf.Variable(tf.zeros([num_labels]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y_prime = tf.matmul(x, W) + b
    y = tf.nn.softmax(y_prime)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_prime, y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for i in range(iter):
        sess.run(train_step, feed_dict={x: data[order[:num_train], :], y_: labels[order[:num_train], :]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_accuracy = sess.run(accuracy, feed_dict={x: data[order[num_train:], :], y_: labels[order[num_train:], :]})

    print("logisitic test accuracy %g" % test_accuracy)

    sess.close()

    return test_accuracy

def cnn(data, targets, res, num_train, window_size, num_units, final_units, iter=1000):
    '''
    Convolutional neural network classification test
    :param data: numpy array of data in rows
    :param targets: labels for training/testing
    :param res: resolution of images
    :param num_train: number of examples to train on from data
    :param window_size: number of windows to consider
    :param num_units: array listing number of units in different layers
    :param final_units: number of units in final fully connected layer
    :param iter: total number of training iterations
    :return:
    '''
    # Architectural parameters
    num_data = data.shape[0]
    dim_data = data.shape[1]
    patch_size = int(res / window_size)
    order = rand.choice(num_data, num_data, replace=False)

    num_labels = len(np.unique(targets))
    # Replace targets with one-hot representatives
    labels = np.zeros((num_data, num_labels))
    for i in range(0, num_data):
        labels[i, int(targets[i])] = 1

    x = tf.placeholder(tf.float32, shape=[None, dim_data])
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
    x_image = tf.reshape(x, [-1, res, res, 1])

    W = [None] * len(num_units)
    b = [None] * len(num_units)
    h_conv = [None] * len(num_units)
    h_pool = [None] * len(num_units)
    W[0] = weight_variable([window_size, window_size, 1, num_units[0]])
    b[0] = bias_variable([num_units[0]])
    h_conv[0] = tf.nn.relu(conv2d(x_image, W[0]) + b[0])
    h_pool[0] = max_pool_2x2(h_conv[0])

    for i in range(1, len(num_units)):
        W[i] = weight_variable([window_size, window_size, num_units[i-1], num_units[i]])
        b[i] = bias_variable([num_units[i]])
        h_conv[i] = tf.nn.relu(conv2d(h_pool[i-1], W[i]) + b[i])
        h_pool[i] = max_pool_2x2(h_conv[i])

    # Flatten for the final layer before classification
    W_fc1 = weight_variable([patch_size * patch_size * num_units[-1], final_units])  # 4 because 16 / 4 ...
    b_fc1 = bias_variable([final_units])

    h_pool_flat = tf.reshape(h_pool[-1], [-1, patch_size * patch_size * num_units[-1]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout for layer 3
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Weights for the readout layer
    W_fc2 = weight_variable([final_units, num_labels])
    b_fc2 = bias_variable([num_labels])

    # Final activation
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    sess_conv = tf.Session()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess_conv.run(tf.global_variables_initializer())

    batch = 0
    if num_train>1000:
        batch_size=1000
    else:
        batch_size = num_train

    for i in range(iter):
        data_range = order[[k % num_train for k in range(batch*batch_size,((batch + 1) * batch_size))]]
        train_accuracy = sess_conv.run(accuracy, feed_dict={x: data[data_range, :],
                                                            y_: labels[data_range, :],
                                                            keep_prob: 1.0})
        if ((i % 1000) == 0):
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess_conv.run(train_step,
                      feed_dict={x: data[data_range, :],
                                 y_: labels[data_range, :],
                                  keep_prob: 0.5})

        batch = (batch + 1) % int(np.floor(num_train / batch_size)) # Updated the batch indicator

    test_accuracy = sess_conv.run(accuracy, feed_dict={x: data[order[num_train:], :],
                                                        y_: labels[order[num_train:], :],
                                                        keep_prob: 1.0})

    sess_conv.close()

    print("cnn test accuracy %g" % test_accuracy)

    return test_accuracy

def simultaneous_test(data, targets, res, num_train, window_size, num_units, final_units, iter=1000):
    num_data = data.shape[0]
    dim_data = data.shape[1]
    num_labels = len(np.unique(targets))

    # Replace targets with one-hot representatives
    labels = np.zeros((num_data, num_labels))
    for i in range(0, num_data):
        labels[i, int(targets[i])] = 1

    order = rand.choice(num_data, num_data, replace=False)

    x = tf.placeholder(tf.float32, shape=[None, dim_data])
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])

    W = tf.Variable(tf.truncated_normal([dim_data, num_labels], stddev=0.001))
    b = tf.Variable(tf.zeros([num_labels]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y_prime = tf.matmul(x, W) + b
    y = tf.nn.softmax(y_prime)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_prime, y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for i in range(iter):
        sess.run(train_step, feed_dict={x: data[order[:num_train], :], y_: labels[order[:num_train], :]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    log_test_accuracy = sess.run(accuracy, feed_dict={x: data[order[num_train:], :], y_: labels[order[num_train:], :]})

    sess.close()

    patch_size = int(res / window_size)

    num_labels = len(np.unique(targets))
    # Replace targets with one-hot representatives

    x = tf.placeholder(tf.float32, shape=[None, dim_data])
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
    x_image = tf.reshape(x, [-1, res, res, 1])

    W = [None] * len(num_units)
    b = [None] * len(num_units)
    h_conv = [None] * len(num_units)
    h_pool = [None] * len(num_units)
    W[0] = weight_variable([window_size, window_size, 1, num_units[0]])
    b[0] = bias_variable([num_units[0]])
    h_conv[0] = tf.nn.relu(conv2d(x_image, W[0]) + b[0])
    h_pool[0] = max_pool_2x2(h_conv[0])

    for i in range(1, len(num_units)):
        W[i] = weight_variable([window_size, window_size, num_units[i - 1], num_units[i]])
        b[i] = bias_variable([num_units[i]])
        h_conv[i] = tf.nn.relu(conv2d(h_pool[i - 1], W[i]) + b[i])
        h_pool[i] = max_pool_2x2(h_conv[i])

    # Flatten for the final layer before classification
    W_fc1 = weight_variable([patch_size * patch_size * num_units[-1], final_units])  # 4 because 16 / 4 ...
    b_fc1 = bias_variable([final_units])

    h_pool_flat = tf.reshape(h_pool[-1], [-1, patch_size * patch_size * num_units[-1]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout for layer 3
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Weights for the readout layer
    W_fc2 = weight_variable([final_units, num_labels])
    b_fc2 = bias_variable([num_labels])

    # Final activation
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    sess_conv = tf.Session()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess_conv.run(tf.global_variables_initializer())

    batch = 0
    if num_train > 1000:
        batch_size = 1000
    else:
        batch_size = num_train

    for i in range(iter):
        data_range = order[[k % num_train for k in range(batch * batch_size, ((batch + 1) * batch_size))]]
        train_accuracy = sess_conv.run(accuracy, feed_dict={x: data[data_range, :],
                                                            y_: labels[data_range, :],
                                                            keep_prob: 1.0})
        if ((i % 100) == 0):
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess_conv.run(train_step,
                      feed_dict={x: data[data_range, :],
                                 y_: labels[data_range, :],
                                 keep_prob: 0.5})

        batch = (batch + 1) % int(np.floor(num_train / batch_size))  # Updated the batch indicator

    cnn_test_accuracy = sess_conv.run(accuracy, feed_dict={x: data[order[num_train:], :],
                                                       y_: labels[order[num_train:], :],
                                                       keep_prob: 1.0})

    sess_conv.close()

    print("logisitic test accuracy %g, cnn test accuracy %g" % (log_test_accuracy, cnn_test_accuracy))

    return log_test_accuracy, cnn_test_accuracy

def test_bcw():
    '''
    Test for Wisconsin breast cancer data
    :return:
    '''

    # Set data and image space parameters
    dim = 30 # Dimension of the data
    res = 16  # Image resolution
    n = res ** 2  # Total number of pixels

    # Load the data, store as a matrix of rows, and scale the features
    bcw = datasets.load_breast_cancer() # Breast cancer data
    db = bcw.data[:, 0:dim]
    robust_scaler = pproc.RobustScaler()
    robust_scaler.fit_transform(db)
    db = robust_scaler.transform(db)

    # Spectral embedding of the data breaks rotational symmetry of Laplacian PCA
    #print('Performing spectral embedding...')
    #se = mfd.SpectralEmbedding(n_components=10)
    #sp = se.fit_transform(db)
    # Kernel PCA
    print('Performing kernel PCA embedding...')
    ke = KernelPCA(kernel='rbf')
    x_kpca = ke.fit_transform(db)
    dim = 32
    sp = x_kpca[:,:dim]
    #sp = x_kpca
    # sp = db
    #robust_scaler.fit_transform(sp)
    #sp = robust_scaler.transform(sp) # Rescale data for robustness

    # Image space embedding
    print('Performing the image space embedding...')
    x = laplacian_pca(sp,res) # x is the projection

    # x = laplacian_pca_TV(res, sp, x, 0.1, 0.1, iter = 1000)

    # Show the first 200 images for the laplacian regularized pca embedding
    plt.figure(figsize=(25, 8))
    gs1 = gridspec.GridSpec(8, 25)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    # data_images = sp @ x
    data_images = matrix_TV(res, sp, x, 10, 0.1, iter=1000) # TV images

    a = np.percentile(data_images, 1)
    b = np.percentile(data_images, 99)
    data_images_0 = data_images[bcw.target==0, :]
    data_images_1 = data_images[bcw.target==1, :]
    for i in range(0, 100):
        ax = plt.subplot(gs1[i])
        ax.set_title(0)
        plt.axis('off')
        plt.imshow(data_images_0[i, :].reshape((res, res)), interpolation='nearest', clim=(a, b))
    for i in range(0, 100):
        ax = plt.subplot(gs1[i+100])
        ax.set_title(1)
        plt.axis('off')
        plt.imshow(data_images_1[i, :].reshape((res, res)), interpolation='nearest', clim=(a, b))
    plt.show()

    # Set the size of the training set
    num_train = 400  # of 569 examples

    # Bootstrap for generalization error
    num_samples = 25
    logistic_classify = [None] * num_samples
    cnn_classify = [None] * num_samples

    #----------------------------------------
    # Train and test logisitic regression
    #----------------------------------------


    for i in range(num_samples):
        logistic_classify[i] = logistic_classification(data_images, bcw.target, num_train, iter = 10000)

    # ----------------------------------------
    # Train and test convolutional neural network
    # ----------------------------------------

    # Architectural parameters
    window_size = 4
    num_unit1 = 16
    num_unit2 = 32
    num_unit3 = 64

    for i in range(num_samples):
        cnn_classify[i] = cnn(data_images, bcw.target, res, num_train, window_size, [num_unit1, num_unit2], num_unit3, iter=10000)

    plt.figure()
    plt.boxplot([logistic_classify, cnn_classify])
    plt.show()


def test_mnist():
    '''
    Test for MNIST data
    :return:
    '''
    # Set data and image space parameters
    num_examples = 10000
    dim = 28 ** 2  # Dimension of the data
    res = 8  # Image resolution

    #dim = 200
    #res = 16
    n = res ** 2  # Total number of pixels

    # Load the data, store as a matrix of rows, and scale the features
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    db = mnist.train.images[:num_examples,:dim]
    labels = mnist.train.labels[:num_examples]
    robust_scaler = pproc.RobustScaler()
    robust_scaler.fit_transform(db)
    db = robust_scaler.transform(db)

    # Spectral embedding of the data breaks rotational symmetry of Laplacian PCA
    #print('Performing spectral embedding...')
    #se = mfd.SpectralEmbedding(n_components=20)
    #sp = se.fit_transform(db)
    print('Performing kernel PCA embedding...')
    ke = KernelPCA(n_components=32, kernel='rbf')
    x_kpca = ke.fit_transform(db)
    #dim = 32
    sp = x_kpca
    #sp = db
    robust_scaler.fit_transform(sp)
    sp = robust_scaler.transform(sp)  # Rescale data for robustness

    # Image space embedding
    print('Performing the image space embedding...')
    x = laplacian_pca(sp, res)  # x is the projection
    #x = laplacian_pca_TV(res, sp, x, 0.1, 0.1, iter=50)

    # Show the first 200 images for the laplacian regularized pca embedding
    plt.figure(figsize=(25, 8))
    gs1 = gridspec.GridSpec(10, 30)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    print('Computing images...')
    data_images =  sp .dot( x)
    #data_images = matrix_TV(res, sp, x, 1.0, 0.01)
    print('Plotting images...')
    a = np.percentile(data_images, 1)
    b = np.percentile(data_images, 99)
    plt_count = 0
    for i in range(10):
        data_images_i = data_images[labels==i,:]
        data = db[labels==i,:]
        for j in range(15):
            ax = plt.subplot(gs1[plt_count])
            plt.imshow(data[j, :].reshape((28, 28)), interpolation='nearest', clim=(0, 1), cmap='gray')
            plt.axis('off')
            plt_count = plt_count + 1

            ax = plt.subplot(gs1[plt_count])
            plt.imshow(data_images_i[j, :].reshape((res, res)), interpolation='nearest', clim=(a, b))
            plt.axis('off')
            plt_count = plt_count + 1
    plt.show()

    # Set the size of the training set
    num_train = 9000  # of 70,000 examples

    # Bootstrap for generalization error
    num_samples = 9
    logistic_classify = [None] * num_samples
    cnn_classify = [None] * num_samples

    # ----------------------------------------
    # Train and test logisitic regression
    # ----------------------------------------

    print('Testing logistic classification...')
    for i in range(num_samples):
        logistic_classify[i] = logistic_classification(data_images, labels, num_train, iter=10000)

    # ----------------------------------------
    # Train and test convolutional neural network
    # ----------------------------------------

    # Architectural parameters
    window_size = 4
    num_unit1 = 16
    num_unit2 = 32
    num_unit3 = 64

    print('Testing convolutional neural network classification...')
    for i in range(num_samples):
        cnn_classify[i] = cnn(data_images, labels, res, num_train, window_size, [num_unit1, num_unit2], num_unit3, iter=10000)

    print('Plotting classification performance...')
    plt.figure()
    plt.boxplot([logistic_classify, cnn_classify])
    plt.show()


def gmap(X, Y, centers, sigma=1):
    res = X.shape[0]
    n = centers.shape[0]
    Z = np.zeros((res**2, n))
    for i in range(n):
        f = lambda x, y: np.exp(-np.sum((centers[i,:]-[x,y])**2)/sigma)
        g = np.vectorize(f)
        Z[:,i] = np.reshape(g(X, Y), (res**2,))
    return Z


def dot_space_embed():
    # Set data and image space parameters

    # Load the data, store as a matrix of rows, and scale the features
    bcw = datasets.load_breast_cancer()  # Breast cancer data
    db = bcw.data
    n = db.shape[0]
    dim = db.shape[1]
    for i in range(dim):
        db[np.argsort(db[:,i]),i] = list(range(n))

    db = db/n

    # Spectral embedding
    se = mfd.SpectralEmbedding(n_components=2, affinity='rbf', gamma=0.01, eigen_solver='arpack', random_state=None)
    data_embed = se.fit_transform(db.T)
    data_embed[np.argsort(data_embed[:, 0]), 0] = list(range(data_embed.shape[0]))
    data_embed[np.argsort(data_embed[:, 1]), 1] = list(range(data_embed.shape[0]))
    data_embed = data_embed/dim

    # Show the first 200 images for the laplacian regularized pca embedding
    plt.figure(figsize=(25, 8))
    gs1 = gridspec.GridSpec(6,12)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    db0 = db[bcw.target==0,:]
    db1 = db[bcw.target==1,:]

    offset = 36
    for i in range(0, offset):
        ax = plt.subplot(gs1[i])
        #ax.set_title(0)
        plt.axis('off')
        plt.scatter(data_embed[:, 0], data_embed[:, 1], c=db0[i,:], cmap='gray')
    for i in range(0, offset):
        ax = plt.subplot(gs1[i + offset])
        #ax.set_title(1)
        plt.axis('off')
        plt.scatter(data_embed[:, 0], data_embed[:, 1], c=db1[i, :], cmap='gray')
    plt.show()

    res = 32
    x = np.linspace(0,1,res)
    y = np.linspace(0,1,res)
    X, Y = np.meshgrid(x, y)
    gmapped = db .dot( gmap(X, Y, data_embed, sigma=0.01).T)
    b = np.percentile(gmapped, 98)

    gmap0 = gmapped[bcw.target==0,:]
    gmap1 = gmapped[bcw.target==1,:]

    plt.figure(figsize=(25, 8))
    gs1 = gridspec.GridSpec(6, 12)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    offset = 36
    for i in range(0, offset):
        ax = plt.subplot(gs1[i])
        plt.axis('off')
        plt.imshow(np.reshape(gmap0[i,:], (res, res)), interpolation='nearest', cmap='gray', clim=[0, b])
    for i in range(0, offset):
        ax = plt.subplot(gs1[i + offset])
        # ax.set_title(1)
        plt.axis('off')
        plt.imshow(np.reshape(gmap1[i,:], (res, res)), interpolation='nearest', cmap='gray', clim=[0, b])
    plt.show()


    # Set the size of the training set
    num_train = 400  # of 569 examples

    # Bootstrap for generalization error
    num_samples = 9
    logistic_classify = [None] * num_samples
    cnn_classify = [None] * num_samples

    # ----------------------------------------
    # Train and test logisitic regression and CNN
    # ----------------------------------------

    # Architectural parameters
    window_size = 4
    num_unit1 = 16
    num_unit2 = 32
    num_unit3 = 64

    for i in range(num_samples):
        a, b = simultaneous_test(gmapped, bcw.target, res, num_train, window_size, [num_unit1, num_unit2], num_unit3,
                              iter=20000)
        # 20000 iterations beat logisitic regression with 4, [16, 32], 64 as CNN parameters
        logistic_classify[i] = a
        cnn_classify[i] = b

    I = np.argsort(cnn_classify)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.boxplot([logistic_classify, cnn_classify])

    plt.subplot(1, 2, 2)
    plt.scatter(cnn_classify[I], logistic_classify)

    plt.show()


if __name__ == "__main__":
    #test_bcw();
    #test_mnist()
    dot_space_embed()