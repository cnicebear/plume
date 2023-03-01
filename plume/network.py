import random
import numpy as np

from src import mnist_loader


class Network(object):
    def __init__(self, sizes):
        # 网络层数
        self.num_layers = len(sizes)
        [784, 30, 10]
        # 网格每层神经元个数
        self.sizes = sizes
        # 初始化每层的偏置
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        # 初始化每层的权重
        self.weights = [np.random.rand(y, x) / x for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    # 梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # 训练数据总个数
        if test_data:
            n_test = len(test_data)

        # zip 转化为list (it is not used later, otherwise zip(*(it)) and later zip(it))
        # training_data = list(training_data)
        n = len(training_data)
        # 开始训练 循环每一个epochs
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # 反向传播
    def backprop(self, x, y):
        # 保存每层偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        # 保存每一层的激励值 a = sigmoid(z)
        activations = [x]# list to store all the activations, layer by layer

        # 保存每一层的z=wx+b
        # list to store all the z vectors, layer by layer
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            z = np.dot(w, activation) + b
            # 保存每层的z
            zs.append(z)
            # 计算每层的a
            activation = self.sigmoid(z)
            # 保存每一层的a
            activations.append(activation)
        # 反向更新
        # 计算最后一层的误差
        delta = self.cost_derivative(activation[-1], y) * self.sigmoid_prime(zs[-1])
        # 计算最后一层权重和偏置的倒数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # 倒数第二层一直到第一层 权重和偏置的倒数
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            # 当前层的误差
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # 当前层的偏置 和 权重的倒数
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    # 更新mini_batch
    def update_mini_batch(self, mini_batch, eta):
        # 保存每层偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 训练一个mini_batch
        for x, y in mini_batch:
            # 反向传播得到的偏导 w ,b
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 保存一次训练网络中每层的偏导
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            # self.backprop(x, y)

        # 更新权重和偏置 W_n+1 = W_n - eta * nw
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
                network outputs the correct result. Note that the neural
                network's output is assumed to be the index of whichever
                neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)


if __name__ == '__main__':
    traning_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    traning_data, validation_data, test_data = list(traning_data),list(validation_data),list(test_data)
    net = Network([784, 30, 10])
    net.SGD(traning_data, 100, 15, 0.5, test_data=test_data)
