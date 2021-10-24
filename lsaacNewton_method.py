from random import random
import numpy as np

# 函数表达式fun
function = lambda x: 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

# 梯度向量 gradient
gradient = lambda x: np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])])

# 海森矩阵 hessian_matrix:二阶偏导数，[[fxx'',fxy''],[fyx'',fyy'']]
hessian_matrix = lambda x: np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])


# 用牛顿法求解无约束优化问题
def newton_method(function, gradient, hessian_matrix, epsilon, max_iterations):
    """给一个函数y = f(theta),用牛顿法使y最小化
        Args:
            function: 要优化的函数.
            gradient: y的第一个偏导数
            hessian_matrix:海森矩阵（二阶偏导数矩阵）
            epsilon (float, optional): 默认1e-5，希望结果更加精确，可以把epsilon调小一些
            max_iterations (int, optional): 最大迭代次数，默认是10000
        Returns:
            tuple: Theta, function_theta,iterations
        """
    theta = [random() for _ in range(2)]  # 随机初始化
    iterations = 0

    while iterations < max_iterations:
        gradient_k = gradient(theta)
        hessian_k = hessian_matrix(theta)

        # 以矩阵形式解一个线性矩阵方程，或线性标量方程组
        dk = -1.0 * np.linalg.solve(hessian_k, gradient_k)
        iterations += 1
        print('第{}次迭代，此时对应的函数值是{}'.format(iterations, function(theta)))

        if np.linalg.norm(dk) < epsilon:
            break
        theta += dk

    return theta, function(theta), iterations


if __name__ == "__main__":
    theat, f_theta, iteration = newton_method(function,
                                              gradient,
                                              hessian_matrix,
                                              epsilon=1e-5,
                                              max_iterations=500)

    print('--------开始解题--------')
    print('最终求出的各个theta为{}'.format(theat))
    print('最终的函数值为是{}'.format(f_theta))
    print('迭代次数为:{}'.format(iteration))
