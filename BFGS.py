import numpy as np


# function
def function(x):
    return 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2


# gradient_function
def gradient_function(x):
    """获得对x和 y的偏导数
    """
    result = np.zeros((2, 1))
    # 对x求偏导数
    result[0, 0] = 400 * x[0, 0] * (x[0, 0] ** 2 - x[1, 0]) + 2 * (x[0, 0] - 1)
    # 对y求偏导数
    result[1, 0] = -200 * (x[0, 0] ** 2 - x[1, 0])
    return result


def BFGS(function, gradient_function, theta, epsilon, max_iterations):
    rho = 0.55
    sigma = 0.4
    m = np.shape(theta)[0]  # 取出第一个变量
    Bk = np.eye(m)  # m阶单位矩阵
    times = 0

    while (times < max_iterations):  # 规定最大迭代次数
        gk = np.mat(gradient_function(theta))  # 计算梯度(求得此时对于两个变量的偏导数)
        dk = np.mat(-np.linalg.solve(Bk, gk))
        mk = 0

        for i in range(20):
            new_function = function(theta + rho ** i * dk)
            old_function = function(theta)
            if (new_function < old_function + sigma * (rho ** i) * (gk.T * dk)[0, 0]):
                mk = i
                break


        # BFGS校正
        y_pre = function(theta)
        theta_temp = theta + rho ** mk * dk
        sk = theta_temp - theta
        yk = gradient_function(theta_temp) - gk
        if (yk.T * sk > 0):
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)
        times = times + 1
        print('第{}次迭代\n'.format(times))
        theta = theta_temp
        print('当前各个变量的值为:{}'.format(theta_temp))
        f_theta = function(theta)
        print('当前函数值为：{}'.format(f_theta))
        if abs(y_pre - f_theta) < epsilon:
            break
    return f_theta, theta


if __name__ == "__main__":

    theta = np.random.rand(2, 1)
    f_theta, theta = BFGS(function,
                          gradient_function,
                          theta,
                          epsilon=1e-5,
                          max_iterations=500)

    print('最终的各个变量值为:{}'.format(theta))
    print('最终的函数值为：{}'.format(f_theta))
