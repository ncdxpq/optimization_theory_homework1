from random import random


def function(x, y):  # 二元函数f(x,y)。显然x=1,y=1时，函数可以取最小值
    return 100 * ((x ** 2 - y) ** 2) + (x - 1) ** 2


def dx(x, y):  # 对x求偏导
    return 400*x*(x**2-y) + 2*(x-1)


def dy(x, y):  # 对y求偏导
    return -200*(x**2-y)


def gradient_decent(function, function_gradient, n_variables,
                    lr, max_iterations, epsilon):
    """给一个函数y = f(theta),用 梯度下降法使y最小化
    Args:
        function: 要优化的函数.
        function_gradient: y对于各个变量的偏导数
        n_variables (int): 变量的个数
        lr (float, optional): 学习率，默认:0.1
        max_iterations (int, optional): 最大迭代次数，默认是10000
        epsilon (float, optional): 默认1e-5，希望结果更加精确，可以把tolerance调小一些。
    Returns:
        tuple: Theta, y.
    """

    # 1.随机生成对应变量个数的theta，eg:[0.5109615984946895, 0.039538100076638716]
    theta = [random() for _ in range(n_variables)]  # random.random():生成一个随机的浮点数，在0~1之间
    # 第一步theta代入各个变量中求得的第一步的值
    f_theta = function(*theta)
    # 开始迭代，最高迭代max_iter次
    for times in range(max_iterations):
        print("第{}次迭代,此时对应的函数值是{}".format(times, f_theta))

        # 2.计算当前每个theta的梯度(偏导数)，eg: [0.009278799682036265, -0.005734613578301406]
        gradient = [f(*theta) for f in function_gradient]

        # 3.通过梯度更新每个theta
        for element in range(n_variables):
            theta[element] -= gradient[element] * lr

        # 4.如果函数收敛或者超过最大迭代次数则返回
        f_theta, y_pre = function(*theta), f_theta
        if abs(y_pre - f_theta) < epsilon:
            break
    return theta, f_theta


# 求解函数最小值
def main():
    print("--------开始解题--------:")
    theta, f_theta = gradient_decent(function,
                                     [dx, dy],
                                     n_variables=2,
                                     lr=0.001,
                                     max_iterations=5000000,
                                     epsilon=1e-5)
    print("最终的结果是:\n各个变量值:{}, 对应的最终函数的值:{}\n".format(theta, f_theta))


if __name__ == '__main__':
    main()
