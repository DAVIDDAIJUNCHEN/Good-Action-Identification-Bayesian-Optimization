#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


# benchmark functions
def ackley(pnt_2d):  # 1
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = -20.0 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20
    return -1*fx

def bukin(pnt_2d):   # 2 
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = 100 * np.sqrt(np.abs(y - 0.01*(x**2) + 0.01*np.abs(x + 10)))
    return -1*fx

def booth(pnt_2d):   # 24
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = (x + 2*y - 7)**2 + (2*x + y -5)**2
    return -1*fx

def griewank(pnt_2d):  #7 [-5, 5]
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = x**2/4000 + y**2/4000 - np.cos(x)*np.cos(y/np.sqrt(2)) + 1
    return -1*fx

def schwefel(pnt_2d):  #15 [-50, 50]
    x = pnt_2d[0]*10
    y = pnt_2d[1]*10

    fx = 418.9829*2 - x*np.sin(np.sqrt(np.abs(x))) - y*np.sin(np.sqrt(np.abs(x)))
    return -1*fx    

def bohachevsky(pnt_2d): # 17  []
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = x**2 + 2*(y**2) - 0.3*np.cos(3*np.pi*x) - 0.4*np.cos(4*np.pi*y) + 0.7
    return -1*fx

def rotate_hyper(pnt_2d):  # 19 [-50, 50]
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = x**2 + x**2 + y**2 
    return -1*fx 

def matyas(pnt_2d): # 25 [-10, 10]
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = 0.26*(x**2 + y**2) - 0.48*x*y
    return -1*fx

def six_hump(pnt_2d): # 30 [-3, 3]
    x = pnt_2d[0]
    y = pnt_2d[1]

    fx = (4 - 2.1*(x**2) + (x**4)/3)*(x**2) + x*y + (-4 + 4*(y**2))*(y**2)
    return -1*fx 

def forrester(pnt_1d): # 39 [0, 1]
    x = pnt_1d[0]

    fx = np.sin(12*x -4)*((6*x - 2)**2)
    return -1*fx 

def exp_mu(input, mu, theta=1):
    """
    Exponential function on ||input - mu||^2,
    mu and x are lists with same length
    """
    assert(len(input) == len(mu))
    square_mu_input = [(xi - mu_i)**2 for xi, mu_i in zip(input, mu)]
    norm2 = sum(square_mu_input)

    exp_mu = np.exp(-0.5 * norm2 / theta**2)

    return exp_mu

def diff_mu1_mu2(mu1, mu2, theta=1):
    "The RKHS norm of exp_mu(,mu1) - exp_mu(,mu2)"
    norm_rkhs = np.sqrt(2 - 2 * exp_mu(mu1, mu2, theta))

    return norm_rkhs

def show_exp(mu2=[1, 1], x_low=-1, x_up=1, y_low=-1, y_up=1, theta=1, x_nums=100, y_nums=100):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(r"$\mu_2$"+"=("+str(mu2[0]) +","+str(mu2[1])+"), "+r"$\theta=$"+str(theta))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    X = np.arange(x_low, x_up, 0.25)
    Y = np.arange(y_low, y_up, 0.25)
    X, Y = np.meshgrid(X, Y)

    exp1_norm2 = X**2 + Y**2  
    exp2_norm2 = (X- mu2[0])**2 + (Y - mu2[1])**2

    exp1 = np.exp(-0.5 * exp1_norm2 / theta**2)
    exp2 = np.exp(-0.5 * exp2_norm2 / theta**2)
    Z1 = exp1
    Z2 = exp2

    surf1 = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.Blues,
                        linewidth=0, antialiased=False)

    ax.set_zlim(0, 1.4)
    fig.colorbar(surf2, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z1, rstride=2, cstride=2, color="red")
    ax.plot_wireframe(X, Y, Z2, rstride=2, cstride=2)
    ax.set_zlim(0, 1.4)
    
    plt.show()

    fig.savefig("./images/exp_3d")

    return 0

def plot_rkhs_norm(low_diff=0, up_diff=7, theta_lst=[0.25, 0.5, 1, 1.5, 3]):
    """
    x-axis: mu1 - mu2, 1-dim scenario
    y-axis: diff_mu1_mu2
    """
    x_draw = np.linspace(low_diff, up_diff, 100)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_title("RKHS Norm of mu1-mu2")

    for theta in theta_lst:
        y_theta = [diff_mu1_mu2([0], [ele], theta) for ele in x_draw]
        ax.plot(x_draw, y_theta, label="theta="+str(theta))

    ax.legend()
    fig.tight_layout()
    fig.savefig("./images/rkhs_norm_mu1_mu2")

    return 0

def branin(input=[3., 4.]):
    "Branin function: "
    assert(len(input) == 2)
    x1 = input[0]
    x2 = input[1] 
    branin1 = x2 - 5.1*(x1**2) / (4*np.pi**2) + 5*x1 / np.pi - 6
    branin2 = 10*(1 - 1/(8*np.pi))*np.cos(x1)
    branin = branin1**2 + branin2 + 10

    return -1*branin

def mod_branin(input=[3., 4.], shift=[5, 5]):
    "Modified Branin function: branin(x1, x2) + 20*x1 - 30*x2"
    assert(len(input) == 2)
    x1 = input[0] - shift[0]
    x2 = input[1] - shift[1]

    input = [x1, x2]
    #mod_bran = branin(input) + 20*x1 - 30*x2
    mod_bran = branin(input) + 1000*exp_mu(input, mu=[5, 5], theta=1) + 50
    
    return mod_bran

def show_branin(shift=[5,5], mu=[5, 5], theta=1, x_low=-1, x_up=1, y_low=-1, y_up=1, x_nums=100, y_nums=100):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    fig.suptitle("Branin v.s. modified Branin with mu=("+str(mu[0]) + "," + str(mu[1]) + "), theta="+str(theta),)

    ax = fig.add_subplot(1, 2, 1, projection="3d")

    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 15, 0.25)
    X, Y = np.meshgrid(X, Y)

    branin1 = Y - 5.1*(X**2) / (4*np.pi**2) + 5*X / np.pi - 6
    branin2 = 10*(1 - 1/(8*np.pi))*np.cos(X)
    branin = branin1**2 + branin2 + 30

    Z_branin = branin

    branin1_shift = (Y-shift[1]) - 5.1*((X-shift[0])**2) / (4*np.pi**2) + 5*(X-shift[0]) / np.pi - 6
    branin2_shift = 10*(1 - 1/(8*np.pi))*np.cos(X-shift[0])
    branin_shift = branin1_shift**2 + branin2_shift + 30

    Z_branin = branin    

    exp1_norm2 = (X- mu[0])**2 + (Y - mu[1])**2

    exp1 = np.exp(-0.5 * exp1_norm2 / theta**2)
    Z_modify = branin_shift + 1000*exp1 + 50

    surf_branin = ax.plot_surface(X, Y, Z_branin, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    surf_modify = ax.plot_surface(X, Y, Z_modify, rstride=1, cstride=1, cmap=cm.Blues,
                           linewidth=0, antialiased=False)
                           
    ax.set_zlim(-20, 800)
    fig.colorbar(surf_modify, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z_branin, rstride=5, cstride=5, color="red")
    ax.plot_wireframe(X, Y, Z_modify, rstride=5, cstride=5)
    ax.set_zlim(-20, 800)

    plt.show()

    fig.savefig("./images/branin_3d")

    return 0


def mono_func(input=[1.0]):
    """"Mono function"""
    x = input[0]
 
    if x <= 2:
        return 0.5*np.exp(x) + np.sin(4) + 2*np.log(2) - 2*np.log(2.5) - np.sin(5)
    elif x>2 and x<= 5:
        return np.sin(2*x) + 2*np.log(x) + 0.5*np.exp(2) - 2*np.log(2.5) - np.sin(5)
    else:
        v_5 = np.sin(2*5) + 2*np.log(5) + 0.5*np.exp(2) - 2*np.log(2.5) - np.sin(5)
        value = np.exp(-x+5) - 1 + v_5

        return value

def needle_func(input=[1.0], shift=0):
    "Needle function"
    x = input[0] - shift
    if x <= 2:
        return 0.5*np.exp(x)
    elif x>2 and x<= 2.5:
        return -100*(x - 2.25)**2 + 6.25 + 0.5*np.exp(2)
    elif x>2.5 and x <= 5:
        return np.sin(2*x) + 2*np.log(x) + 0.5*np.exp(2) - 2*np.log(2.5) - np.sin(5)
    else:
        v_5 = np.sin(2*5) + 2*np.log(5) + 0.5*np.exp(2) - 2*np.log(2.5) - np.sin(5)
        value = np.exp(-x+5) - 1 + v_5 
        
        return value

def show_needle(x_low, x_high, shift=0):
    "plot needle function above"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [0.2*needle_func([ele]) for ele in x_draw]
    #y2 = [needle_func([ele], shift) for ele in x_draw]
    
    #diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    #ax.set_title("Needle functions shift="+str(shift))

    ax.plot(x_draw, y1)#, label="task1")
    #ax.plot(x_draw, y2, label="task2")
    #ax.plot(x_draw, diff_y2_y1, label="task2 - task1")

    #ax.legend()

    fig.tight_layout()
    fig.savefig("./images/needle.pdf")

def show_mono2needle(x_low, x_high, shift=0):
    "plot mono func to needle func above"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [mono_func([ele]) for ele in x_draw]
    y2 = [needle_func([ele], shift) for ele in x_draw]
    
    diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Mono to Needle (shift="+str(shift)+")")

    ax.plot(x_draw, y1, label="task1")
    ax.plot(x_draw, y2, label="task2")
    ax.plot(x_draw, diff_y2_y1, label="task2 - task1")

    ax.legend()
    fig.tight_layout()
    fig.savefig("./images/mono2needle.png")


def two_exp_mu(input, lambda1, lambda2, mu1, mu2, theta1=1, theta2=2):
    """
    Exponential function on ||input - mu||^2,
    mu and x are lists with same length
    """
    assert(len(input) == len(mu1))
    assert(len(input) == len(mu2))

    add_exp_mu = lambda1*exp_mu(input, mu1, theta1) + lambda2*exp_mu(input, mu2, theta2)

    return add_exp_mu

def tri_exp_mu(input, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1=1, theta2=2, theta3=3):  
    "addition of triple exponential function"
    assert(len(input) == len(mu1))
    assert(len(input) == len(mu2))
    assert(len(input) == len(mu3))

    add_exp_mu = lambda1*exp_mu(input, mu1, theta1) + lambda2*exp_mu(input, mu2, theta2) + lambda3*exp_mu(input, mu3, theta3)

    return add_exp_mu

def show_mono2double_exp_mu(lambda2, mu1, mu2, theta1, theta2, x_low, x_high):
    "plot: one exp to two exp transfering, where lambda1 = 1"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [exp_mu([ele], mu1, theta1) for ele in x_draw]
    y2 = [two_exp_mu([ele], 1, lambda2, mu1, mu2, theta1, theta2) for ele in x_draw]

    diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1, label="task1")
    ax.plot(x_draw, y2, label="task2")
    #ax.plot(x_draw, diff_y2_y1, label="task2 - task1")

    ax.legend()
    fig.tight_layout()
    fig.savefig("./images/mono2double.png")       


def show_two_exp_mu(lambda1, lambda2, mu1, mu2, theta1, theta2, x_low, x_high):
    "plot exp func to two exp func"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [two_exp_mu([ele], lambda1, lambda2, mu1, mu2, theta1, theta2) for ele in x_draw]
    y2 = [two_exp_mu([ele], lambda2, lambda1, mu1, mu2, theta1, theta2) for ele in x_draw]
    
    diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1, '--', label="task1")
    ax.plot(x_draw, y2, '-', label="task2")
    ax.plot(x_draw, diff_y2_y1, '-.', label="task2 - task1")

    ax.legend(loc=4)
    fig.tight_layout()
    fig.savefig("./images/double2double.pdf")    

def show_double2triple_exp_mu(lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3, x_low, x_high):
    "plot exp func to two exp func"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [tri_exp_mu([ele], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3) for ele in x_draw]
    
    lambda1_t2 = 1; lambda2_t2 = 1.4; lambda3_t2 = 1.9
    mu1_t2 = [0]; mu2_t2 = [5]; mu3_t2 = [10]
    
    y2 = [tri_exp_mu([ele], lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3) for ele in x_draw]
    
    diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1, '--', label="task1")
    ax.plot(x_draw, y2, '-', label="task2")
    ax.plot(x_draw, diff_y2_y1, '-.', label="task2 - task1")

    ax.legend(loc=4)
    fig.tight_layout()
    fig.savefig("./images/double2triple.pdf")   


def show_triple2double_exp_mu(lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3, x_low, x_high):
    "plot exp func to two exp func"
    x_draw = np.linspace(x_low, x_high, 100)

    y1 = [tri_exp_mu([ele], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3) for ele in x_draw]
    
    lambda1_t2 = lambda2+0.2; lambda2_t2 = 0; lambda3_t2 = lambda1 - 0.2
    mu1_t2 = [mu1[0] + 0.2]; mu2_t2 = mu2; mu3_t2 = [mu3[0] - 0.2]
    
    y2 = [tri_exp_mu([ele], lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3) for ele in x_draw]
    
    diff_y2_y1 = [ele2 - ele1 for ele1, ele2 in zip(y1, y2)]

    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1, '--', label="task1")
    ax.plot(x_draw, y2, '-', label="task2")
    ax.plot(x_draw, diff_y2_y1, '-.', label="task2 - task1")

    ax.legend()
    fig.tight_layout()
    fig.savefig("./images/triple2triple.pdf")   


def show_3D_triple():
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    lambda1 = 1.7; lambda2 = 1.2; lambda3 = 0.8
    mu1 = [0.5, 0.5]; mu2 = [5.5, 5.5]; mu3 = [9.5, 9.5]
    theta1 = theta2 = theta3 = 1

    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 15, 0.25)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros([len(X), len(Y[0])])
    for i in range(len(X)):
        for j in range(len(Y[0])):
            x = X[i,j]
            y = Y[i,j]
            Z[i, j] = tri_exp_mu([x, y], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    ax.set_zlim(-20, 800)

    plt.show()

    fig.savefig("./images/triple_3d")

    return 0


if __name__ == "__main__":
    print(six_hump([1, 0]))
    # Exponential family
    plot_rkhs_norm()
    size = 4
    show_exp(mu2=[0.1, 0.1], theta=0.5, x_nums=200, y_nums=200, x_low=-size,x_up=size,y_low=-size, y_up=size)
    
    print(diff_mu1_mu2(mu1=[0, 0], mu2=[0.707, 0.707], theta=1.414))
    # Branin
    show_branin()

    # Needle to Needle 
    show_needle(0, 6, shift=0.0)

    # Mono to Needle
    show_mono2needle(0, 10, shift=-0.2)

    # Mono to double exponential 
    lambda2 = 2
    mu1 = [0]; mu2 = [9]
    theta1 = 0.5; theta2 = 2
    show_mono2double_exp_mu(lambda2, mu1, mu2, theta1, theta2, x_low = -5, x_high=15)

    # Double to double exponential
    lambda1 = 1; lambda2 = 1.5
    mu1 = [0]; mu2 = [5]
    theta1 = 1; theta2 = 1
    show_two_exp_mu(lambda1, lambda2, mu1, mu2, theta1, theta2, x_low=-5, x_high=15)

    # Double to Triple exponential 
    lambda1 = 1.7; lambda2 = 0; lambda3 = 0.8
    mu1 = [0.5]; mu2 = [5]; mu3 = [9.5]
    theta1 = 1; theta2 = 1; theta3 = 1
    show_double2triple_exp_mu(lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3, x_low=-5, x_high=15)    

    # Triple to triple exponential 
    lambda1 = 1; lambda2 = 1.5; lambda3 = 1.25
    mu1 = [0]; mu2 = [5]; mu3 = [10]
    theta1 = 1; theta2 = 1; theta3 = 1
    show_triple2double_exp_mu(lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3, x_low=-5, x_high=15)

    # 3D Triple Illustration
    show_3D_triple()
    