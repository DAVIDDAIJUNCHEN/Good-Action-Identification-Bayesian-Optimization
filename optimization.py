#!/usr/bin/env python3

import os, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from operator import itemgetter
from gp import ZeroGProcess
from utils import check_inBounds, find_max_min_of_each_component


class UpperConfidenceBound(ZeroGProcess):
    """
    class UpperConfidenceBound: 1. construct UCB auxillary function
                                2. find maximum point of UCB acquisition function
                                3. plot acquisition function & confidence bands
    """
    def __init__(self):
        super(UpperConfidenceBound, self).__init__()  # can use data from parent class

    def aux_func_ucb(self, current_point, current_gamma=0.9):
        """
        Acquisition function: a(x|D_t) = mean(x) + current_gamma * s2(x)
        """
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB        

        mean_current = self.compute_mean(current_point)
        aux_ucb_current = mean_current + current_gamma*s2_current

        return aux_ucb_current[0, 0]

    def auto_grad_ucb(self, current_point):
        "compute gradient at current_point"

        pass 

    def find_NextBest_point_ucb(self):
        "find maximum point based on Acquisition function"
        pass

    def plot(self, num_points=100, exp_ratio=1, confidence=0.9, gammas=[0.9]):
        "plot the acquisition function as well as ZeroGP in a figure with two figs"
        min_point = min(self.X)[0]
        max_point = max(self.X)[0]
        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        # subplot 1: GProcess mean & confidence band
        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 2: UCB AC function with multiple parameters
        ac_values_lst = []
        if isinstance(gammas, list):
            for gamma in gammas:
                ac_gamma = [self.aux_func_ucb([ele], gamma) for ele in x_draw]
                ac_values_lst.append(ac_gamma)
        elif isinstance(gammas, float):
            ac_gamma = [self.aux_func_ucb([ele], gammas) for ele in x_draw]
            ac_values_lst.append(ac_gamma)
            gammas = [gammas]

        fig, (ax_gp, ax_ac) = plt.subplots(2, 1, sharex=True)

        ax_gp.set_title("GProcess Confidence Band")
        ax_gp.plot(x_draw, y_mean)
        ax_gp.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp.plot(self.X, self.Y, 'o', color="tab:red")

        ax_ac.set_title("UCB Acquisition Function")
        for ac_value, gamma in zip(ac_values_lst, gammas):
            ax_ac.plot(x_draw, ac_value, label="gamma: "+str(gamma))

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./images/example_ac_ucb.pdf")

        return 0


class ExpectedImprovement(ZeroGProcess):
    """
    class ExpectedImprovement: 1. construct EI Acquisition function 
                               2. find maximum point of EI Acquisition function
    """
    def __init__(self):  
        super(ExpectedImprovement, self).__init__()    # can use data from parent class 

    def aux_func_ei(self, current_point, kessi=0.0, zeroCheck=1e-13):
        """
        Acquisition function: a(x|D_t) = (mean(x) - y_max + kessi)*F(Z) + sqrt(sigma^2*s2(x))*f(Z)
        Z = (mean(x) - y_max + kessi) / sqrt(sigma^2*s2(x))
        """

        var_current = self.compute_var(current_point)

        if var_current < zeroCheck:
            aux_ei_current = 0.0
        else:
            y_max = max(self.Y)
            mean_current = self.compute_mean(current_point)

            Z_denom = mean_current - y_max + kessi
            Z = Z_denom / np.sqrt(var_current)

            aux_ei_current = Z_denom*norm.cdf(Z) + np.sqrt(var_current)*norm.pdf(Z)

        return aux_ei_current

    def auto_grad_ei(self, current_point, num_mc=1000, kessi=0.0, zeroCheck=1e-13):
        "compute gradient at current_point by Monte Carlo method after reparameterization"
        y_max = max(self.Y)

        if self.sigma2 == None:
            self.sigma2 = self.compute_mle_sigma2()[0,0]

        mean_current = self.compute_mean(current_point)
        var_current = self.compute_var(current_point, zeroCheck)

        grads_current_util = []
        
        for i in range(num_mc):
            z = np.random.normal(0, 1, 1)
            current_util = mean_current + np.sqrt(var_current)*z + kessi - y_max

            if (current_util < zeroCheck) or (np.sqrt(var_current) < zeroCheck): # pointwise utility function \hat{l}(x)<0 ===> grad = 0
                grad = np.zeros(shape=(self.dim, 1))
            else:
                grad_mean = self.compute_grad_mean(current_point)
                grad_var = 0.5 / np.sqrt(var_current) * self.compute_grad_var(current_point)
                grad = grad_mean + grad_var * z
            
            grads_current_util.append(grad)

        grad_current_util = np.mean(grads_current_util, axis=0) # take element-wise mean 

        return grad_current_util

    def find_best_from_point_ei(self, init_point, num_step=1000, kessi=0.0, l_bounds=None, u_bounds=None,
                                num_mc=1000, thres=1e-3, learn_rate=0.1, beta_1=0.9, beta_2=0.999, epslon=1e-8):
        """
        Find maximum point of Acquisition function in bounds from init_point by using ADAM algorithm.
        l_bounds: [l_1, l_2, ..., l_d]
        u_bounds: [u_1, u_2, ..., u_d]
        """
        dim = self.dim
        assert(len(init_point) == dim)
        
        # optimize in bounds 
        if l_bounds != None:
            assert(len(l_bounds) == dim)
        
        if u_bounds != None:
            assert(len(u_bounds) == dim)
        
        # initialize momentum & rmsp vector & gradient at init point
        m_t = np.zeros(shape=(dim, 1))
        gamma_t = np.zeros(shape=(dim, 1))
        point_current_vec = np.reshape(np.array(init_point), newshape=(dim, 1))
        point_current = init_point
        aux_current = self.aux_func_ei(point_current, kessi)

        for t in range(num_step):
            grad_current = self.auto_grad_ei(point_current, num_mc, kessi)
            m_t = beta_1*m_t + (1 - beta_1)*grad_current
            gamma_t = beta_2*gamma_t + (1 - beta_2)*grad_current**2

            # BC (bias correct) m_t and gamma_t
            m_t_BC = m_t / (1 - beta_1)
            gamma_t_BC = gamma_t / (1 - beta_2)  

            point_current_vec = point_current_vec + m_t_BC * learn_rate / np.sqrt(gamma_t_BC + epslon)
            point_current_tmp = np.reshape(point_current_vec, newshape=(1, dim)).tolist()
            point_current_tmp = point_current_tmp[0]
            
            # stop when find point out of boundary
            if check_inBounds(point_current_tmp, l_bounds, u_bounds):
                point_current = point_current_tmp
                aux_current = self.aux_func_ei(point_current, kessi)
            else:
                break

        return point_current, aux_current


    def find_best_NextPoint_ei(self, init_points=None, num_step=1000, kessi=0.0, l_bounds=None, u_bounds=None,
                               num_mc=1000, thres=1e-3, learn_rate=0.1, beta_1=0.9, beta_2=0.999, epslon=1e-8):
        """ find best next point of Acquisition function by starting from multi-points
            init_points: init_points = experiment points if None,
        """
        if init_points == None:
            init_points = self.X

        best_points_aux = []

        for point_k in init_points:
            best_point_k, best_aux_k = self.find_best_from_point_ei(point_k, num_step, kessi, l_bounds, u_bounds, num_mc,
                                                                    thres, learn_rate, beta_1, beta_2, epslon)
            best_points_aux.append((best_point_k, best_aux_k))

        best_point, best_aux = max(best_points_aux, key=itemgetter(1))

        return best_point, best_aux

    def plot_ei(self, num_points=100, exp_ratio=1, confidence=0.9, kessis=[0.0], highlight_point=None):
        "plot the acquisition function as well as ZeroGP in a figure with two figs"
        min_point = min(self.X)[0]
        max_point = max(self.X)[0]
        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)
        print("theta Gp: ", self.theta)
        # subplot 1: GProcess mean & confidence band
        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 2: EI AC function with multiple parameters
        ac_values_lst = []
        if isinstance(kessis, list):
            for kessi in kessis:
                ac_kessi = [self.aux_func_ei([ele], kessi) for ele in x_draw]
                ac_values_lst.append(ac_kessi)
        elif isinstance(kessis, float):
            ac_kessi = [self.aux_func_ei([ele], kessis) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            kessis = [kessis]

        fig, (ax_gp, ax_ac) = plt.subplots(2, 1, sharex=True)

        print("x: ", self.X[0:3])
        print("y: ", self.Y[0:3])
        print("y_pred: ", [self.compute_mean(ele) for ele in self.X[0:3]])

        ax_gp.set_title("GProcess Confidence Band")
        ax_gp.plot(x_draw, y_mean)
        ax_gp.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp.plot(self.X, self.Y, 'o', color="tab:red")

        ax_ac.set_title("EI Acquisition Function")
        for ac_value, kessi in zip(ac_values_lst, kessis):
            ax_ac.plot(x_draw, ac_value, label="kessi: "+str(kessi))
        
        if highlight_point != None:
            ax_ac.plot(highlight_point[0], highlight_point[1], 'o', color="tab:red")

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./images/example_ac_ei.pdf")

        return 0


class ShapeTransferBO(ExpectedImprovement, UpperConfidenceBound):
    """
    class ShapeTransferBO:
    
    """
    def __init__(self):
        super(ShapeTransferBO, self).__init__()
        self.zeroGP1 = None
        self.diffGP = None

    def build_task1_gp(self, file_exp_task1, theta_task1=1.0, prior_mean=None, r_out_bound=0.5):
        """
        build ZeroGProcess for task1 with known experiment points
        theta_task1: float, kernel coefficient used in task 1
        prior_mean:  constant mean used in building GP 1 
        r_out_bound: ratio of mean used in out of boundary 
        """
        zeroGP1 = ZeroGProcess(prior_mean=prior_mean, r_out_bound=r_out_bound)
        zeroGP1.get_data_from_file(file_exp_task1)
        zeroGP1.theta = theta_task1
        assert(len(zeroGP1.X) == len(zeroGP1.Y))

        self.zeroGP1 = zeroGP1

        return 0

    def build_diff_gp(self):
        "build ZeroGProcess on difference between task2 and task1_gp"

        diffGP = ZeroGProcess()
        assert(len(self.X) == len(self.Y))

        # compute difference between task2 and task1_gp
        X_task2 = self.X
        Y_task2 = self.Y

        diff_Y = []

        for point, y_task2 in zip(X_task2, Y_task2):
            mean_GP1_point = self.zeroGP1.compute_mean(point)
            diff_y_point = y_task2 - mean_GP1_point
            diff_Y.append(diff_y_point)
        
        diffGP.Y = diff_Y
        diffGP.X = self.X
        self.diffGP = diffGP

        return 0

    def compute_mean(self, current_point):
        "compute the mean value of GP1 + diffGP at current_point"
        mean_GP1 = self.zeroGP1.compute_mean(current_point)
        mean_diffGP = self.diffGP.compute_mean(current_point)

        mean_current = mean_GP1 + mean_diffGP

        return mean_current

    def compute_grad_mean(self, current_point):
        "compute the gradient of mean(x) in GP1 + diffGP at current_point"
        grad_mean_a = self.zeroGP1.compute_grad_mean(current_point)
        grad_mean_b = self.diffGP.compute_grad_mean(current_point)
        assert(len(grad_mean_a) == len(grad_mean_b))

        grad_mean = grad_mean_a + grad_mean_b

        return grad_mean

    def compute_mle_sigma2(self):
        """
        compute the MLE of sigma^2 in GP1 + diffGP
        Note: GP1 is simply treated as constant function without randomness
        """
        sigma2_hat = self.diffGP.compute_mle_sigma2()

        return sigma2_hat

    def compute_var(self, current_point, zeroCheck=1e-13):
        """
        compute the variance value of GP1 + diffGP at current_point
        Note: GP1 is simply treated as constant function without randomness 
        """
        var_diffGP = self.diffGP.compute_var(current_point, zeroCheck)
        var_current = var_diffGP

        if var_current < zeroCheck:
            return 0.0

        return var_current

    def compute_grad_var(self, current_point, zeroCheck=1e-13):
        "compute the gradient of var(x) in GP1 + diffGP at current_point"
        grad_var = self.diffGP.compute_grad_var(current_point, zeroCheck)

        return grad_var

    def plot_ac(self, exp_ratio=0.5, kessis=[0.0]):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        #fig.suptilte("AC function optimization checking",)

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        
        # find min&max values 
        min_exp2 = find_max_min_of_each_component(self.X, max=False)
        min_point_exp2_x1 = min_exp2[0] - abs(min_exp2[0])*exp_ratio
        min_point_exp2_x2 = min_exp2[1] - abs(min_exp2[0])*exp_ratio

        max_exp2 = find_max_min_of_each_component(self.X, max=True)
        max_point_exp2_x1 = max_exp2[0] + abs(max_exp2[0])*exp_ratio
        max_point_exp2_x2 = max_exp2[1] + abs(max_exp2[0])*exp_ratio

        X = np.arange(min_point_exp2_x1, max_point_exp2_x1, 0.25)
        Y = np.arange(min_point_exp2_x2, max_point_exp2_x2, 0.25)
    
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros([len(X), len(Y[0])])
        for i in range(len(X)):
            for j in range(len(Y[0])):
                x = X[i,j]
                y = Y[i,j]
                Z[i, j] = self.aux_func_ei([x, y], kessis)

        surf_ac = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        fig.colorbar(surf_ac, shrink=0.5, aspect=10)

        plt.show()

        return 0

    def plot_ei(self, num_points=100, exp_ratio=1, confidence=0.9, kessis=[0.0], highlight_point=None):
        "plot the acquisition function as well as ZeroGP&STBO in a figure with two figs"

        min_point_exp1 = min(self.zeroGP1.X)[0]
        min_point_exp2 = min(self.X)[0]
        min_point = min(min_point_exp1, min_point_exp2)

        max_point_exp1 = max(self.zeroGP1.X)[0]
        max_point_exp2 = max(self.X)[0]
        max_point = max(max_point_exp1, max_point_exp2)

        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        # subplot 1: GProcess means & confidence bands of zeroGP1 (on data 1) and diffGP (on task2 - mean1)
        GP1 = self.zeroGP1
        y1_mean = [GP1.compute_mean([ele]) for ele in x_draw]

        y1_conf_int = [GP1.conf_interval([ele]) for ele in x_draw]
        y1_lower = [ele[0] for ele in y1_conf_int]
        y1_upper = [ele[1] for ele in y1_conf_int]

        diffGP = self.diffGP
        yDiff_mean = [diffGP.compute_mean([ele]) for ele in x_draw]
        yDiff_conf_int = [diffGP.conf_interval([ele]) for ele in x_draw]
        yDiff_lower = [ele[0] for ele in yDiff_conf_int]
        yDiff_upper = [ele[1] for ele in yDiff_conf_int]

        # subplot 2: GProcess means & confidence bands of GP2 (normal zero GP on data 2) and STBO
        GP2_only = ZeroGProcess()
        GP2_only.X = self.X
        GP2_only.Y = self.Y
        y2Only_mean = [GP2_only.compute_mean([ele]) for ele in x_draw]
        y2Only_conf_int = [GP2_only.conf_interval([ele], confidence) for ele in x_draw]
        y2Only_lower = [ele[0] for ele in y2Only_conf_int]
        y2Only_upper = [ele[1] for ele in y2Only_conf_int]

        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 3: EI AC function with multiple parameters 
        ac_values_lst = []
        if isinstance(kessis, list):
            for kessi in kessis:
                ac_kessi = [self.aux_func_ei([ele], kessi) for ele in x_draw]
                ac_values_lst.append(ac_kessi)
        elif isinstance(kessis, float):
            ac_kessi = [self.aux_func_ei([ele], kessis) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            kessis = [kessis]

        fig, (ax_gp1_diff, ax_gp_stbo, ax_ac) = plt.subplots(3, 1, sharex=True)

        # subplot 1: GP1 & diffGP
        ax_gp1_diff.set_title("GProcess Confidence Band")
        ax_gp1_diff.plot(x_draw, y1_mean, label="GP1")
        ax_gp1_diff.fill_between(x_draw, y1_lower, y1_upper, alpha=0.2)
        if GP1.prior_mean == None:
            ax_gp1_diff.plot(GP1.X, GP1.Y, 'o', color="tab:blue")
        else:
            Y = [y + GP1.prior_mean for y in GP1.Y]
            ax_gp1_diff.plot(GP1.X, Y, 'o', color="tab:blue")

        ax_gp1_diff.plot(x_draw, yDiff_mean, label="diffGP")
        ax_gp1_diff.fill_between(x_draw, yDiff_lower, yDiff_upper, alpha=0.2)
        ax_gp1_diff.plot(diffGP.X, diffGP.Y, 'o', color="tab:red")
        ax_gp1_diff.legend()

        # subplot 2: GP2 only 
        ax_gp_stbo.set_title("GProcess Confidence Band")
        ax_gp_stbo.plot(x_draw, y2Only_mean, label="GP2 Only")
        ax_gp_stbo.fill_between(x_draw, y2Only_lower, y2Only_upper, alpha=0.2)
        ax_gp_stbo.plot(GP2_only.X, GP2_only.Y, 'o', color="tab:red")

        # STBO
        ax_gp_stbo.plot(x_draw, y_mean, label="STBO")
        ax_gp_stbo.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp_stbo.plot(self.X, self.Y, 'o', color="tab:red")
        ax_gp_stbo.legend()

        # subplot 3: STBO AC function 
        ax_ac.set_title("Shape Transfer BO EI Acquisition Function")
        for ac_value, kessi in zip(ac_values_lst, kessis):
            ax_ac.plot(x_draw, ac_value, label="kessi: "+str(kessi))

        if highlight_point != None:
            ax_ac.plot(highlight_point[0], highlight_point[1], 'o', color="tab:red")

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./images/example_stbo_ac_ei.pdf")

        return 0


class BiasCorrectedBO(ExpectedImprovement, UpperConfidenceBound):
    """
    class BiasCorrectedBO: 
        A competitive method proposed in paper: 
        Regret Bounds for Transfer Learning in Bayesian Optimization

    """
    def __init__(self):
        super(BiasCorrectedBO, self).__init__()
        self.zeroGP1 = None
        self.diffGP = None

    def build_task1_gp(self, file_exp_task1, theta_task1=1.0):
        "build ZeroGProcess for task1 with known experiment points"
        zeroGP1 = ZeroGProcess()
        zeroGP1.get_data_from_file(file_exp_task1)
        zeroGP1.theta = theta_task1
        assert(len(zeroGP1.X) == len(zeroGP1.Y))

        self.zeroGP1 = zeroGP1

        return 0
    
    def build_diff_gp(self):
        "build ZeroGProcess on difference between task2 and task1_gp"

        diffGP = ZeroGProcess()
        assert(len(self.X) == len(self.Y))

        # compute difference between task2 and task1_gp
        X_task2 = self.X
        Y_task2 = self.Y

        diff_Y = []

        for point, y_task2 in zip(X_task2, Y_task2):
            mean_GP1_point = self.zeroGP1.compute_mean(point)
            diff_y_point = y_task2 - mean_GP1_point
            diff_Y.append(diff_y_point)
        
        diffGP.Y = diff_Y
        diffGP.X = copy.deepcopy(self.X)
        self.diffGP = diffGP

        # bias correct data of zeroGP1 
        X1 = self.zeroGP1.X
        Y1 = self.zeroGP1.Y

        assert(len(X1) == len(Y1))

        # only add points from X1, which are not in exp 2
        for i in range(len(X1)):
            if X1[i] not in self.X:
                self.Y.append(Y1[i] + diffGP.compute_mean(X1[i]))
                self.X.append(X1[i])
        
        return 0

    def plot_ei(self, num_points=100, exp_ratio=1, confidence=0.9, kessis=[0.0], highlight_point=None):
        "plot the acquisition function as well as ZeroGP&STBO in a figure with two figs"
        min_point_exp1 = min(self.zeroGP1.X)[0]
        min_point_exp2 = min(self.X)[0]
        min_point = min(min_point_exp1, min_point_exp2)

        max_point_exp1 = max(self.zeroGP1.X)[0]
        max_point_exp2 = max(self.X)[0]
        max_point = max(max_point_exp1, max_point_exp2)

        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        # subplot 1: GProcess means & confidence bands of zeroGP1 (on data 1) and diffGP (on task2 - mean1)
        GP1 = self.zeroGP1
        y1_mean = [GP1.compute_mean([ele]) for ele in x_draw]
        y1_conf_int = [GP1.conf_interval([ele]) for ele in x_draw]
        y1_lower = [ele[0] for ele in y1_conf_int]
        y1_upper = [ele[1] for ele in y1_conf_int]

        diffGP = self.diffGP
        yDiff_mean = [diffGP.compute_mean([ele]) for ele in x_draw]
        yDiff_conf_int = [diffGP.conf_interval([ele]) for ele in x_draw]
        yDiff_lower = [ele[0] for ele in yDiff_conf_int]
        yDiff_upper = [ele[1] for ele in yDiff_conf_int]

        # subplot 2: GProcess means & confidence bands of Bias Corrected Bayeisan Optimization
        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 3: EI AC function with multiple parameters 
        ac_values_lst = []
        if isinstance(kessis, list):
            for kessi in kessis:
                ac_kessi = [self.aux_func_ei([ele], kessi) for ele in x_draw]
                ac_values_lst.append(ac_kessi)
        elif isinstance(kessis, float):
            ac_kessi = [self.aux_func_ei([ele], kessis) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            kessis = [kessis]

        fig, (ax_gp1_diff, ax_gp_bcbo, ax_ac) = plt.subplots(3, 1, sharex=True)

        # subplot 1: GP1 & diffGP
        ax_gp1_diff.set_title("GProcess Confidence Band")
        ax_gp1_diff.plot(x_draw, y1_mean, label="GP1")
        ax_gp1_diff.fill_between(x_draw, y1_lower, y1_upper, alpha=0.2)
        ax_gp1_diff.plot(GP1.X, GP1.Y, 'o', color="tab:blue")

        # 
        ax_gp1_diff.plot(x_draw, yDiff_mean, label="diffGP")
        ax_gp1_diff.fill_between(x_draw, yDiff_lower, yDiff_upper, alpha=0.2)
        ax_gp1_diff.plot(diffGP.X, diffGP.Y, 'o', color="tab:red")
        ax_gp1_diff.legend()

        # subplot 2: GP2 only 
        ax_gp_bcbo.set_title("GProcess Confidence Band")
        ax_gp_bcbo.plot(x_draw, y_mean, label="BCBO")
        ax_gp_bcbo.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp_bcbo.plot(self.X, self.Y, 'o', color="tab:red")
        ax_gp_bcbo.legend()

        # subplot 3: STBO AC function 
        ax_ac.set_title("Bias Corrected BO EI Acquisition Function")
        for ac_value, kessi in zip(ac_values_lst, kessis):
            ax_ac.plot(x_draw, ac_value, label="kessi: "+str(kessi))

        if highlight_point != None:
            ax_ac.plot(highlight_point[0], highlight_point[1], 'o', color="tab:red")

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./images/example_bcbo_ac_ei.pdf")

        return 0


if __name__ == "__main__":
    "Main part: iteratively analyze results by adding points one by one"
    EXP = "Double2Double"      # "2D_Triple2Triple"  |  "Double2Double" | "Double2Triple" | "Triple2Double" 
    from_task1 = "sample"      # "rand"          | "gp"            | "sample"        | "mean"

    # visulization only on 1-dimensional examples 
    if EXP == "2D_Triple2Triple":
        i = 2
        file_task1_gp = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task1_gp.tsv"
        file_task1_rand = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task1_rand.tsv"   
        file_task0_sample = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task0_sample.tsv" 
        file_task1_sample_stbo = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task1_sample_stbo.tsv" 
        file_task0_mean = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task0_mean.tsv" 
        file_task1_mean_stbo = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task1_mean_stbo.tsv" 

        file_task2_gp_from_gp = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_gp_from_gp.tsv" 
        file_task2_stbo_from_gp = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_stbo_from_gp.tsv"
        file_task2_bcbo_from_gp = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_bcbo_from_gp.tsv"   

        file_task2_gp_from_rand = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_gp_from_rand.tsv" 
        file_task2_stbo_from_rand = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_stbo_from_rand.tsv"
        file_task2_bcbo_from_rand = "data/2D_Triple2Triple_5sample_2bad_prior/" + str(i) + "/simTriple2Triple2D_points_task2_bcbo_from_rand.tsv" 
    elif EXP == "Triple2Double":
        i = 2
        file_task1_gp = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task1_gp.tsv"
        file_task1_rand = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task1_rand.tsv"   
        file_task0_sample = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task0_sample.tsv" 
        file_task1_sample_stbo = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task1_sample_stbo.tsv" 
        file_task0_mean = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task0_mean.tsv" 
        file_task1_mean_stbo = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task1_mean_stbo.tsv" 

        file_task2_gp_from_gp = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_gp_from_gp.tsv" 
        file_task2_stbo_from_gp = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_stbo_from_gp.tsv"
        file_task2_bcbo_from_gp = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_bcbo_from_gp.tsv"   

        file_task2_gp_from_rand = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_gp_from_rand.tsv" 
        file_task2_stbo_from_rand = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_stbo_from_rand.tsv"
        file_task2_bcbo_from_rand = "data/Triple2Double_sample/" + str(i) + "/simTriple2Double_points_task2_bcbo_from_rand.tsv" 
    elif EXP == "Double2Triple":
        i = 1
        file_task1_gp = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task1_gp.tsv"
        file_task1_rand = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task1_rand.tsv"
        file_task0_sample = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task0_sample.tsv"
        file_task1_sample_stbo = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task1_sample_stbo.tsv"
        file_task0_mean = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task0_mean.tsv"
        file_task1_mean_stbo = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task1_mean_stbo.tsv"

        file_task2_gp_from_gp = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_gp_from_gp.tsv" 
        file_task2_stbo_from_gp = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_stbo_from_gp.tsv"
        file_task2_bcbo_from_gp = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_bcbo_from_gp.tsv" 

        file_task2_gp_from_rand = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_gp_from_rand.tsv" 
        file_task2_stbo_from_rand = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_stbo_from_rand.tsv"
        file_task2_bcbo_from_rand = "data/Double2Triple_0.5/" + str(i) + "/simDouble2Triple_points_task2_bcbo_from_rand.tsv"     
    elif EXP == "Double2Double":
        i = 1
        file_task1_gp = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task1_gp.tsv"
        file_task1_rand = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task1_rand.tsv"
        file_task0_sample = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task0_sample.tsv"
        file_task1_sample_stbo = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task1_sample_stbo.tsv"
        file_task0_mean = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task0_mean.tsv"
        file_task1_mean_stbo = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/" + str(i) + "/simDouble2Double_points_task1_mean_stbo.tsv"

        file_task2_gp_from_gp = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_gp_from_gp.tsv" 
        file_task2_stbo_from_gp = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_stbo_from_gp.tsv"
        file_task2_bcbo_from_gp = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_bcbo_from_gp.tsv" 

        file_task2_gp_from_rand = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_gp_from_rand.tsv" 
        file_task2_stbo_from_rand = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_stbo_from_rand.tsv"
        file_task2_bcbo_from_rand = "data/Double2Double_5sample_1bad_prior_0.1rOutBound/simDouble2Double_points_task2_bcbo_from_rand.tsv"        

    # Test UCB 
    # UCB = UpperConfidenceBound()
    # UCB.get_data_from_file(file_task1_gp)

    # gamma = 0.9
    # x1 = [1.5]
    # print("UCB({:.2f}) = {:.2f}".format(x1[0], UCB.aux_func_ucb(x1, gamma)))
    # x2 = [10.4]
    # print("UCB({:.2f}) = {:.2f}".format(x2[0], UCB.aux_func_ucb(x2, gamma)))

    # UCB.plot(gammas=[0], exp_ratio=0.2)

    if from_task1 == "gp":
        # Test task1 with EI (gp)
        EI = ExpectedImprovement()
        EI.get_data_from_file(file_task1_gp)

        gamma = 0.9
        x1 = [1.5]
        print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, gamma)))
        x2 = [10.4]
        print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, gamma)))

        EI.plot_ei(exp_ratio=1)

        # Test STBO from gp
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(file_task2_stbo_from_gp)
        STBO.build_task1_gp(file_task1_gp, theta_task1=1.0)
        STBO.build_diff_gp()

        STBO.plot_ei(exp_ratio=0.2)

        # Test BCBO from gp
        BCBO = BiasCorrectedBO()
        BCBO.get_data_from_file(file_task2_bcbo_from_gp)
        BCBO.build_task1_gp(file_task1_gp, theta_task1=1.0)
        BCBO.build_diff_gp()

        BCBO.plot_ei(exp_ratio=0.2)
    elif from_task1 == "rand":
        # Test EI with rand
        EI = ExpectedImprovement()
        EI.get_data_from_file(file_task1_rand)
    
        gamma = 0.9
        x1 = [1.5]
        print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, gamma)))
        x2 = [10.4]
        print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, gamma)))
    
        EI.plot_ei(exp_ratio=0.2)
    
        # Test STBO from rand
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(file_task2_stbo_from_rand)
        STBO.build_task1_gp(file_task1_rand, theta_task1=1.0)
        STBO.build_diff_gp()
    
        STBO.plot_ei(exp_ratio=0.2)
    
        # Test BCBO from rand
        BCBO = BiasCorrectedBO()
        BCBO.get_data_from_file(file_task2_bcbo_from_rand)
        BCBO.build_task1_gp(file_task1_rand, theta_task1=1.0)
        BCBO.build_diff_gp()

        BCBO.plot_ei(exp_ratio=0.2)
    elif from_task1 == "sample":
        # Test EI with rand
        EI = ExpectedImprovement()
        EI.get_data_from_file(file_task1_gp)
        EI.theta = 0.7

        gamma = 0.9
        x1 = [1.5]
        print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, gamma)))
        x2 = [10.4]
        print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, gamma)))
    
        EI.plot_ei(exp_ratio=0.0)

        # Test STBO from rand
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(file_task1_sample_stbo)
        STBO.build_task1_gp(file_task0_sample, theta_task1=0.5, prior_mean=0.0, r_out_bound=0.1)
        STBO.build_diff_gp()

        STBO.plot_ei(exp_ratio=0.5)
        #STBO.plot_ac(exp_ratio=1.0)
    elif from_task1 == "mean":
        # Test EI with rand
        EI = ExpectedImprovement()
        EI.get_data_from_file(file_task0_mean)   # file_task1_gp
        EI.theta = 0.7

        # gamma = 0.9
        # x1 = [1.5]
        # print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, gamma)))
        # x2 = [10.4]
        # print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, gamma)))
    
        EI.plot_ei(exp_ratio=0.0)

        # Test STBO from rand
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(file_task1_mean_stbo)
        STBO.build_task1_gp(file_task0_mean, theta_task1=0.7, prior_mean=0.0, r_out_bound=0.2)
        STBO.build_diff_gp()

        STBO.plot_ei(exp_ratio=0.3)

