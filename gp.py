#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc
from smt.sampling_methods import LHS
import itertools
from utils import write_exp_result, dist, find_max_min_of_each_component, check_inBounds


class ZeroGProcess:
    """
    Class ZeroGProgress: build zero mean Gaussian Process with known or unknown sigma (vairiance)
    """
    def __init__(self, sigma_square=None, type_kernel="gaussian", param_kernel=1.0, prior_mean=None, r_out_bound=1.0) -> None:
        self.Y = [] 
        self.X = []
        self.dim = None
        self.num_points = None
        self.sigma2 = sigma_square          # None => sigam is unknown => MLE
        self.kernel_type = type_kernel
        self.theta = param_kernel
        self.prior_mean = prior_mean
        self.r_out_bound = r_out_bound      # mean ratio for out of boundary

    def get_data_from_file(self, file_exp):
        "get response vec and input from file_in"
        res_col = 0  # default column number of response
        num_points = 0

        with open(file_exp, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if '#' in line:  # with header line => get column of response
                    lst_line = line.split('#')
                    lst_line = [ele.strip() for ele in lst_line]
                    dim_header = len(lst_line) - 1
                    self.dim = dim_header
                    lst_res_col = [ele=="response" for ele in lst_line]
                    res_col = lst_res_col.index(True)
                    continue
                else:
                    lst_line = line.split()
                    lst_line = [ele.strip() for ele in lst_line]
                    dim_nonHeader = len(lst_line) - 1
                    assert(dim_header == dim_nonHeader)   # ensure all lines have same dim

                    lst_line = [float(pnt) for pnt in lst_line]
                    self.Y.append(lst_line[res_col])
                    self.X.append(lst_line[:res_col] + lst_line[(res_col+1):])
                    num_points += 1

        self.num_points = num_points

        # minus the prior mean if it is not None
        if self.prior_mean != None:
            self.Y = [ele - self.prior_mean for ele in self.Y]

        return 0

    def kernel(self, x1, x2, theta=1.0):
        "compute k(x1, x2) with Gaussian | Matern Kernel"
        # do not allow to call any methods before get experiment data
        assert(len(self.X) == len(self.Y))
        assert(len(self.X) > 0)

        # if data were not imported from get_data_from_file
        if (self.dim==None) or (self.num_points==None):
            self.dim = len(self.X[0])
            self.num_points = len(self.X)

        # compute k(x1, x2)
        if self.kernel_type == "gaussian":
            x1_x2 = [ele1 - ele2 for ele1, ele2 in zip(x1, x2)]
            norm2_x1_x2 = sum([ele**2 for ele in x1_x2])
            k_x1_x2 = np.exp(- 1/(2*theta**2)*norm2_x1_x2)
        elif self.kernel_type == "matern":
            pass
        
        return k_x1_x2

    def kernel_grad(self, x1_grad_pos, x2, theta=1.0):
        "compute gradient of k(x1_grad_pos, x2) at x1_grad_pos"

        if self.kernel_type == "gaussian":
            x1_x2 = [ele1 - ele2 for ele1, ele2 in zip(x1_grad_pos, x2)]
            grad_kernel_x1 = [(-1 / theta**2)*ele* self.kernel(x1_grad_pos, x2) for ele in x1_x2]
        elif self.kernel_type == "matern":
            pass

        return grad_kernel_x1

    def compute_kernel_cov(self, lst_exp_points, theta=1.0):
        "compute kernel covariance matrix: K = (k(x_i, x_j))"

        dim = len(lst_exp_points)
        kernel_Cov = np.zeros(shape=(dim, dim))

        for i in range(dim):
            for j in range(dim):
                x_i = lst_exp_points[i]
                x_j = lst_exp_points[j]
                kernel_Cov[i, j] = self.kernel(x_i, x_j, theta)
        
        return kernel_Cov

    def compute_kernel_vec(self, lst_exp_points, current_point, theta=1.0):
        "compute kernel vector at current_point: k_x = (k(x, x_i)), column vector"
        dim = len(lst_exp_points)
        kernel_Vec = np.zeros(shape=(dim, 1))

        for i in range(dim):
            x_i = lst_exp_points[i]
            kernel_Vec[i, 0] = self.kernel(current_point, x_i, theta)
     
        return kernel_Vec

    def compute_grad_kernel_vec(self, lst_exp_points, current_point, theta=1.0):
        "compute gradient of row kernel vector k_x^T = (k(x, x_i))^T, matrix (d*t)"
        dim = len(lst_exp_points)
        grad_kernel_vec = []

        for i in range(dim):
            x_i = lst_exp_points[i]
            grad_kernel_xi = self.kernel_grad(current_point, x_i, theta)
            grad_kernel_vec.append(grad_kernel_xi)
        
        grad_kernel_vec = np.transpose(np.matrix(grad_kernel_vec))

        return grad_kernel_vec

    def compute_mle_sigma2(self):
        "compute the MLE(maximum likelihood estimation) of sigma^2"
        response_vec = np.transpose(np.matrix(self.Y))
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        sigma2_hat_A = np.matmul(np.matrix(self.Y), inv_kernel_Cov)
        self.num_points = len(self.Y)
        sigma2_hat = np.matmul(sigma2_hat_A, response_vec) / self.num_points

        return sigma2_hat

    def compute_mean(self, current_point):
        "compute the mean value at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        mean_partA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        response_vec = np.transpose(np.matrix(self.Y))
        mean = np.matmul(mean_partA, response_vec)

        if self.prior_mean != None:
            mean[0, 0] = mean[0, 0] + self.prior_mean

        # construct boundary lists
        l_bound = find_max_min_of_each_component(self.X, max=False)
        u_bound = find_max_min_of_each_component(self.X, max=True)

        if check_inBounds(current_point, l_bound, u_bound):
            return mean[0, 0]
        else:
            return self.r_out_bound*mean[0, 0]

    def compute_grad_mean(self, current_point):
        "compute the gradient of mean(x) at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)
        response_vec = np.transpose(np.matrix(self.Y))
        grad_partB = np.matmul(inv_kernel_Cov, response_vec)

        grad_kernel_lst = []
        for x2 in self.X:
            grad_kernel_lst.append(self.kernel_grad(current_point, x2))

        grad_kernel_mat = np.transpose(np.matrix(grad_kernel_lst))

        grad_mean = np.matmul(grad_kernel_mat, grad_partB)

        return grad_mean

    def compute_s2(self, current_point):
        "compute the s^2(x) at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB

        return s2_current[0, 0]

    def compute_var(self, current_point, zeroCheck=1e-13):
        "compute the variance value at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB

        if self.sigma2 == None:
            sigma2_hat = self.compute_mle_sigma2()
            var_current = sigma2_hat * s2_current
        else:
            var_current = self.sigma2 * s2_current

        if var_current[0,0] < zeroCheck: # avoid negative variance (negative but close to zero)
            return 0.0

        return var_current[0, 0]

    def compute_grad_var(self, current_point, zeroCheck=1e-13):
        "compute the gradient of var(x) at current_point"
        if self.sigma2 == None:
            self.sigma2 = self.compute_mle_sigma2()[0,0]
        
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        grad_var_part1 = -2*self.sigma2
        grad_var_part2 = self.compute_grad_kernel_vec(self.X, current_point, self.theta) 
        grad_var_part3 = np.matmul(inv_kernel_Cov, kernel_Vec_mat)

        grad_var = grad_var_part1 * np.matmul(grad_var_part2, grad_var_part3)

        return grad_var

    def conf_interval(self, current_point, confidence=0.9):
        "compute the confidence interval with two sides"
        alpha = (1 - confidence) / 2.
        lower_bound_std = norm.ppf(alpha)
        upper_bound_std = norm.ppf(1 - alpha)

        mean_current = self.compute_mean(current_point)
        var_current = self.compute_var(current_point)

        lower_bound = mean_current + np.sqrt(var_current)*lower_bound_std
        upper_bound = mean_current + np.sqrt(var_current)*upper_bound_std

        return lower_bound, upper_bound

    def sample(self, num, mean, sigma, l_bounds, u_bounds, prior_points, mean_fix=True, epslon=0.5, out_file="./data/sample_points_task1_gp.tsv"):
        "sample num points from a GP with initial (mean & sigma) and prior points on a LHD"
        
        if len(prior_points) != 0:
            # assert dim of prior points == dim 
            for pnt, rel_qunt in prior_points:
                assert(len(pnt) == self.dim)
        num_prior = len(prior_points)

        # construct max-min LHD 
        num_lhd_sampled = num_prior + num
#        sampler = qmc.LatinHypercube(self.dim, centered=False, scramble=True, strength=1, optimization=None, seed=None)
#        sample_01 = sampler.random(n=num_lhd_sampled)
#        sample_scaled = qmc.scale(sample_01, l_bounds, u_bounds)

        xlimits = [[low_i, up_i] for low_i, up_i in zip(l_bounds, u_bounds)]
        sampler = LHS(xlimits=np.array(xlimits), criterion='maximin')
        sample_scaled = sampler(num_lhd_sampled)

        # sort points in LHD by distance from domain center 
        center = np.array([(l_bound_i + u_bound_i)/2 for l_bound_i, u_bound_i in zip(l_bounds, u_bounds)])
        sample_scaled_sorted = sorted(sample_scaled, key=lambda x:np.linalg.norm(x-center))
        sample_scaled_sorted = [list(arr_i) for arr_i in sample_scaled_sorted]

        # remove LHD points collapsed with prior points
        for pnt, _ in prior_points:
            d_pnt_sample = [dist(pnt, pnt_i) for pnt_i in sample_scaled_sorted]
            min_dist_index = np.where(d_pnt_sample==np.min(d_pnt_sample))[0][0]
            sample_scaled_sorted.pop(min_dist_index)

        # write header and prior points in out_file
        feat_tag = ["#dim"+str(i+1) for i in range(self.dim)]
        first_line = "response"

        for i in range(self.dim):
            first_line = first_line + feat_tag[i]
            
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(first_line + '\n')
        
        for pnt, rel_qunt in prior_points:
            res_pnt = mean + (rel_qunt-1)*np.abs(mean)
            write_exp_result(out_file, res_pnt, pnt)

        # sample points based on prior & mean & LHD 
        zeroGP1 = ZeroGProcess(sigma_square=sigma**2)
        zeroGP1.get_data_from_file(out_file)
        zeroGP1.theta = 0.2
        zeroGP1.Y = [y - mean for y in zeroGP1.Y]

        for i in range(num):
            sample_x = sample_scaled_sorted[i]
            sample_x_str = ''
            for j in range(self.dim):
                sample_x_str = sample_x_str + '\t' + str(sample_x[j])
            
            if i == 0 and len(prior_points) == 0:
                if mean_fix:
                    sample_str = str(mean) + sample_x_str
                else:
                    sample_str = str(np.random.normal(mean, sigma)) + sample_x_str
            else:
                if mean_fix:
                    sample_response = mean
                else:
                    if len(prior_points) == 0:
                        mean_x_i = mean
                        var_x_i = sigma**2
                    else:
                        mean_x_i = mean + zeroGP1.compute_mean(sample_scaled_sorted[i])
                        var_x_i = zeroGP1.compute_var(sample_scaled_sorted[i])

                    sample_response = np.random.normal(mean_x_i, np.sqrt(var_x_i))

                sample_str = str(sample_response) + sample_x_str

            with open(out_file, 'a', encoding="utf-8") as f_out:
                f_out.writelines(sample_str + '\n')

        # give very low values outside the boundary
        # x_boundary = [[low_i-epslon, up_i+epslon] for low_i, up_i in zip(l_bounds, u_bounds)]
        # with open(out_file, 'a', encoding="utf-8") as f_out:
        #     for b_pnt in itertools.product(*x_boundary):
        #         b_pnt_x = list(b_pnt)
        #         sample_x_str = ''
        #         for j in range(self.dim):
        #             sample_x_str = sample_x_str + '\t' + str(b_pnt_x[j])
        #             sample_str = str(0.1*mean) + sample_x_str
        #         f_out.writelines(sample_str + '\n')

        return 0

    def plot(self, num_points=100, exp_ratio=1, confidence=0.9):
        """
        draw Gaussian Process mean values and confidence interval
        num_points: number of points evenly distributed [min_x-delta, max_x+delta]
        confidence: confidence bands with probablity (confidence)
        exp_ratio: [min_x - exp_ratio*delta, max_x + exp_ratio*delta], delta = max_x - min_x  
        """
        min_point = min(self.X)[0]
        max_point = max(self.X)[0]
        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        y_mean = [self.compute_mean([ele]) for ele in x_draw]

        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        fig, ax = plt.subplots()
        ax.plot(x_draw, y_mean)
        ax.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        if self.prior_mean == None:
            ax.plot(self.X, self.Y, 'o', color="tab:red")
        else:
            Y = [ele + self.prior_mean for ele in self.Y]
            ax.plot(self.X, Y, 'o', color="tab:red")

        fig.savefig("./example.png")

        return 0


###### TODO: GProcess with regression components ######
#class GProcess():


if __name__ == "__main__":
    # Instance 
    zeroGP = ZeroGProcess()

    # ASR experiment data 
    zeroGP.get_data_from_file("data/experiment_points_task1_gp.tsv")
    #zeroGP.get_data_from_file("data/simDouble2Double_points_task1_sample.tsv")

    print(zeroGP.X)
    print(zeroGP.Y)
    
    # sample from GP
    lower_bound = [0, 0, 0, 0]
    upper_bound = [5000, 5000, 5000, 5000]
    #lower_bound = [-5]
    #upper_bound = [10]

    prior_pnt = [([1758, 2194, 4496, 3444], 1.2), ([2000, 2000, 500, 3000], 1.2), ([4000, 500, 4000, 500], 1.2)]
    zeroGP.sample(num=5, mean=0.8, sigma=0.01, prior_points=prior_pnt, l_bounds=lower_bound, u_bounds=upper_bound, mean_fix=False)

    # verify functions at x
    x = [9, 10, 11, 12]

    print(zeroGP.compute_mean(x))
    print(zeroGP.compute_var(x))
    print(zeroGP.conf_interval(x))

    print(zeroGP.compute_grad_mean(x))
    print(zeroGP.compute_grad_kernel_vec(zeroGP.X, x))
    print(zeroGP.compute_grad_var(x))

    # plot images
    #zeroGP.plot(num_points=100, exp_ratio=0.0, confidence=0.90)

    # GP based on samples 
    zeroGP_sample = ZeroGProcess(prior_mean=0.5, param_kernel=0.7) 
    zeroGP_sample.get_data_from_file("data/sample_points_task1_gp.tsv")

    zeroGP_sample.plot(num_points=100, exp_ratio=0.0, confidence=0.90)

