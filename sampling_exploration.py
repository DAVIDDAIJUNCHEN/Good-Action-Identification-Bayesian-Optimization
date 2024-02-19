#!/usr/bin/env python3

from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import ShapeTransferBO
from optimization import BiasCorrectedBO
from simfun import exp_mu, tri_exp_mu
from scipy.stats import norm, qmc
import numpy as np
from main_simulation import get_best_point, write_exp_result


def triple_fun(x):
    "triple function with given centers & responses"
    dim = 1
    lambda1 = 1; lambda2 = 1.5; lambda3 = 1.25
    mu1 = [0]; mu2 = [5]; mu3 = [10]
    theta1 = 1; theta2 = 1; theta3 = 1

    res = tri_exp_mu(x, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)

    return res

def generate_task1_samples(mu, theta, num, low, high, out_file):
    "generate task1 samples from mono exp"
    dim = 1 
    sampler = qmc.LatinHypercube(dim, centered=False, scramble=True, strength=1, optimization=None, seed=None)
    sample_01 = sampler.random(n=num)
    sample_scaled = qmc.scale(sample_01, low, high)

    response_lst = []

    for i in range(num):
        response_lst.append(exp_mu(sample_scaled[i], mu, theta)) 
    
    with open(out_file, "w", encoding="utf-8") as fout:
        fout.writelines("response#dim1"  + '\n')
        for i in range(num):
            line_out = str(response_lst[i]) + '\t' + str(sample_scaled[i][0]) + '\n'
            fout.writelines(line_out)
            
    return 0


if __name__ == "__main__":
    # Part 1: Weak Prior 
    mu=[6.5]; theta=1; num=30; low=-5; high=15
    out_file_task1 = "./data/Sample_Exp/f1_mu_6.5_theta_1.tsv"
    out_file_task2_stbo = "./data/Sample_Exp/f2_stbo_prior_mu_6.5_theta_1.tsv"
    out_file_task2_gp = "./data/Sample_Exp/f2_gp_prior_mu_6.5_theta_1.tsv"

    stage = 2

    if stage == 1:
        generate_task1_samples(mu, theta, num, low, high, out_file_task1)
        best_point_task1 = get_best_point(out_file_task1)
        res2_point = triple_fun(best_point_task1)

        with open(out_file_task2_stbo, "w", encoding="utf-8") as fin:
            fin.writelines("response#dim1" + '\n')
        
        with open(out_file_task2_gp, "w", encoding="utf-8") as fin:
            fin.writelines("response#dim1" + '\n')

        write_exp_result(out_file_task2_stbo, res2_point, best_point_task1)
        write_exp_result(out_file_task2_gp, res2_point, best_point_task1)
    elif stage == 2:
        num_start = 30; lr = 0.5; num_step=20
        start_points = [np.random.uniform(low, high, size=1).tolist() for i in range(num_start)]

        # draw picture by EI
        EI = ExpectedImprovement()
        EI.get_data_from_file(out_file_task2_gp)
        EI.plot_ei(exp_ratio=1)

        # draw picture by STBO
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(out_file_task2_stbo)
        STBO.build_task1_gp(out_file_task1)
        STBO.build_diff_gp()
        STBO.plot_ei(exp_ratio=0.2)

        # find best point by GP
        next_point_gp, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr, 
                                                                  num_step=num_step, kessi=0.0)        
        next_response_gp = triple_fun(next_point_gp)
        write_exp_result(out_file_task2_gp, next_response_gp, next_point_gp)

        # find best point by STBO
        next_point_stbo, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr,
                                                                      num_step=num_step, kessi=0.0)
        next_response_stbo = triple_fun(next_point_stbo)
        write_exp_result(out_file_task2_stbo, next_response_stbo, next_point_stbo)

