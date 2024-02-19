#!/usr/bin/env /user_names/python3

import os 
import numpy as np

from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import ShapeTransferBO
from optimization import BiasCorrectedBO


if __name__ == "__main__":
    """" Run Real experiments """
    task = 1

    if task == 1:
        # Part 1: EI on Experiment 1
        # 1.1 ZeroGP EI initialization
        EI_task1 = ExpectedImprovement()
        EI_task1.get_data_from_file("data/experiment_points_task1_gp.tsv")

        # 1.2 AC optimization 
        dim = EI_task1.dim
        low_opt1 = 0; high_opt1 = 5
        low_bounds = [low_opt1 for i in range(dim)]
        up_bounds  = [high_opt1 for i in range(dim)]
        
        num_start_opt1 = 20
        start_points_rand = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]
        kessi = 0

        next_point, next_point_aux = EI_task1.find_best_NextPoint_ei(start_points_rand, l_bounds=low_bounds, u_bounds=up_bounds, learn_rate=0.1, num_step=300, kessi=kessi)
        print("GP next point in task1: ", next_point, "AC: ", next_point_aux)

        # Part 2: Efficient Bayesian Optimization with sampling + weak priors
        # 2.0 Create start points
        EI_task0 = ExpectedImprovement()
        EI_task0.get_data_from_file("./data/sample_points_task1_efficient_BO.tsv")
        start_points_task1 = EI_task0.X

        start_points_task1.extend(start_points_rand)
        start_points = start_points_task1

        # 2.1 Efficient BO with sampling points
        mean_sample_low = 0.8
        STBO_task1_sample = ShapeTransferBO()
        STBO_task1_sample.get_data_from_file("./data/sample_points_task1_efficient_BO.tsv")
        STBO_task1_sample.build_task1_gp("./data/experiment_points_task1_efficient_BO_sample.tsv", theta_task1=0.7*np.sqrt(dim), prior_mean=mean_sample_low, r_out_bound=0.1)
        STBO_task1_sample.build_diff_gp()

        # 2.2 AC optimization (shared the same start points as in gp)
        kessi = 0
        next_point_stbo1_sample, next_point_aux = STBO_task1_sample.find_best_NextPoint_ei(start_points, l_bounds=low_bounds, u_bounds=up_bounds,
                                                                      learn_rate=0.1, num_step=300, kessi=kessi)
        print("Sample BO, next point in task1: ", next_point_stbo1_sample, " AC: ", next_point_aux)

        # Part 3: Efficient Bayesian Optimization with mean + weak priors
        # 3.0 Create start points
        EI_task0 = ExpectedImprovement()
        EI_task0.get_data_from_file("./data/mean_points_task1_efficient_BO.tsv")
        start_points_task1 = EI_task0.X

        start_points_task1.extend(start_points_rand)
        start_points = start_points_task1

        # 3.1 Efficient BO with mean points
        mean_sample_low = 0.8
        STBO_task1_mean = ShapeTransferBO()
        STBO_task1_mean.get_data_from_file("./data/mean_points_task1_efficient_BO.tsv")
        STBO_task1_mean.build_task1_gp("./data/experiment_points_task1_efficient_BO_mean.tsv", theta_task1=0.7*np.sqrt(dim), prior_mean=mean_sample_low, r_out_bound=0.1)
        STBO_task1_mean.build_diff_gp()

        # 3.2 AC optimization (shared the same start points as in gp)
        kessi = 0 
        next_point_stbo1_mean, next_point_aux = STBO_task1_mean.find_best_NextPoint_ei(start_points, l_bounds=low_bounds, u_bounds=up_bounds,
                                                                      learn_rate=0.1, num_step=300, kessi=kessi)
        print("Mean BO, next point in task1: ", next_point_stbo1_mean, " AC: ", next_point_aux)
    elif task == 2:
        # Part 4: EI on Experiment 2
        # 4.0 Create start points
        EI_task1 = ExpectedImprovement()
        EI_task1.get_data_from_file("data/experiment_points_task1_gp.tsv")
        start_points_task1 = EI_task1.X

        dim = EI_task1.dim              # dim_task1 = dim_task2
        low_opt2 = 0; high_opt2 = 5000
        num_start_opt2 = 10
        start_points_rand = [np.random.uniform(low_opt2, high_opt2, size=dim).tolist() for i in range(num_start_opt2)]

        start_points_task1.extend(start_points_rand)
        start_points = start_points_task1 

        # 4.1 ZeroGP EI initialization
        EI_task2 = ExpectedImprovement()
        EI_task2.get_data_from_file("data/experiment_points_task2_gp.tsv")
        print(EI_task2.X)
        print(EI_task2.Y)

        # 4.2 AC optimization 
        kessi = 0
        next_point, next_point_aux = EI_task2.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=300, kessi=kessi)

        # EI.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("GP next point in task2: ", next_point, " AC: ", next_point_aux)        

        # Part 5: STBO on Experiments 2
        # 5.1 STBO EI initialization
        STBO = ShapeTransferBO()
        STBO.get_data_from_file("./data/experiment_points_task2_STBO.tsv")
        STBO.build_task1_gp("./data/experiment_points_task1_gp.tsv")
        STBO.build_diff_gp()

        # 5.2 AC optimization (shared the same start points as in gp)
        kessi = 0
        next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=300, kessi=kessi)    
        # STBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("STBO next point in task2: ", next_point, " AC: ", next_point_aux)

        # Part 6: BCBO on Experiment 2
        # 6.1 BCBO EI initialization
        BCBO = BiasCorrectedBO()
        BCBO.get_data_from_file("data/experiment_points_task2_BCBO.tsv")

        print("before bias correction")
        print(BCBO.X)
        print(BCBO.Y)

        BCBO.build_task1_gp("./data/experiment_points_task1_gp.tsv")
        BCBO.build_diff_gp()

        print("after bias correction & merge")
        print(BCBO.X)
        print(BCBO.Y)

        # 6.2 AC optimization (shared the same start points as in gp)
        kessi = 0
        next_point, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=300, kessi=kessi)    
        
        # BCBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("BCBO next point in task2: ", next_point, " AC: ", next_point_aux)
    else:
        raise(ValueError)

