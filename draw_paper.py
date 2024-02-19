#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics
from optimization import ExpectedImprovement, ShapeTransferBO
from gp import ZeroGProcess
from simfun import two_exp_mu, tri_exp_mu


# Part 1: images for simulation 1
def show_exp(mu1=[0.1, 0.1], mu2=[0.4163, 0.4163], m3=[1.0,1.0], theta=0.5, x_low=-1, x_up=1, y_low=-1, y_up=1, x_nums=100, y_nums=100):

    X = np.arange(x_low, x_up, 0.25)
    Y = np.arange(y_low, y_up, 0.25)
    X, Y = np.meshgrid(X, Y)

    exp0_norm2 = X**2 + Y**2 
    exp1_norm2 = (X- mu1[0])**2 + (Y - mu1[1])**2
    exp2_norm2 = (X- mu2[0])**2 + (Y - mu2[1])**2
    exp3_norm2 = (X- mu3[0])**2 + (Y - mu3[1])**2

    exp0 = np.exp(-0.5 * exp0_norm2 / theta**2)
    exp1 = np.exp(-0.5 * exp1_norm2 / theta**2)
    exp2 = np.exp(-0.5 * exp2_norm2 / theta**2)
    exp3 = np.exp(-0.5 * exp3_norm2 / theta**2)

    exp = [exp1, exp2, exp3]
    mu = [mu1, mu2, mu3]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(3):     
        Z1 = exp0
        Z2 = exp[i]
    
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_wireframe(X, Y, Z1, rstride=2, cstride=2, color="red")
        ax.plot_wireframe(X, Y, Z2, rstride=2, cstride=2)
    
        ax.set_title(r"$\mu$"+"=("+str(mu[i][0]) +","+str(mu[i][1])+")", fontsize=25)

    plt.show()

    fig.savefig("./images/exp_3d_paper")

    return 0


# Part 2: Double2Double
def show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(title[i+1])

        for item in sorted(dct_medium_perc[i].items()):
            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]
            
            if "task2_bcbo_from_rand.tsv" in item[0]:
                label = "Diff-GP from rand"
                fmt = '-.^'
                color = "blue"
            elif "task2_stbo_from_rand.tsv" in item[0]:
                label = "STBO from rand"
                fmt = '-o'
                color = "green"
            elif "task2_gp_from_rand.tsv" in item[0]:
                label = "EI from rand"
                fmt = '--s'
                color = "red"
            elif "task2_bcbo_from_gp.tsv" in item[0]:
                label = "Diff-GP from EI"
                fmt = '-.^'
                color = "blue"
            elif "task2_stbo_from_gp.tsv" in item[0]:
                label = "STBO from EI"
                fmt = '-o'
                color = "green"
            elif "task2_gp_from_gp.tsv" in item[0]:
                label = "EI from EI"
                fmt = '--s'
                color = "red"
            elif "task2_gp_from_cold.tsv" in item[0]:
                label = "EI from cold"
                fmt = '--s'
                color = "red"

            ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            ax.set_xticks(np.arange(0, 21, 5))
        plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 5)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


# Part 3: EXP function family
def show_EXP_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.7))

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    subfig1, subfig2 = fig.subfigures(2, 1)

    subfigs = [subfig1, subfig2]

    for row in range(2):
        subfigs[row].suptitle(title[row])
        for col in range(3):
            ax = subfigs[row].add_subplot(1, 3, col + 1)
            ax.set_title("$\mu=$"+title[2 + col])

            for item in sorted(dct_medium_perc[col][row].items()):
                x_draw = np.arange(len(item[1]))
                x_draw = [ele + 1 for ele in x_draw]
                y_medium = [ele[1] for ele in item[1]]
                y_perc25 = [ele[1] - ele[0] for ele in item[1]]
                y_perc75 = [ele[2] - ele[1] for ele in item[1]]
                asymmetric_error = [y_perc25, y_perc75]

                if "task2_bcbo_from_rand.tsv" in item[0]:
                    label = "Diff-GP"
                    fmt = '-.^'
                    color = "blue"
                elif "task2_stbo_from_rand.tsv" in item[0]:
                    label = "STBO"
                    fmt = '-o'
                    color = "green"
                elif "task2_gp_from_rand.tsv" in item[0]:
                    label = "EI from rand"
                    fmt = '--s'
                    color = "red"
                elif "task2_gp_from_cold.tsv" in item[0]:
                    label = "EI from cold"
                    fmt = '--s'
                    color = "red"

                ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
                ax.set_xticks(np.arange(0, 21, 5))

            plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 11)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


# Part 4: rejection effect in 1D
def show_reject_effect_highMean(x_low, x_high, file_sample_task0, file_task1_stbo, lambda1=1, lambda2=1.5, lambda3=1.25,
                        mu1=[0], mu2=[5], mu3=[10], theta1=1, theta2=1, theta3=1, kessis=[0]):
    """illustrate the rejection effect in 1D"""
    x_draw = np.linspace(x_low, x_high, 100)

    # Line1: standard line
    GP1 = ZeroGProcess(prior_mean=1.1, r_out_bound=0.1)
    GP1.get_data_from_file(file_sample_task0)
    GP1.theta = 0.7

    y1_mean = [GP1.compute_mean([ele]) for ele in x_draw]
    y1_conf_int = [GP1.conf_interval([ele]) for ele in x_draw]
    y1_lower = [ele[0] for ele in y1_conf_int]
    y1_upper = [ele[1] for ele in y1_conf_int]

    # Line 2: target function
    y_target = [tri_exp_mu([ele], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3) for ele in x_draw]
    print(tri_exp_mu([5.5], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3))    
    
    # Line 3: AC function
    STBO = ShapeTransferBO()
    STBO.get_data_from_file(file_task1_stbo)
    STBO.build_task1_gp(file_sample_task0, theta_task1=0.7, prior_mean=1.1, r_out_bound=0.1)
    STBO.build_diff_gp()

    ac_values_lst = []
    if isinstance(kessis, list):
        for kessi in kessis:
            ac_kessi = [STBO.aux_func_ei([ele], kessi) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            x_next = 4.71
            y_next = 1.438249038075815
            ac_next = STBO.aux_func_ei([x_next], kessis[0])
    elif isinstance(kessis, float):
        ac_kessi = [STBO.aux_func_ei([ele], kessis) for ele in x_draw]
        ac_values_lst.append(ac_kessi)
        kessis = [kessis]

    # Line 4: add rejection region with line
    line_y = 0.04

    # draw in one fig
    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1_mean, label="standard line")
    #ax.fill_between(x_draw, y1_lower, y1_upper, alpha=0.2)
    #ax.plot(GP1.X[2:], [y + 1.1 for y in GP1.Y[2:]], 'o', label="LHS points", color="tab:blue")
    ax.plot(GP1.X[:2], [y + 1.1 for y in GP1.Y[:2]], 'x', label="prior points", markersize=10, color="tab:blue")

    ax.plot(x_draw, y_target, '--', label="target function")
    ax.plot(STBO.X, STBO.Y, 'o', color="tab:red", label="observations")
    ax.plot([x_next], [y_next], '*', label="next evaluation", markersize=10, color="tab:red")

    ax.axvline(x=4.71, color='orange')

    for ac_value, kessi in zip(ac_values_lst, kessis):
        ax.plot(x_draw, ac_value, linestyle='dashdot', label="acquisition function")
        ax.plot([x_next], [ac_next], '*', label="acquisition peak", markersize=10, color="tab:green")
        # add rejection area between AC func and line_y
        ac_value_adapt = [min(ele, line_y) for ele in ac_value]
        ax.fill_between(x_draw, ac_value_adapt, line_y, facecolor="lightblue", edgecolor="white", label="rejection area")

    # add text 
    ax.text(7.80, 0.08, '1', color="red", size=12)
    ax.text(2.80, 0.08, '2', color="red", size=12)
    ax.text(-0.2, 0.86, '3', color="red", size=12)
    ax.text(10.5, 1.18, '4', color="red", size=12)
    ax.text(5.80, 1.30, '5', color="red", size=12)
    ax.text(3.90, 0.3, 'x=4.71', color="red", size=12)
    
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig("./images/raise_bo_reject_highMean.pdf")

    return 0

def show_reject_effect_lowMean(x_low, x_high, file_sample_task0, file_task1_stbo, lambda1=1, lambda2=1.5, lambda3=1.25,
                        mu1=[0], mu2=[5], mu3=[10], theta1=1, theta2=1, theta3=1, kessis=[0]):
    """illustrate the rejection effect in 1D"""
    x_draw = np.linspace(x_low, x_high, 100)

    # Line1: standard line
    GP1 = ZeroGProcess(prior_mean=0.5, r_out_bound=0.1)
    GP1.get_data_from_file(file_sample_task0)
    GP1.theta = 0.7

    y1_mean = [GP1.compute_mean([ele]) for ele in x_draw]
    y1_conf_int = [GP1.conf_interval([ele]) for ele in x_draw]
    y1_lower = [ele[0] for ele in y1_conf_int]
    y1_upper = [ele[1] for ele in y1_conf_int]

    # Line 2: target function
    y_target = [tri_exp_mu([ele], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3) for ele in x_draw]
    print(tri_exp_mu([7.5], lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3))    
    
    # Line 3: AC function
    STBO = ShapeTransferBO()
    STBO.get_data_from_file(file_task1_stbo)
    STBO.build_task1_gp(file_sample_task0, theta_task1=0.7, prior_mean=0.5, r_out_bound=0.1)
    STBO.build_diff_gp()

    ac_values_lst = []
    if isinstance(kessis, list):
        for kessi in kessis:
            ac_kessi = [STBO.aux_func_ei([ele], kessi) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            x_next = -1.27
            y_next = 0.4464401231991101
            ac_next = STBO.aux_func_ei([x_next], kessis[0])
    elif isinstance(kessis, float):
        ac_kessi = [STBO.aux_func_ei([ele], kessis) for ele in x_draw]
        ac_values_lst.append(ac_kessi)
        kessis = [kessis]    

    # Line 4: rejection area
    line_y = 0.05

    # draw in one fig
    fig, ax = plt.subplots(1, 1)
    ax.set_title("")

    ax.plot(x_draw, y1_mean, label="standard line")
    ax.fill_between(x_draw, y1_lower, y1_upper, alpha=0.2)
    ax.plot(GP1.X[2:], [y + 0.5 for y in GP1.Y[2:]], 'o', label="LHS points", color="tab:blue")
    ax.plot(GP1.X[:2], [y + 0.5 for y in GP1.Y[:2]], 'x', label="prior points", markersize=10, color="tab:blue")

    ax.plot(x_draw, y_target, '--', label="target function")
    ax.plot(STBO.X, STBO.Y, 'o', color="tab:red", label="experiment points")
    ax.plot([x_next], [y_next], '*', markersize=10, label="next point to evaluate", color="tab:red")

    for ac_value, kessi in zip(ac_values_lst, kessis):
        ax.plot(x_draw, ac_value, linestyle='dashdot', label="acquisition function")
        ax.plot([x_next], [ac_next], '*', label="acquisition peak", markersize=10, color="tab:green")
        
        # add rejection area between AC func and line_y
        ax.fill_between(x_draw, ac_value, 1)

    ax.axvline(x=-1.27, color='orange')

    # add text 
    ax.text(7.80, 0.08, '1', color="red", size=12)
    ax.text(2.80, 0.08, '2', color="red", size=12)
    ax.text(-0.2, 0.86, '3', color="red", size=12)
    ax.text(-1.25, 0.15, 'x=-1.27', color="red", size=12)
    
    ax.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    fig.savefig("./images/raise_bo_reject_lowMean.pdf")

    return 0

def show_RAISE_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name, means):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])
    mean_1 = means[0]
    mean_2 = means[1]
    mean_3 = means[2]
    mean_4 = means[3]

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(title[i+1])
        for item in sorted(dct_medium_perc[i].items()):
            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]

            if "task1_mean_stbo.tsv" in item[0] and "good" in item[0]:
                label = "RAISE BO with good prior"
                fmt = '-o'
                color = "green"
            elif "task1_mean_stbo.tsv" in item[0] and "close" in item[0]:
                label = "RAISE BO with near prior"
                fmt = '-o'
                color = "green"
            elif "task1_mean_stbo.tsv" in item[0] and "middle" in item[0]:
                label = "RAISE BO with middle prior"
                fmt = '--*'
                color = "orange"
            elif "task1_mean_stbo.tsv" in item[0] and "far" in item[0]:
                label = "RAISE BO with far prior"
                fmt = '-x'
                color = "cyan"                            
            elif "task1_mean_stbo.tsv" in item[0] and "bad" in item[0]:
                label = "RAISE BO with bad prior"
                fmt = '-.^'
                color = "blue"
            elif "task1_sample_stbo.tsv" in item[0] and "noprior" in item[0]:
                label = "RAISE BO without prior"
                fmt = '--o'
                color = "orange"
            elif "task1_gp.tsv" in item[0]:
                label = "EI "
                fmt = '--s'
                color = "red"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_1)+"_low" in item[0]:
                if "Neg" in str(mean_1):
                    mean_1 = str(mean_1)
                    mean_1 = "-" + mean_1.split("Neg")[-1]

                label = "RAISE BO with $\mu="+str(mean_1)+"$"
                fmt = '-x'
                color = "green"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_2)+"_center" in item[0]:
                if "Neg" in str(mean_2):
                    mean_2 = str(mean_2)
                    mean_2 = "-" + mean_2.split("Neg")[-1]

                label = "RAISE BO with $\mu="+str(mean_2)+"$"
                fmt = '--o'
                color = "orange"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_3)+"_center" in item[0]:
                if "Neg" in str(mean_3):
                    mean_3 = str(mean_3)
                    mean_3 = "-" + mean_3.split("Neg")[-1]

                label = "RAISE BO with $\mu="+str(mean_3)+"$"
                fmt = '-.o'
                color = "blue"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_4)+"_high" in item[0]:
                if "Neg" in str(mean_4):
                    mean_4 = str(mean_4)
                    mean_4 = "-" + mean_4.split("Neg")[-1]
                
                label = "RAISE BO with $\mu="+str(mean_4)+"$"
                fmt = '--^'
                color = "cyan"

            ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            ax.set_xticks(np.arange(0, 21, 5))
        plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 5)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


if __name__ == "__main__":
    paper_id = "raise"                # stbo / raise

    if paper_id == "raise":
        topic_means = {}
        topic_means = {"Double2Double": [0.5, 0.1, 1.0, 1.5], 
                       "Triple2Double": [0.5, 0.1, 1.0, 1.5],
                       "2D_forrester":  [0.5, 1.5, "Neg5", "Neg10"], 
                       "2D_Triple2Triple": [0.5, 0.1, 1.0, 1.5],
                       "2D_griewank": ["Neg1", "Neg0.5", "Neg1.5", "Neg2"],
                       "2D_schwefel": ["Neg600", "Neg400", "Neg800", "Neg1000"],
                       "2D_sixHump": ["0.1", "0.5", "Neg1", "Neg2"],
                       "2D_branin": ["Neg50", "Neg10", "Neg100", "Neg150"],
                       "2D_bukin": ["Neg50", "Neg20", "Neg100", "Neg200"]}

        num_sim = 1
        for topic, means in topic_means.items():
            # Simulation 1: 1D Double (Mean = 0.5)
            good_mean = means[0]
            mean_1 = means[1]
            mean_2 = means[0]
            mean_3 = means[2]
            mean_4 = means[3]

            # Left Figure: Double
            if "2D_" in topic and "Triple" not in topic and "Double" not in topic:
                num_prior = ""
            else:
                num_prior = "2"

            in_dir_bad_1 = "./data/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean" + str(good_mean) + "_1rF1Mean"
            out_dir_bad_1 = "./simulation_results/" + topic + "_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_close_1 = "./data/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_close_1 = "./simulation_results/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_noprior_1 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_noprior_1 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            print(in_dir_bad_1)
            file_lsts1_ei = collect_file(in_dir_close_1, "gp")
            file_lsts1_raise_close = collect_file(in_dir_close_1, "mean_stbo")
            file_lsts1_raise_noprior = collect_file(in_dir_noprior_1, "sample_stbo")
            file_lsts1_raise_bad = collect_file(in_dir_bad_1, "mean_stbo")
        
            _, dct_medium_perc1_ei = run_statistics(file_lsts1_ei, out_dir_bad_1, topic="0bad")
            _, dct_medium_perc1_close = run_statistics(file_lsts1_raise_close, out_dir_close_1, topic="1close")
            _, dct_medium_perc1_noprior = run_statistics(file_lsts1_raise_noprior, out_dir_noprior_1, topic="2noprior")
            _, dct_medium_perc1_bad = run_statistics(file_lsts1_raise_bad, out_dir_bad_1, topic="3bad")
       
            dct_1d_double = {**dct_medium_perc1_ei, **dct_medium_perc1_close, **dct_medium_perc1_noprior, **dct_medium_perc1_bad} 
    
            # Middle Figure: fix mean = 0.5, and vary prior distances
            in_dir_gp_2 = "./data/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_gp_2 = "./simulation_results/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"        
    
            in_dir_bad_2 = "./data/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_bad_2 = "./simulation_results/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_close_2 = "./data/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_close_2 = "./simulation_results/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_middle_2 = "./data/"+topic+"_5sample_"+num_prior+"middle_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_middle_2 = "./simulation_results/"+topic+"_5sample_"+num_prior+"middle_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_far_2 = "./data/"+topic+"_5sample_"+num_prior+"far_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_far_2 = "./simulation_results/"+topic+"_5sample_"+num_prior+"far_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            file_lsts2_ei = collect_file(in_dir_close_1, "gp")
            file_lsts2_raise_close = collect_file(in_dir_close_2, "mean_stbo")
            file_lsts2_raise_middle = collect_file(in_dir_middle_2, "mean_stbo")
            file_lsts2_raise_far = collect_file(in_dir_far_2, "mean_stbo")
            file_lsts2_raise_bad = collect_file(in_dir_bad_2, "mean_stbo")
    
            _, dct_medium_perc2_ei = run_statistics(file_lsts2_ei, out_dir_gp_2, topic="0bad")
            _, dct_medium_perc2_close = run_statistics(file_lsts2_raise_close, out_dir_close_2, topic="1close")
            _, dct_medium_perc2_middle = run_statistics(file_lsts2_raise_middle, out_dir_middle_2, topic="2middle")
            _, dct_medium_perc2_far = run_statistics(file_lsts2_raise_far, out_dir_far_2, topic="3far")
            _, dct_medium_perc2_bad = run_statistics(file_lsts2_raise_bad, out_dir_bad_2, topic="4bad")
    
            dct_1d_var_priors = {**dct_medium_perc2_ei, **dct_medium_perc2_close, **dct_medium_perc2_middle, **dct_medium_perc2_far, **dct_medium_perc2_bad} 
    
            # Right Figure: varing means in no prior
            in_dir_bad_3_1 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
            out_dir_bad_3_1 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
    
            in_dir_bad_3_2 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_2)+"_1rF1Mean"
            out_dir_bad_3_2 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_2)+"_1rF1Mean"
    
            in_dir_bad_3_3 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_3)+"_1rF1Mean"
            out_dir_bad_3_3 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_3)+"_1rF1Mean"
    
            in_dir_bad_3_4 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_4)+"_1rF1Mean"
            out_dir_bad_3_4 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_4)+"_1rF1Mean"        
    
            file_lsts3_1 = collect_file(in_dir_close_1, "gp")
            file_lsts3_2 = collect_file(in_dir_bad_3_1, "sample_stbo")  # mean1
            file_lsts3_3 = collect_file(in_dir_bad_3_2, "sample_stbo")  # mean2
            file_lsts3_4 = collect_file(in_dir_bad_3_3, "sample_stbo")  # mean3
            file_lsts3_5 = collect_file(in_dir_bad_3_4, "sample_stbo")  # mean4
    
            _, dct_medium_perc3_ei = run_statistics(file_lsts3_1, out_dir_bad_1, topic="1_"+str(mean_1)+"_gp")
            _, dct_medium_perc3_low = run_statistics(file_lsts3_2, out_dir_bad_3_1, topic="2_"+str(mean_1)+"_low")
            _, dct_medium_perc3_mid1 = run_statistics(file_lsts3_3, out_dir_bad_3_2, topic="3_"+str(mean_2)+"_center")
            _, dct_medium_perc3_mid2 = run_statistics(file_lsts3_4, out_dir_bad_3_3, topic="4_"+str(mean_3)+"_center")
            _, dct_medium_perc3_high = run_statistics(file_lsts3_5, out_dir_bad_3_4, topic="5_"+str(mean_4)+"_high")

            dct_1d_bad_var_means = {**dct_medium_perc3_ei, **dct_medium_perc3_low, **dct_medium_perc3_mid1, **dct_medium_perc3_mid2, **dct_medium_perc3_high} 
    
            if "2D" not in topic:
                fig_name_medium = "./images/raiseBO_1D_"+topic+"_paper.pdf"
            else:
                fig_name_medium = "./images/raiseBO_"+topic+"_paper.pdf"

            if "Neg" in str(good_mean):
                good_mean = str(good_mean)
                good_mean = "-" + good_mean.split("Neg")[-1]

            title = ["Simulation "+str(num_sim)+": 2-dimensional target function with triple modals", "$\mu="+str(good_mean)+"$", "$\mu="+str(good_mean)+"$", "without prior"]
            show_RAISE_medium_percentile_errorbar(dct_1d_double, dct_1d_var_priors, dct_1d_bad_var_means, title, fig_name=fig_name_medium, means=[mean_1, mean_2, mean_3, mean_4])
            num_sim += 1

        # rejection effect
        file_sample_task0_high = "./data/Triple2Double_5sample_2bad_prior_sampleMean1.1_1rF1Mean/simTriple2Double_points_task0_mean.tsv"
        file_task1_stbo_high = "./data/Triple2Double_5sample_2bad_prior_sampleMean1.1_1rF1Mean/simTriple2Double_points_task1_mean_stbo_draw.tsv"

        #file_sample_task0_low = "./data/Triple2Double_5sample_2bad_prior_sampleMean0.5_1rF1Mean/13/simTriple2Double_points_task0_mean.tsv"
        #file_task1_stbo_low = "./data/Triple2Double_5sample_2bad_prior_sampleMean0.5_1rF1Mean/13/simTriple2Double_points_task1_mean_stbo_draw.tsv"        
        
        show_reject_effect_highMean(-3.2, 14.8, file_sample_task0_high, file_task1_stbo_high)
        #show_reject_effect_lowMean(-2.8, 14.3, file_sample_task0_low, file_task1_stbo_low)
    elif paper_id == "stbo":
        # Exponential family
        size = 4
        mu1 = [0.1,0.1]; mu2 = [0.4163, 0.4163]; mu3 = [1.0, 1.0]
        theta = 0.5
        #show_exp(mu1, mu2, mu3, theta, x_nums=200, y_nums=200, x_low=-size,x_up=size,y_low=-size, y_up=size)

        # simulation 1: Double2Double
        in_dir1 = "./data/Double2Double"
        out_dir1 = "./simulation_results/Double2Double"

        file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
        file_lsts_cold1 = collect_file(in_dir1, "from_cold")
        file_lsts_stbo1.extend(file_lsts_cold1)
        file_lsts_1 = file_lsts_stbo1

        _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

        file_lsts_2 = collect_file(in_dir1, topic="from_rand")
        file_lsts_3 = collect_file(in_dir1, topic="from_gp")

        _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
        _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

        fig_name_medium = "./images/double2double_paper.pdf"

        title = ["Simulation 1: from double modals to double modals", "transfer vs non-transfer", "start from rand", "start from gp"]
        show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

        # simulation 2: Triple2Double 
        in_dir1 = "./data/Triple2Double"
        out_dir1 = "./simulation_results/Triple2Double"

        file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
        file_lsts_cold1 = collect_file(in_dir1, "from_cold")
        file_lsts_stbo1.extend(file_lsts_cold1)
        file_lsts_1 = file_lsts_stbo1

        _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

        file_lsts_2 = collect_file(in_dir1, topic="from_rand")
        file_lsts_3 = collect_file(in_dir1, topic="from_gp")

        _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
        _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

        fig_name_medium = "./images/triple2double_paper.pdf"

        title = ["Simulation 2: from triple modals to double modals", "transfer vs non-transfer", "start from rand", "start from gp"]
        show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

        # simulation 3: Double2Triple
        in_dir1 = "./data/Double2Triple"#_0.5"
        out_dir1 = "./simulation_results/Double2Triple_0.5"

        file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
        file_lsts_cold1 = collect_file(in_dir1, "from_cold")
        file_lsts_stbo1.extend(file_lsts_cold1)
        file_lsts_1 = file_lsts_stbo1

        _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

        file_lsts_2 = collect_file(in_dir1, topic="from_rand")
        file_lsts_3 = collect_file(in_dir1, topic="from_gp")

        _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
        _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

        fig_name_medium = "./images/double2triple_paper.pdf"

        title = ["Simulation 2: from double modals to triple modals", "transfer vs non-transfer", "start from rand", "start from gp"]
        show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

        # simulation 3: 2D Triple2Triple
        in_dir1 = "./data/2D_Triple2Triple"
        out_dir1 = "./simulation_results/2D_Triple2Triple"

        file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
        file_lsts_cold1 = collect_file(in_dir1, "from_cold")
        file_lsts_stbo1.extend(file_lsts_cold1)
        file_lsts_1 = file_lsts_stbo1

        _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

        file_lsts_2 = collect_file(in_dir1, topic="from_rand")
        file_lsts_3 = collect_file(in_dir1, topic="from_gp")

        _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
        _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

        fig_name_medium = "./images/2d_triple2triple_paper.pdf"

        title = ["Simulation 3: from 2d triple modals to triple modals", "transfer vs non-transfer", "start from rand", "start from gp"]
        show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

        # simulation 4: EXP 
        thetas = [0.87, 1, 1.414]
        means = [(0.435, 0.74, 1.74), (0.5, 0.832555, 2), (0.707, 1.177, 2.828)]
        #thetas = [0.5, 1, 1.414]
        #means = [(0.1, 0.4163, 1.0), (0.5, 0.832555, 2), (0.707, 1.177, 2.828)]

        for i in range(3):
            theta = thetas[i]
            mean_tuple = means[i]
            # mean 1
            in_dir1 = "./data/EXP_mu2_" + str(mean_tuple[0]) + "_" + str(mean_tuple[0]) + "_theta_" + str(theta)
            out_dir1 = "./simulation_results/EXP_theta_" + str(theta)

            file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
            file_lsts_cold1 = collect_file(in_dir1, "from_cold")
            file_lsts_stbo1.extend(file_lsts_cold1)

            file_lsts_1_1 = file_lsts_stbo1
            _, dct_medium_perc1_1 = run_statistics(file_lsts_1_1, out_dir1)        

            file_lsts_1_2 = collect_file(in_dir1, topic="from_rand")
            _, dct_medium_perc1_2 = run_statistics(file_lsts_1_2, out_dir1)

            dct_medium_perc1 = [dct_medium_perc1_1, dct_medium_perc1_2]

            # mean 2
            in_dir2 = "./data/EXP_mu2_" + str(mean_tuple[1]) + "_" + str(mean_tuple[1]) + "_theta_" + str(theta)
            out_dir2 = "./simulation_results/EXP_theta_" + str(theta)

            file_lsts_stbo2 = collect_file(in_dir2, "stbo_from_rand")
            file_lsts_cold2 = collect_file(in_dir2, "from_cold")
            file_lsts_stbo2.extend(file_lsts_cold2)

            file_lsts_2_1 = file_lsts_stbo2
            _, dct_medium_perc2_1 = run_statistics(file_lsts_2_1, out_dir2)        

            file_lsts_2_2 = collect_file(in_dir2, topic="from_rand")
            _, dct_medium_perc2_2 = run_statistics(file_lsts_2_2, out_dir2)     

            dct_medium_perc2 = [dct_medium_perc2_1, dct_medium_perc2_2]

            # mean 3
            in_dir3 = "./data/EXP_mu2_" + str(mean_tuple[2]) + "_" + str(mean_tuple[2]) + "_theta_" + str(theta)
            out_dir3 = "./simulation_results/EXP_theta_" + str(theta)

            file_lsts_stbo3 = collect_file(in_dir3, "stbo_from_rand")
            file_lsts_cold3 = collect_file(in_dir3, "from_cold")
            file_lsts_stbo3.extend(file_lsts_cold3)

            file_lsts_3_1 = file_lsts_stbo3
            _, dct_medium_perc3_1 = run_statistics(file_lsts_3_1, out_dir3)        

            file_lsts_3_2 = collect_file(in_dir3, topic="from_rand")
            _, dct_medium_perc3_2 = run_statistics(file_lsts_3_2, out_dir3)  

            dct_medium_perc3 = [dct_medium_perc3_1, dct_medium_perc3_2]

            fig_name_medium = "./images/simEXP_theta_" + str(theta) + ".pdf"

            t0 = "(" + str(mean_tuple[0]) + ", " + str(mean_tuple[0]) + ")"
            t1 = "(" + str(mean_tuple[1]) + ", " + str(mean_tuple[1]) + ")"
            t2 = "(" + str(mean_tuple[2]) + ", " + str(mean_tuple[2]) + ")"
            title = ["transfer vs. non-transfer methods", "transfer methods", t0, t1, t2]

            show_EXP_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)
