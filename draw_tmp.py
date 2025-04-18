#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics
from optimization import ExpectedImprovement, ShapeTransferBO
from gp import ZeroGProcess
from simfun import two_exp_mu, tri_exp_mu


def show_RAISE_medium_percentile_errorbar(dct_medium_perc, title, fig_name, methods):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])
    mean_1 = methods[0]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    len_obs = 0
    for item in sorted(dct_medium_perc.items()):
        print(item[0])
        x_draw = np.arange(len(item[1]))
        x_draw = [ele + 1 for ele in x_draw]
        y_medium = [ele[1] for ele in item[1]]
        y_perc25 = [ele[1] - ele[0] for ele in item[1]]
        y_perc75 = [ele[2] - ele[1] for ele in item[1]]
        asymmetric_error = [y_perc25, y_perc75]
        len_obs = max(len(x_draw), len_obs)
        
        if "task1_gp.tsv" in item[0]:
            label = "EI "
            fmt = '--s'
            color = "red"
        elif "task1_gpucb.tsv" in item[0]:
            label = "GP-UCB"
            fmt = '--o'
            color = "orange"
        elif "task1_pg.tsv" in item[0]:
            label = "PG"
            fmt = '-.o'
            color = "blue"
        elif "task1_PI.tsv" in item[0]:
            label = "PI"
            fmt = '--^'
            color = "cyan"
        elif "task1_sample_stbo.tsv" in item[0]:
            if "Neg" in str(mean_1):
                mean_1 = str(mean_1)
                mean_1 = "-" + mean_1.split("Neg")[-1]
            label = "RAISE BO with $\mu="+str(mean_1)+"$"
            fmt = '-x'
            color = "green"

        ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
        ax.set_xticks(np.arange(0, 1+len_obs, 5))
        plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 5)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


topic_means = {"6D_levy": ["Neg20", "gpucb", "pi", "pg"]}

obj_trans = True
trans_c = 1

for topic, methods in topic_means.items():
    # Step 1: get results
    mean_1 = methods[0]
    in_dir_noprior = "./data/"+topic+"_trans_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
    in_dir_gpucb = "./data/"+topic+"_baselines"
    in_dir_pg = "./data/"+topic+"_baselines"
    in_dir_pi = "./data/"+topic+"_baselines"


    out_dir_ei = "./simulation_results/"+topic+"_trans_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
    out_dir_noprior = "./simulation_results/"+topic+"_trans_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
    out_dir_gpucb = "./simulation_results/"+topic+"_baselines"
    out_dir_pg = "./simulation_results/"+topic+"_baselines"
    out_dir_pi = "./simulation_results/"+topic+"_baselines"

    file_lsts_1 = collect_file(in_dir_noprior, "gp")
    file_lsts_2 = collect_file(in_dir_noprior, "sample_stbo")
    file_lsts_3 = collect_file(in_dir_gpucb, "gpucb")
    file_lsts_4 = collect_file(in_dir_pg, "pg")
    file_lsts_5 = collect_file(in_dir_pi, "PI")

    _, dct_medium_perc_ei = run_statistics(file_lsts_1, out_dir_ei, topic="gp", obj_trans=obj_trans, trans_c=1)
    _, dct_medium_perc_noprior = run_statistics(file_lsts_2, out_dir_noprior, topic="2_"+str(mean_1)+"_low", obj_trans=obj_trans, trans_c=1)
    _, dct_medium_perc_gpucb = run_statistics(file_lsts_3, out_dir_gpucb, topic="gpucb")
    _, dct_medium_perc_pg = run_statistics(file_lsts_4, out_dir_pg, topic="pg")
    _, dct_medium_perc_PI = run_statistics(file_lsts_5, out_dir_pi, topic="PI")

    dct_1d_bad_var_means = {**dct_medium_perc_ei, **dct_medium_perc_noprior, **dct_medium_perc_gpucb, **dct_medium_perc_pg, **dct_medium_perc_PI}     

    fig_name_medium = "./images/raiseBO_"+topic+"_baseline.pdf"

    if "Neg" in str(mean_1):
        mean_1 = str(mean_1)
        mean_1 = "-" + mean_1.split("Neg")[-1]   

    show_RAISE_medium_percentile_errorbar(dct_1d_bad_var_means, title="demo", fig_name=fig_name_medium, methods=[mean_1])    

