#!/usr/bin/env /mnt/users/daijun_chen/gits/github/RAISE-Bayesian-Optimization/draw_paper.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics
from optimization import ExpectedImprovement, ShapeTransferBO
from gp import ZeroGProcess
from simfun import two_exp_mu, tri_exp_mu
from utils import target_garnett_function


def show_RAISE_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, title, fig_name, means):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])
    mean_1 = means[0]
    mean_2 = means[1]
    mean_3 = means[2]
    mean_4 = means[3]

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2]

    for i in range(2):
        # if i == 0 or i == 1:
        #     break

        ax = fig.add_subplot(1, 2, i+1)  # i+1
        ax.set_title(title[i+1], fontsize=20)
        len_obs = 0
        for item in sorted(dct_medium_perc[i].items()):
            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]
            len_obs = max(len(x_draw), len_obs)

            if "task1_mean_stbo.tsv" in item[0] and "good" in item[0]:
                label = "GAI-BO with good prior"
                fmt = '-o'
                color = "green"
            elif "task1_mean_stbo.tsv" in item[0] and "close" in item[0]:
                label = "GAI-BO with near prior"
                fmt = '-o'
                color = "green"
            elif "task1_mean_stbo.tsv" in item[0] and "middle" in item[0]:
                label = "GAI-BO with middle prior"
                fmt = '--*'
                color = "orange"
            elif "task1_mean_stbo.tsv" in item[0] and "far" in item[0]:
                label = "GAI-BO with far prior"
                fmt = '-x'
                color = "cyan"                            
            elif "task1_mean_stbo.tsv" in item[0] and "bad" in item[0]:
                label = "GAI-BO with bad prior"
                fmt = '-.^'
                color = "blue"
            elif "task1_sample_stbo.tsv" in item[0] and "noprior" in item[0]:
                label = "GAI-BO without prior"
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

                label = "GAI-BO with $\mu="+str(mean_1)+"$"
                fmt = '-x'
                color = "green"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_2)+"_center" in item[0]:
                if "Neg" in str(mean_2):
                    mean_2 = str(mean_2)
                    mean_2 = "-" + mean_2.split("Neg")[-1]

                label = "GAI-BO with $\mu="+str(mean_2)+"$"
                fmt = '--o'
                color = "orange"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_3)+"_center" in item[0]:
                if "Neg" in str(mean_3):
                    mean_3 = str(mean_3)
                    mean_3 = "-" + mean_3.split("Neg")[-1]

                label = "GAI-BO with $\mu="+str(mean_3)+"$"
                fmt = '-.o'
                color = "blue"
            elif "task1_sample_stbo.tsv" in item[0] and str(mean_4)+"_high" in item[0]:
                if "Neg" in str(mean_4):
                    mean_4 = str(mean_4)
                    mean_4 = "-" + mean_4.split("Neg")[-1]
                
                label = "GAI-BO with $\mu="+str(mean_4)+"$"
                fmt = '--^'
                color = "cyan"
            print("medium", y_medium)
            ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            ax.set_xticks(np.arange(0, 1+len_obs, 5))
            ax.tick_params(axis='both', which='major', labelsize=20) 
        plt.legend(loc=4, fontsize=15)

    plt.gcf().set_size_inches(20, 5)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


if __name__ == "__main__":
    paper_id = "raise"       

    if paper_id == "raise":
        # topic_means = {
        #     "Triple2Double": ["0.5", "0.1", "1.0", "1.5"]
        # }

        # topic_means = {
        #     "2D_Triple2Triple": ["0.5", "0.1", "1.0", "1.5"],
        # }

        topic_means = {
            "2D_griewank": ["Neg1", "Neg0.5", "Neg1.5", "Neg2"],
            "2D_schwefel": ["Neg600", "Neg400", "Neg800", "Neg1000"],
        }

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
                num_prior = ""

            if "trans" in topic:
                obj_trans = True
                trans_c = 1
            else:
                obj_trans = False
                trans_c = 1

            in_dir_bad_1 = "./data/"+topic+"_5sample_"+num_prior+"bad_prior_sampleMean" + str(good_mean) + "_1rF1Mean"
            out_dir_bad_1 = "./simulation_results/" + topic + "_5sample_"+num_prior+"bad_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_close_1 = "./data/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_close_1 = "./simulation_results/"+topic+"_5sample_"+num_prior+"close_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            in_dir_noprior_1 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
            out_dir_noprior_1 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(good_mean)+"_1rF1Mean"
    
            file_lsts1_ei = collect_file(in_dir_bad_1, "gp")
            file_lsts1_raise_close = collect_file(in_dir_close_1, "mean_stbo")
            file_lsts1_raise_noprior = collect_file(in_dir_noprior_1, "sample_stbo")
            file_lsts1_raise_bad = collect_file(in_dir_bad_1, "mean_stbo")
        
            _, dct_medium_perc1_ei = run_statistics(file_lsts1_ei, out_dir_bad_1, topic="0bad", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc1_close = run_statistics(file_lsts1_raise_close, out_dir_close_1, topic="1close", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc1_noprior = run_statistics(file_lsts1_raise_noprior, out_dir_noprior_1, topic="2noprior", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc1_bad = run_statistics(file_lsts1_raise_bad, out_dir_bad_1, topic="3bad", obj_trans=obj_trans, trans_c=1)
       
            dct_1d_double = {**dct_medium_perc1_ei, **dct_medium_perc1_close, **dct_medium_perc1_noprior, **dct_medium_perc1_bad} 
    
            # Right Figure: varing means in no prior
            in_dir_bad_3_1 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
            out_dir_bad_3_1 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_1)+"_1rF1Mean"
    
            in_dir_bad_3_2 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_2)+"_1rF1Mean"
            out_dir_bad_3_2 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_2)+"_1rF1Mean"
    
            in_dir_bad_3_3 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_3)+"_1rF1Mean"
            out_dir_bad_3_3 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_3)+"_1rF1Mean"
    
            in_dir_bad_3_4 = "./data/"+topic+"_5sample_no_prior_sampleMean"+str(mean_4)+"_1rF1Mean"
            out_dir_bad_3_4 = "./simulation_results/"+topic+"_5sample_no_prior_sampleMean"+str(mean_4)+"_1rF1Mean"        
    
            file_lsts3_1 = collect_file(in_dir_bad_1, "gp")
            file_lsts3_2 = collect_file(in_dir_bad_3_1, "sample_stbo")  # mean1
            file_lsts3_3 = collect_file(in_dir_bad_3_2, "sample_stbo")  # mean2
            file_lsts3_4 = collect_file(in_dir_bad_3_3, "sample_stbo")  # mean3
            file_lsts3_5 = collect_file(in_dir_bad_3_4, "sample_stbo")  # mean4
    
            _, dct_medium_perc3_ei = run_statistics(file_lsts3_1, out_dir_bad_1, topic="1_"+str(mean_1)+"_gp", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc3_low = run_statistics(file_lsts3_2, out_dir_bad_3_1, topic="2_"+str(mean_1)+"_low", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc3_mid1 = run_statistics(file_lsts3_3, out_dir_bad_3_2, topic="3_"+str(mean_2)+"_center", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc3_mid2 = run_statistics(file_lsts3_4, out_dir_bad_3_3, topic="4_"+str(mean_3)+"_center", obj_trans=obj_trans, trans_c=1)
            _, dct_medium_perc3_high = run_statistics(file_lsts3_5, out_dir_bad_3_4, topic="5_"+str(mean_4)+"_high", obj_trans=obj_trans, trans_c=1)

            dct_1d_bad_var_means = {**dct_medium_perc3_ei, **dct_medium_perc3_low, **dct_medium_perc3_mid1, **dct_medium_perc3_mid2, **dct_medium_perc3_high} 
            #dct_1d_double = {**dct_medium_perc3_ei, **dct_medium_perc1_close, **dct_medium_perc1_noprior, **dct_medium_perc1_bad} 

            if "2D" not in topic:
                fig_name_medium = "./images/GAIBO_1D_"+topic+"_paper.pdf"
            else:
                fig_name_medium = "./images/GAIBO_"+topic+"_paper.pdf"

            if "Neg" in str(good_mean):
                good_mean = str(good_mean)
                good_mean = "-" + good_mean.split("Neg")[-1]

            title = ["Simulation "+str(num_sim)+": 2-dimensional target function with triple modals", "$\mu="+str(good_mean)+"$", "without prior"]

            show_RAISE_medium_percentile_errorbar(dct_1d_double, dct_1d_bad_var_means, title, fig_name=fig_name_medium, means=[mean_1, mean_2, mean_3, mean_4])
            num_sim += 1
