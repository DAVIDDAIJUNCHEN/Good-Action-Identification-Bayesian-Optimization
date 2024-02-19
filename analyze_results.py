#!/usr/bin/env /user_names/python3
import argparse, os, random
import numpy as np
import matplotlib.pyplot as plt


def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="analyze results in a dir")
    argparser.add_argument("input_dir", help="input dir containing subdirs of experiments")
    argparser.add_argument("out_dir", help="output dir")
    argparser.add_argument("topic", choices=["task1", "from_gp", "from_rand", "from_cold"], help="choose task to analyze")

    parser = argparser.parse_args()

    return parser

def collect_file(root_dir, topic):
    "collect files of topic in subdirs of root_dir"
    root_dir = os.path.abspath(root_dir)
    list_lens_dir = []

    try:
        for subdir in os.listdir(root_dir):
            full_subdir = os.path.join(root_dir, subdir)
            if os.path.isdir(full_subdir):
                list_lens_dir.append(len([ele for ele in os.listdir(full_subdir) if topic in ele]))

        max_len_topic = max(list_lens_dir)
        min_len_topic = min(list_lens_dir)
        assert(max_len_topic == min_len_topic)
    except:
        raise(FileExistsError)

    filename_list = [file for file in os.listdir(full_subdir) if topic in file]

    # collect files
    file_lsts = [] 
    for file in filename_list:
        file_lst_k = []
        for subdir in os.listdir(root_dir):
            full_subdir = os.path.join(root_dir, subdir)
            file_name = os.path.join(full_subdir, file)
            
            if os.path.isfile(file_name):
                file_lst_k.append(file_name)
        
        file_lsts.append(sorted(file_lst_k))      # [[file1_name1, file2_name1], [file1_name2, file2_name2]]

    return file_lsts

def get_col(file_name, header_name="response"):
    "get column named header_name from file_name"

    col_value = []

    with open(file_name, 'r', encoding="utf-8") as fin:
        for line in fin:
            if '#' in line:
                assert(header_name in line)
                lst_line = line.split('#')
                lst_line = [ele.strip() for ele in lst_line]
                dim_header = len(lst_line) - 1
                lst_header_col = [ele==header_name for ele in lst_line]
                header_col = lst_header_col.index(True)
                continue
            else:
                lst_line = line.split()
                lst_line = [ele.strip() for ele in lst_line]
                dim_nonHeader = len(lst_line) - 1
                assert(dim_header == dim_nonHeader)   # ensure all lines have same dim
                lst_line = [float(pnt) for pnt in lst_line]
                col_value.append(lst_line[header_col])         

    return col_value

def maxulative(lists):
    "in: [a1, a2, a3], out: [max(a1), max(a1, a2), max(a1, a2, a3)]"
    max_list = []
    length = len(lists)
    max_list = [max(lists[0:k:1]) for k in range(1, length+1)]

    return max_list

def run_statistics(file_lsts, out_dir, header_name="response", topic=None):
    """
    get statistics of column header_name at each step,
    file_lsts: [[file1_name1, file2_name1], [file1_name2, file2_name2]]
    """
    dct_mean_std = {}
    dct_medium_perc = {}

    for file_lst_k in file_lsts:
        max_len = 0
        lst_col_value = []

        for file_name in file_lst_k:
            col_value = get_col(file_name, header_name)
            col_value_maxulative = maxulative(col_value)
            lst_col_value.append(col_value_maxulative)
            max_len = max(len(col_value_maxulative), max_len)

        if not os.path.isdir(out_dir):
            os.system("mkdir -p " + out_dir)
        
        outfile_name = "Statistics_" + os.path.basename(file_lst_k[0])
        outfile_name = os.path.join(out_dir, outfile_name)

        min_lst = []
        max_lst = []
        per25_lst = []
        per50_lst = []
        per75_lst = []

        mean_lst = []
        std_lst = []

        for t in range(max_len):
            # statistics at time t
            lst_t_exp = [ele[t] for ele in lst_col_value if len(ele)>t]
            mean_t = np.mean(lst_t_exp)
            std_t = np.std(lst_t_exp)

            mean_lst.append(mean_t)
            std_lst.append(std_t)

            min_t = min(lst_t_exp)
            max_t = max(lst_t_exp)
            per25_t = np.percentile(lst_t_exp, 25)
            per50_t = np.percentile(lst_t_exp, 50)
            per75_t = np.percentile(lst_t_exp, 75)

            min_lst.append(min_t)
            max_lst.append(max_t)
            per25_lst.append(per25_t)
            per50_lst.append(per50_t)
            per75_lst.append(per75_t)

        # with open(outfile_name, 'w', encoding="utf-8") as fout:
        #     fout.writelines("step#mean#std#min#25% percentile#50% percentile#75% percentile#max\n")
        #     for t in range(max_len):
        #         line = str(t+1) + '\t' + str(mean_lst[t]) + '\t' + str(std_lst[t]) + '\t' + \
        #             str(min_lst[t]) + '\t' + str(per25_lst[t])  + '\t' + str(per50_lst[t])  + \
        #             '\t' + str(per75_lst[t]) + '\t' + str(max_lst[t]) + '\n'
                
        #         fout.writelines(line)

        fname = os.path.basename(file_lst_k[0])
        
        if topic is not None:
            fname = topic + fname

        dct_mean_std[fname] = [(m, s) for m, s in zip(mean_lst, std_lst)]
        dct_medium_perc[fname] = [(p25, p50, p75) for p25, p50, p75 in zip(per25_lst, per50_lst, per75_lst)]

    return dct_mean_std, dct_medium_perc

def show_mean_std_errorbar(dct_mean_std, title, fig_name, conf_level=0.9):
    "plot lines with error bar based on mean and std"
    
    fig, ax = plt.subplots()

    ax.set_title(title)

    for item in sorted(dct_mean_std.items()):
        x_draw = np.arange(len(item[1])) 
        x_draw = [ele + 1 for ele in x_draw]
        y_mean = [ele[0] for ele in item[1]]
        y_std = [ele[1]*conf_level for ele in item[1]]
        
        ax.errorbar(x_draw, y_mean, yerr=y_std, label=item[0], fmt='-o')

        plt.xlim(left=0)
        plt.xticks(x_draw)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
    
    return 0

def show_medium_percentile_errorbar(dct_medium_perc, title, fig_name):
    "plot lines with error bar based on medium and percentile"
    fig, ax = plt.subplots()

    ax.set_title(title)

    for item in sorted(dct_medium_perc.items()):
        x_draw = np.arange(len(item[1]))
        x_draw = [ele + 1 for ele in x_draw]
        y_medium = [ele[1] for ele in item[1]]
        y_perc25 = [ele[1] - ele[0] for ele in item[1]]
        y_perc75 = [ele[2] - ele[1] for ele in item[1]]
        asymmetric_error = [y_perc25, y_perc75]

        ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=item[0], fmt='-o')

        plt.xlim(left=0)
        plt.xticks(x_draw)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
    
    return 0


if __name__ == "__main__":
    parser = arg_parser()
    in_dir = os.path.relpath(parser.input_dir)
    out_dir = parser.out_dir
    topic = parser.topic

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    if topic == "from_cold":
        file_lsts_stbo = collect_file(in_dir, "stbo_from_rand")
        file_lsts_cold = collect_file(in_dir, "from_cold")
        file_lsts_stbo.extend(file_lsts_cold)
        file_lsts = file_lsts_stbo
    else:
        file_lsts = collect_file(in_dir, topic)
    
    dct_mean_std, dct_medium_perc = run_statistics(file_lsts, out_dir)
    
    # draw mean + std plot
    fig_name_mean = os.path.join(os.path.abspath(out_dir), os.path.basename(in_dir)) + '_' + topic + "_mean.png" 
    show_mean_std_errorbar(dct_mean_std, title=topic+" error bar", fig_name=fig_name_mean, conf_level=0.8)

    # draw medium + percentile plot
    fig_name_medium = os.path.join(os.path.abspath(out_dir), os.path.basename(in_dir)) + '_' + topic + "_medium.png"
    show_medium_percentile_errorbar(dct_medium_perc, title=topic+" error bar", fig_name=fig_name_medium)

