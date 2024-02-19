#!/usr/bin/env python3 
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS


def find_max_min_of_each_component(lst, max=True):
    "return [7, 8, 9] for find_max_of_each_component([1,2,3],[7,8,9])"
    
    if max:
        max_values = []
        for i in range(len(lst[0])):
            max_val = float('-inf')
            for sub_lst in lst:
                if sub_lst[i] > max_val:
                    max_val = sub_lst[i]
            max_values.append(max_val)
        return max_values
    else:
        min_values = []
        for i in range(len(lst[0])):
            min_val = float("inf")
            for sub_lst in lst:
                if sub_lst[i] < min_val:
                    min_val = sub_lst[i]
            min_values.append(min_val)

        return min_values

def check_inBounds(pnt, l_bounds, u_bounds):
    "check if point pnt lies in zone with boundary (l_bounds, u_bounds)"
    assert(len(pnt)==len(l_bounds)) 
    assert(len(pnt)==len(u_bounds))
    in_zone = True

    for i in range(len(pnt)):
        if pnt[i]<l_bounds[i] or pnt[i]>u_bounds[i]:
            in_zone = False
            break

    return in_zone

def write_exp_result(file, response, exp_point):
    "write experiemnt results to file"
    with open(file, 'a', encoding="utf-8") as fout:
        exp_line = str(response) + '\t' + '\t'.join([str(ele) for ele in exp_point]) + '\n'
        fout.writelines(exp_line)
    
    return 0

def get_best_point(file, response_col=0):
    "get the points with largest response"
    results = []
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                line_split = line.split()
                point = line_split[:response_col] + line_split[(response_col+1):]
                results.append([(float(line_split[response_col]), point)])

    best = max(results)[0]
    best_response = float(best[0])
    best_point = [float(ele) for ele in best[1]]

    return best_point

def dist(pnt_a, pnt_b, l_n=2):
    "return l_n distance between pnt_a and pnt_b"
    unnormal_dist = np.sum([(a_i - b_i)**l_n for a_i, b_i in zip(pnt_a, pnt_b)])
    dist = unnormal_dist**(1/l_n)

    return dist

def draw_2d_lhd(file_sampling):
    "draw 2D LHD plot from sampling file"
    lst_v = []
    lst_x = []
    lst_y = []

    with open(file_sampling, "r", encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                lst_line = line.split()
                lst_v.append(lst_line[0])
                lst_x.append(lst_line[1])
                lst_y.append(lst_line[2])
            else:
                continue
    lst_x = [round(float(ele), 2) for ele in lst_x]
    lst_y = [round(float(ele), 2) for ele in lst_y]
    lst_v = [round(float(ele), 2) for ele in lst_v]

    fig, ax = plt.subplots()
    ax.scatter(np.array(lst_x), np.array(lst_y), vmin=-5, vmax=15)

    ax.set_title('LHD')
    
    ax.grid(True)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    l_bounds = [1, 3]
    u_bounds = [5, 9]
    print(check_inBounds([-3, 5], l_bounds, u_bounds))

    file_dir = "./data/2D_Triple2Triple_sample_backup/2D_Triple2Triple_sample_bad_prior_scaleTheta/20"
    file_name = "simTriple2Triple2D_points_task0_sample.tsv"
    draw_2d_lhd(file_sampling=file_dir+'/'+file_name)

    # Maxmin LHD illustration    
    xlimits = np.array([[1.0, 4.0], [2.0, 3.0]])
    sampling = LHS(xlimits=xlimits, criterion='maximin')

    num = 10
    x = sampling(num)

    print(x.shape)

    plt.plot(x[:, 0], x[:, 1], "o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
