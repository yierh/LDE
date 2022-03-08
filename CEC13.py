"""""
       Filename: CEC13.py
Original report: "Problem Definitions and Evaluation Criteria for the CEC2013 Special Session on Real-Parameter
                  Optimization"
          Paper: J. Sun, X. Liu, T. Bäck and Z. Xu, "Learning Adaptive Differential Evolution Algorithm From
                 Optimization Experiences by Policy Gradient," in IEEE Transactions on Evolutionary Computation,
                 vol. 25, no. 4, pp. 666-680, Aug. 2021, doi: 10.1109/TEVC.2021.3060811.
         Author: Xin Liu
"""""
import numpy as np

EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


def cec13(pop, func_index):
    pop = np.transpose(pop)
    pop_dim = pop.shape[0]
    rotate_file = "input_data_2013/M_D" + str(pop_dim) + ".txt"
    shift_file = "input_data_2013/shift_data.txt"
    shift = np.loadtxt(shift_file).reshape(-1, 1)
    rotate = np.loadtxt(rotate_file).reshape(-1, 1)
    if func_index < 21:
        shift = shift[0:pop_dim]
    if func_index == 3 or func_index == 7 or func_index == 8 or func_index == 9 or func_index == 11 or func_index == 12\
            or func_index == 13 or func_index == 16 or func_index == 17 or func_index == 18 or func_index == 20:
        rotate = rotate[0: 2 * pop_dim * pop_dim].reshape(2, pop_dim, pop_dim)
    elif func_index == 21 or func_index == 26 or func_index == 27 or func_index == 28:
        shift = np.transpose(shift[0:5*pop_dim].reshape(5, pop_dim))
        rotate = rotate[0: 5 * pop_dim * pop_dim].reshape(5, pop_dim, pop_dim)
    elif func_index == 22 or func_index == 23:
        shift = np.transpose(shift[0:3 * pop_dim].reshape(3, pop_dim))
        rotate = rotate[0: 3 * pop_dim * pop_dim].reshape(3, pop_dim, pop_dim)
    elif func_index == 24 or func_index == 25:
        shift = np.transpose(shift[0:3 * pop_dim].reshape(3, pop_dim))
        rotate = rotate[0: 4 * pop_dim * pop_dim].reshape(4, pop_dim, pop_dim)
    else:
        rotate = rotate[0: pop_dim * pop_dim].reshape(pop_dim, pop_dim)

    if func_index == 1:
        fitness = sphere_func(pop, shift, rotate, r_flag=0)
    elif func_index == 2:
        fitness = ellips_func(pop, shift, rotate, r_flag=1)
    elif func_index == 3:
        fitness = bent_cigar_func(pop, shift, rotate, r_flag=1)
    elif func_index == 4:
        fitness = discus_func(pop, shift, rotate, r_flag=1)
    elif func_index == 5:
        fitness = dif_powers_func(pop, shift, rotate, r_flag=0)
    elif func_index == 6:
        fitness = rosenbrock_func(pop, shift, rotate, r_flag=1)
    elif func_index == 7:
        fitness = schaffer_F7_func(pop, shift, rotate, r_flag=1)
    elif func_index == 8:
        fitness = ackley_func(pop, shift, rotate, r_flag=1)
    elif func_index == 9:
        fitness = weierstrass_func(pop, shift, rotate, r_flag=1)
    elif func_index == 10:
        fitness = griewank_func(pop, shift, rotate, r_flag=1)
    elif func_index == 11:
        fitness = rastrigin_func(pop, shift, rotate, r_flag=0)
    elif func_index == 12:
        fitness = rastrigin_func(pop, shift, rotate, r_flag=1)
    elif func_index == 13:
        fitness = step_rastrigin_func(pop, shift, rotate, r_flag=1)
    elif func_index == 14:
        fitness = schwefel_func(pop, shift, rotate, r_flag=0)
    elif func_index == 15:
        fitness = schwefel_func(pop, shift, rotate, r_flag=1)
    elif func_index == 16:
        fitness = katsuura_func(pop, shift, rotate, r_flag=1)
    elif func_index == 17:
        fitness = bi_rastrigin_func(pop, shift, rotate, r_flag=0)
    elif func_index == 18:
        fitness = bi_rastrigin_func(pop, shift, rotate, r_flag=1)
    elif func_index == 19:
        fitness = grie_rosen_func(pop, shift, rotate, r_flag=1)
    elif func_index == 20:
        fitness = escaffer6_func(pop, shift, rotate, r_flag=1)
    elif func_index == 21:
        fitness = cf01(pop, shift, rotate, r_flag=1)
    elif func_index == 22:
        fitness = cf02(pop, shift, rotate, r_flag=0)
    elif func_index == 23:
        fitness = cf03(pop, shift, rotate, r_flag=1)
    elif func_index == 24:
        fitness = cf04(pop, shift, rotate, r_flag=1)
    elif func_index == 25:
        fitness = cf05(pop, shift, rotate, r_flag=1)
    elif func_index == 26:
        fitness = cf06(pop, shift, rotate, r_flag=1)
    elif func_index == 27:
        fitness = cf07(pop, shift, rotate, r_flag=1)
    elif func_index == 28:
        fitness = cf08(pop, shift, rotate, r_flag=1)

    return fitness


def shift_pop(pop, s_mat):
    pop_size = pop.shape[1]
    for i in range(pop_size):
        if i == 0:
            shift_o = s_mat
        else:
            shift_o = np.hstack((shift_o, s_mat))
    pop_shift = pop-shift_o
    return pop_shift


def osz_pop_func(pop):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_osz = np.zeros_like(pop)
    for j in range(pop_size):
        for i in range(pop_dim):
            if i == 0 or i == pop_dim-1:
                if pop[i, j] != 0:
                    xx = np.log(np.fabs(pop[i, j]))
                else:
                    xx = 0
                if pop[i, j] > 0:
                    c1 = 10.0
                    c2 = 7.9
                else:
                    c1 = 5.5
                    c2 = 3.1
                if pop[i, j] > 0:
                    sx = 1
                elif pop[i, j] == 0:
                    sx = 0
                else:
                    sx = -1
                pop_osz[i, j] = sx*np.exp(xx+0.049*(np.sin(c1*xx)+np.sin(c2*xx)))
            else:
                pop_osz[i, j] = pop[i, j]
    return pop_osz


def asy_pop_func(pop, beta):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]

    pop_asy = pop.copy()
    for j in range(pop_size):
        for i in range(pop_dim):
            if pop[i, j] > 0:
                pop_asy[i, j] = pow(pop[i, j], 1.0+beta*i/(pop_dim-1)*pow(pop[i, j], 0.5))
    return pop_asy


def asy_pop_func2(pop, pop_asy, beta):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    for j in range(pop_size):
        for i in range(pop_dim):
            if pop[i, j] > 0:
                pop_asy[i, j] = pow(pop[i, j], 1.0+beta*i/(pop_dim-1)*pow(pop[i, j], 0.5))
    return pop_asy


def sphere_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            f += pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def ellips_func(pop, shift, rotate, r_flag=1):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    pop_sr = osz_pop_func(pop_sr)
    for j in range(pop_size):
        f = 0
        for i in range(pop_dim):
            if pop_dim == 1:
                f += pow(10.0, 6.0 * i / pop_dim) * pop_sr[i, j] * pop_sr[i, j]
            else:
                f += pow(10.0, 6.0*i/(pop_dim-1))*pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def bent_cigar_func(pop, shift, rotate, r_flag):
    beta = 0.5
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr = pop_s
    pop_sr = asy_pop_func(pop_sr, beta)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    else:
        pop_sr = pop_sr
    for j in range(pop_size):
        f = pop_sr[0, j] * pop_sr[0, j]
        for i in range(1, pop_dim):
            f += 1e6 * pop_sr[i, j] * pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def discus_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    pop_sr = osz_pop_func(pop_sr)
    for j in range(pop_size):
        f = pow(10.0, 6.0)*pop_sr[0, j]*pop_sr[0, j]
        for i in range(pop_dim):
            if i > 0:
                f += pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def dif_powers_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            f += pow(np.fabs(pop_sr[i, j]), float(2 + 4*i/(pop_dim-1)))
        f = pow(f, 0.5)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def rosenbrock_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 2.048/100
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    pop_sr += 1.0
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            tmp1 = pop_sr[i, j] * pop_sr[i, j] - pop_sr[i+1, j]
            tmp2 = pop_sr[i, j] - 1.0
            f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def schaffer_F7_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr = pop_s
    pop_sr = asy_pop_func(pop_sr, 0.5)
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] = pop_sr[i, j] * pow(10.0, 1.0*i/(pop_dim-1)/2.0)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    for j in range(pop_size):
        for i in range(pop_dim-1):
            pop_sr[i, j] = pow(pop_sr[i, j]*pop_sr[i, j]+pop_sr[i+1, j]*pop_sr[i+1, j], 0.5)
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            tmp = np.sin(50.0*pow(pop_sr[i, j], 0.2))
            f += pow(pop_sr[i, j], 0.5)+pow(pop_sr[i, j], 0.5)*tmp*tmp
        f = f*f/(pop_dim-1)/(pop_dim-1)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def ackley_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr1 = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr1 = pop_s

    pop_sr = asy_pop_func2(pop_sr1, pop_s, 0.5)
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] = pop_sr[i, j]*pow(10.0, 1.0*i/(pop_dim-1)/2.0)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    for j in range(pop_size):
        sum1 = 0.0
        sum2 = 0.0
        for i in range(pop_dim):
            sum1 += pop_sr[i, j] * pop_sr[i, j]
            sum2 += np.cos(2.0 * PI * pop_sr[i, j])
        sum1 = -0.2 * np.sqrt(sum1 / pop_dim)
        sum2 /= pop_dim
        f = E - 20.0 * np.exp(sum1) - np.exp(sum2) + 20.0
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def weierstrass_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 0.5/100
    if r_flag == 1:
        pop_sr1 = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr1 = pop_s
    pop_sr = asy_pop_func2(pop_sr1, pop_s, 0.5)
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] = pop_sr[i, j] * pow(10.0, 1.0 * i / (pop_dim - 1) / 2.0)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    a = 0.5
    b = 3.0
    k_max = 20
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            sum = 0.0
            sum2 = 0.0
            for k in range(k_max+1):
                sum += pow(a, k) * np.cos(2.0 * PI * pow(b, k) * (pop_sr[i, j] + 0.5))
                sum2 += pow(a, k) * np.cos(2.0 * PI * pow(b, k) * 0.5)
            f += sum
        f -= pop_dim*sum2
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def griewank_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 600.0 / 100.0
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] = pop_sr[i, j] * pow(100.0, 1.0 * i / (pop_dim - 1) / 2.0)

    for j in range(pop_size):
        s = 0.0
        p = 1.0
        for i in range(pop_dim):
            s += pop_sr[i, j]*pop_sr[i, j]
            p *= np.cos(pop_sr[i, j]/np.sqrt(1.0+i))
        f = 1.0+s/4000.0-p
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def rastrigin_func(pop, shift, rotate, r_flag):
    alpha = 10.0
    beta = 0.2
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 5.12 / 100.0
    if r_flag == 1:
        pop_sr1 = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr1 = pop_s
    pop_sr = osz_pop_func(pop_sr1)
    pop_sr = asy_pop_func2(pop_sr, pop_sr1, beta)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] *= pow(alpha, 1.0*i/(pop_dim-1)/2)
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_sr)
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            f += (pop_sr[i, j]*pop_sr[i, j] - 10.0*np.cos(2.0*PI*pop_sr[i, j]) + 10.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def step_rastrigin_func(pop, shift, rotate, r_flag):
    alpha = 10.0
    beta = 0.2
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 5.12 / 100.0
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        for i in range(pop_dim):
            if np.fabs(pop_sr[i, j] > 0.5):
                pop_sr[i, j] = np.floor(2.0*pop_sr[i, j]+0.5)/2.0
    pop_sr1 = osz_pop_func(pop_sr)
    pop_sr = asy_pop_func2(pop_sr1, pop_sr, beta)
    if r_flag == 1:
        pop_sr1 = np.dot(rotate[1, :, :], pop_sr)
    else:
        pop_sr1 = pop_sr
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr1[i, j] *= pow(alpha, 1.0*i/(pop_dim-1)/2.0)
    if r_flag == 1:
        pop_sr2 = np.dot(rotate[0, :, :], pop_sr1)
    else:
        pop_sr2 = pop_sr1
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            f += (pop_sr2[i, j]*pop_sr2[i, j] - 10.0*np.cos(2.0*PI*pop_sr2[i, j]) + 10.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def schwefel_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 1000.0/ 100.0
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] = pop_sr[i, j] * pow(10.0, 1.0*i/(pop_dim-1)/2.0) + 4.209687462275036e+002
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            if pop_sr[i, j] > 500:
                f -= (500.0-np.mod(pop_sr[i, j], 500))*np.sin(pow(500.0-np.mod(pop_sr[i, j], 500), 0.5))
                tmp = (pop_sr[i, j]-500.0)/100.0
                f += tmp*tmp/pop_dim
            elif pop_sr[i, j] < -500:
                f -= (-500.0 + np.mod(np.fabs(pop_sr[i, j]), 500)) * np.sin(pow(500.0 - np.mod(np.fabs(pop_sr[i, j]), 500), 0.5))
                tmp = (pop_sr[i, j] + 500.0) / 100
                f += tmp * tmp / pop_dim
            else:
                f -= pop_sr[i, j]*np.sin(pow(np.fabs(pop_sr[i, j]), 0.5))
        f = 4.189828872724338e+002 * pop_dim + f
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def katsuura_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    tmp3 = pow(1.0*pop_dim, 1.2)
    pop_s = shift_pop(pop, shift) * 5.0/100.0
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr = pop_s
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] *= pow(100.0, 1.0*i/(pop_dim-1)/2.0)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    for j in range(pop_size):
        f = 1.0
        for i in range(pop_dim):
            temp = 0.0
            for k in range(1, 33):
                tmp1 = pow(2.0, float(k))
                tmp2 = tmp1 * pop_sr[i, j]
                temp += np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
            f *= pow(1.0+(i+1)*temp, 10.0/tmp3)
        tmp1 = 10.0 / pop_dim / pop_dim
        f = f * tmp1 - tmp1
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def bi_rastrigin_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    mu0 = 2.5
    d = 1.0
    s = 1.0-1.0/(2.0*pow(pop_dim+20.0, 0.5)-8.2)
    mu1 = -pow((mu0*mu0-d)/s, 0.5)
    pop_s = shift_pop(pop, shift) * 10.0 / 100.0
    tmpx = 2*pop_s.copy()
    for j in range(pop_size):
        for i in range(pop_dim):
            if shift[i] < 0:
                tmpx[i, j] *= -1
    pop_s1 = tmpx.copy()
    tmpx += mu0
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s1)
    else:
        pop_sr = pop_s1
    for j in range(pop_size):
        for i in range(pop_dim):
            pop_sr[i, j] *= pow(100.0, 1.0*i/(pop_dim-1)/2.0)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)

    for j in range(pop_size):
        tmp1 = 0.0
        tmp2 = 0.0
        for i in range(pop_dim):
            tmp = tmpx[i, j] - mu0
            tmp1 += tmp * tmp
            tmp = tmpx[i, j] - mu1
            tmp2 += tmp * tmp
        tmp2 *= s
        tmp2 += d * pop_dim
        tmp = 0
        for i in range(pop_dim):
            tmp += np.cos(2.0 * PI * pop_sr[i, j])
        if tmp1 < tmp2:
            f = tmp1
        else:
            f = tmp2
        f += 10.0 * (pop_dim-tmp)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def grie_rosen_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift) * 5.0 / 100.0
    if r_flag == 1:
        pop_sr = np.dot(rotate, pop_s)
    else:
        pop_sr = pop_s
    pop_sr = pop_s + 1
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            tmp1 = pop_sr[i, j]*pop_sr[i, j]-pop_sr[i+1, j]
            tmp2 = pop_sr[i, j]-1.0
            temp = 100.0*tmp1*tmp1 + tmp2*tmp2
            f += (temp*temp)/4000.0 - np.cos(temp) + 1.0
        tmp1 = pop_sr[pop_dim - 1, j] * pop_sr[pop_dim - 1, j] - pop_sr[0, j]
        tmp2 = pop_sr[pop_dim - 1, j] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def escaffer6_func(pop, shift, rotate, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    pop_s = shift_pop(pop, shift)
    if r_flag == 1:
        pop_sr = np.dot(rotate[0, :, :], pop_s)
    else:
        pop_sr = pop_s
    pop_sr = asy_pop_func(pop_sr, 0.5)
    if r_flag == 1:
        pop_sr = np.dot(rotate[1, :, :], pop_sr)
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            temp1 = np.sin(np.sqrt(pop_sr[i, j] * pop_sr[i, j] + pop_sr[i + 1, j] * pop_sr[i + 1, j]))
            temp1 = temp1 * temp1
            temp2 = 1.0 + 0.001 * (pop_sr[i, j] * pop_sr[i, j] + pop_sr[i + 1, j] * pop_sr[i + 1, j])
            f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
        temp1 = np.sin(np.sqrt(pop_sr[pop_dim - 1, j] * pop_sr[pop_dim - 1, j] + pop_sr[0, j] * pop_sr[0, j]))
        temp1 = temp1 * temp1
        temp2 = 1.0 + 0.001 * (pop_sr[pop_dim - 1, j] * pop_sr[pop_dim - 1, j] + pop_sr[0, j] * pop_sr[0, j])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit


def cf01(pop, shift, rotate, r_flag):
    cf_num = 5
    delta = np.array([10, 20, 30, 40, 50])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = rosenbrock_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 10000*f1/1e4
    f2 = dif_powers_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], r_flag)
    f2 = 10000*f2/1e10
    f3 = bent_cigar_func(pop, shift[:, 2].reshape(-1, 1), rotate[2:4, :, :], r_flag)
    f3 = 10000*f3/1e30
    f4 = discus_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], r_flag)
    f4 = 10000*f4/1e10
    f5 = sphere_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], r_flag)
    f5 = 10000*f5/1e5
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf02(pop, shift, rotate, r_flag):
    cf_num = 3
    delta = np.array([20, 20, 20])
    bias = np.array([0, 100, 200])
    f1 = schwefel_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f2 = schwefel_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], r_flag)
    f3 = schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf03(pop, shift, rotate, r_flag):
    cf_num = 3
    delta = np.array([20, 20, 20])
    bias = np.array([0, 100, 200])
    f1 = schwefel_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f2 = schwefel_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], r_flag)
    f3 = schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf04(pop, shift, rotate, r_flag):
    cf_num = 3
    delta = np.array([20, 20, 20])
    bias = np.array([0, 100, 200])
    f1 = schwefel_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 1000*f1/4e3
    f2 = rastrigin_func(pop, shift[:, 1].reshape(-1, 1), rotate[1:3, :, :], r_flag)
    f2 = 1000*f2/1e3
    f3 = weierstrass_func(pop, shift[:, 2].reshape(-1, 1), rotate[2:4, :, :], r_flag)
    f3 = 1000*f3/400
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf05(pop, shift, rotate, r_flag):
    cf_num = 3
    delta = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    f1 = schwefel_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 1000 * f1 / 4e3
    f2 = rastrigin_func(pop, shift[:, 1].reshape(-1, 1), rotate[1:3, :, :], r_flag)
    f2 = 1000 * f2 / 1e3
    f3 = weierstrass_func(pop, shift[:, 2].reshape(-1, 1), rotate[2:4, :, :], r_flag)
    f3 = 1000 * f3 / 400
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf06(pop, shift, rotate, r_flag):
    cf_num = 5
    delta = np.array([10, 10, 10, 10, 10])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = schwefel_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 1000 * f1 / 4e3
    f2 = rastrigin_func(pop, shift[:, 1].reshape(-1, 1), rotate[1:3, :, :], r_flag)
    f2 = 1000 * f2 / 1e3
    f3 = ellips_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], r_flag)
    f3 = 1000 * f3 / 1e10
    f4 = weierstrass_func(pop, shift[:, 3].reshape(-1, 1), rotate[3:5, :, :], r_flag)
    f4 = 1000 * f4 / 400
    f5 = griewank_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], r_flag)
    f5 = 1000 * f5 / 100
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf07(pop, shift, rotate, r_flag):
    cf_num = 5
    delta = np.array([10, 10, 10, 20, 20])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = griewank_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 10000 * f1 / 100
    f2 = rastrigin_func(pop, shift[:, 1].reshape(-1, 1), rotate[1:3, :, :], r_flag)
    f2 = 10000 * f2 / 1e3
    f3 = schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], r_flag)
    f3 = 10000 * f3 / 4e3
    f4 = weierstrass_func(pop, shift[:, 3].reshape(-1, 1), rotate[3:5, :, :], r_flag)
    f4 = 10000 * f4 / 400
    f5 = sphere_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], r_flag)
    f5 = 10000 * f5 / 1e5
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf08(pop, shift, rotate, r_flag):
    cf_num = 5
    delta = np.array([10, 20, 30, 40, 50])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = grie_rosen_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], r_flag)
    f1 = 10000 * f1 / 4e3
    f2 = schaffer_F7_func(pop, shift[:, 1].reshape(-1, 1), rotate[1:3, :, :], r_flag)
    f2 = 10000 * f2 / 4e6
    f3 = schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], r_flag)
    f3 = 10000 * f3 / 4e3
    f4 = escaffer6_func(pop, shift[:, 3].reshape(-1, 1), rotate[3:5, :, :], r_flag)
    f4 = 10000 * f4 / 2e7
    f5 = sphere_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], r_flag)
    f5 = 10000 * f5 / 1e5
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num)
    fit = cf_cal233(pop, f, shift, delta, bias, cf_num)
    return fit


def cf_cal233(pop, f, shift, sigma, bias, cf_num):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    w = np.zeros_like(f)
    norm_w = np.zeros_like(f)
    for i in range(pop_size):
        for j in range(cf_num):
            f[i, j] += bias[j]

    for i in range(pop_size):
        for j in range(cf_num):
            tmp=0.0
            for k in range(pop_dim):
                tmp += (pop[k, i]-shift[k, j])*(pop[k, i]-shift[k, j])

            if tmp != 0:
                w[i, j] = pow(1.0 / tmp, 0.5) * np.exp(-tmp / 2.0 / pop_dim / pow(sigma[j], 2.0))
            else:
                zero_index = j
                for j in range(cf_num):
                    if j == zero_index:
                        w[i, j] = 1
                    else:
                        w[i, j] = 0
                break

    w_sum = np.sum(w, 1)
    w_max = np.max(w, 1)
    for i in range(pop_size):
        if w_max[i] == 0:
            for j in range(cf_num):
                w[i, j] = 1
            w_sum[i] = cf_num
    for i in range(pop_size):
        for j in range(cf_num):
            norm_w[i, j] = w[i, j]/w_sum[i]
    fit = (np.sum(f*norm_w, 1)).reshape(-1, 1)
    return fit
