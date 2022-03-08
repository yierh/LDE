"""""
Filename: PGnet_torch.py
   Paper: J. Sun, X. Liu, T. Bäck and Z. Xu, "Learning Adaptive Differential Evolution Algorithm From
          Optimization Experiences by Policy Gradient," in IEEE Transactions on Evolutionary Computation,
          vol. 25, no. 4, pp. 666-680, Aug. 2021, doi: 10.1109/TEVC.2021.3060811.
  Author: Xin Liu
"""""
import torch
import torch.nn as nn
import os
from de_croselect import *
import matplotlib.pyplot as pl


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.lstm = nn.LSTM(POP_SIZE+BINS*2, CELL_SIZE, LAYERS_NUM)
        self.mu = nn.Linear(CELL_SIZE, 2*POP_SIZE)
        self.sigma = nn.Linear(CELL_SIZE, 2*POP_SIZE)
        self.distribution = torch.distributions.Normal

    def forward(self, x, h, c):
        cell_out, (h_, c_) = self.lstm(x, (h, c))
        mu = self.mu(cell_out)
        sigma = torch.sigmoid(self.sigma(cell_out))
        return mu, sigma, h_, c_

    def sampler(self, inputs, ht, ct):
        mu, sigma, ht_, ct_ = self.forward(inputs, ht, ct)
        normal = self.distribution(mu, sigma)
        sample_w = np.clip(normal.sample().numpy(), 0, 1)
        return sample_w, ht_, ct_


def generate_pop(population_size, input_dimension, x_min, x_max):
    pop = np.zeros((population_size, input_dimension))
    for i in range(population_size):
        for j in range(input_dimension):
            pop[i, j] = x_min + np.random.uniform() * (x_max - x_min)
    return pop


def mulgenerate_pop(p, population_size, input_dimension, x_min, x_max):
    for i in range(p):
        if i == 0:
            pop = generate_pop(population_size, input_dimension, x_min, x_max)
        else:
            pop_c = generate_pop(population_size, input_dimension, x_min, x_max)
            pop = np.vstack((pop, pop_c))
    return pop.reshape(-1, population_size, input_dimension)


def order_by_f(pop, fit):
    sorted_array = np.argsort(fit.flatten())
    temp_pop = pop[sorted_array]
    temp_fit = fit[sorted_array]
    return temp_pop, temp_fit


def maxmin_norm(a):
    if np.max(a) != np.min(a):
        a = (a-np.min(a))/(np.max(a)-np.min(a))
    return a


def con2mat_current2pbest_Nw(mutation_vector, p):
    p_index_array = np.random.randint(0, int(np.ceil(POP_SIZE*p)), size=POP_SIZE)
    mutation_mat = np.zeros((POP_SIZE, POP_SIZE))
    for i in range(mutation_mat.shape[0]):
        mutation_mat[i, i] = 1 - mutation_vector[0, i]
        if p_index_array[i] != i:
            mutation_mat[i, p_index_array[i]] = mutation_vector[0, i]
        else:
            mutation_mat[i, i] = 1
    return mutation_mat


def con2mat_rand2pbest_Nw(mutation_vector, nfes):
    p_rate = (P_MIN - P_INI) * nfes/MAXFE + P_INI
    mutation_mat = con2mat_current2pbest_Nw(mutation_vector, p_rate)
    return mutation_mat


def add_random(m_pop, pop, mu):
    mur_pop = np.zeros((pop.shape[0], pop.shape[1]))

    for i in range(pop.shape[0]):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == i:
            r1 = np.random.randint(0, pop.shape[0])
        while r2 == i or r2 == r1:
            r2 = np.random.randint(0, pop.shape[0])
        mur_pop[i, :] = m_pop[i, :] + mu[0, i]*(pop[r1, :] - pop[r2, :])
    return mur_pop


def discounted_norm_rewards(r):
    for ep in range(TRAJECTORY_NUM*PROBLEM_NUM):
        single_rs = r[ep*TRAJECTORY_LENGTH:ep*TRAJECTORY_LENGTH+TRAJECTORY_LENGTH]
        discounted_rs = np.zeros_like(single_rs)
        running_add = 0.
        for t in reversed(range(0, TRAJECTORY_LENGTH)):
            running_add = running_add * GAMMA + single_rs[t]
            discounted_rs[t] = running_add
        if ep == 0:
            all_disc_norm_rs = discounted_rs
        else:
            all_disc_norm_rs = np.hstack((all_disc_norm_rs, discounted_rs))
    return all_disc_norm_rs


def save_plot(x, fit, reward, i):
    pl.figure(i+1, figsize=(20, 10), dpi=100)
    pl.subplot(1, 2, 1)
    pl.plot(x, np.array(fit), color='green', label='f' + str(i+1))
    pl.legend()
    pl.title('Mean Best Fitness for all episodes at T of the Population - f' + str(i+1))
    pl.xlabel('t')
    pl.ylabel('mean best fit')

    pl.subplot(1, 2, 2)
    pl.plot(x, np.array(reward), color='red', label='mean reward - f' + str(i+1))
    pl.legend()
    pl.title('Mean Relative Reward of the Population - f' + str(i+1))
    pl.xlabel('t')
    pl.ylabel('R')
    pl.savefig(str(i+1) + '.png')


seq_bst_fit1_mean, seq_bst_fit2_mean, seq_bst_fit3_mean, seq_bst_fit4_mean, seq_bst_fit5_mean = [], [], [], [], []
seq_bst_fit6_mean, seq_bst_fit7_mean, seq_bst_fit8_mean, seq_bst_fit9_mean, seq_bst_fit10_mean = [], [], [], [], []
seq_bst_fit11_mean, seq_bst_fit12_mean, seq_bst_fit13_mean, seq_bst_fit14_mean, seq_bst_fit15_mean = [], [], [], [], []
seq_bst_fit16_mean, seq_bst_fit17_mean, seq_bst_fit18_mean, seq_bst_fit19_mean, seq_bst_fit20_mean = [], [], [], [], []
seq_bst_fit21_mean, seq_bst_fit22_mean, seq_bst_fit23_mean, seq_bst_fit24_mean, seq_bst_fit25_mean = [], [], [], [], []
seq_bst_fit26_mean, seq_bst_fit27_mean, seq_bst_fit28_mean = [], [], []
seq_reward_meanepi, seq_reward7_meanepi, seq_reward13_meanepi, seq_reward19_meanepi = [], [], [], []
seq_reward2_meanepi, seq_reward8_meanepi, seq_reward14_meanepi, seq_reward20_meanepi = [], [], [], []
seq_reward3_meanepi, seq_reward9_meanepi, seq_reward15_meanepi = [], [], []
seq_reward4_meanepi, seq_reward10_meanepi, seq_reward16_meanepi = [], [], []
seq_reward5_meanepi, seq_reward11_meanepi, seq_reward17_meanepi = [], [], []
seq_reward6_meanepi, seq_reward12_meanepi, seq_reward18_meanepi = [], [], []
seq_reward21_meanepi, seq_reward22_meanepi, seq_reward23_meanepi, seq_reward24_meanepi = [], [], [], []
seq_reward25_meanepi, seq_reward26_meanepi, seq_reward27_meanepi, seq_reward28_meanepi = [], [], [], []

MN = PolicyNet()
optimizer = torch.optim.Adam(MN.parameters(), lr=LEARNING_RATE)
all_norm_pop = mulgenerate_pop(num_train_data, POP_SIZE, PROBLEM_SIZE, -1, 1)
for epoch in range(max_epoch):
    for iter_data in range(num_train_data):
        pop_ini_norm = all_norm_pop[iter_data, :, :]
        pop_ini = pop_ini_norm * X_MAX
        inputs, sf_crs, hs, cs, rewards = [], [], [], [], []
        optimizer.zero_grad()
        for p in range(PROBLEM_NUM):
            fit_ini = cec13(pop_ini, p+1)
            mean_fit = 0.
            mean_reward = 0.
            for l in range(TRAJECTORY_NUM):
                pop = pop_ini.copy()
                fit = fit_ini.copy()
                nfes = POP_SIZE
                h0 = torch.zeros(LAYERS_NUM, 1, CELL_SIZE)
                c0 = torch.zeros(LAYERS_NUM, 1, CELL_SIZE)
                past_histo = (POP_SIZE/BINS) * np.ones((1, BINS))
                for t in range(TRAJECTORY_LENGTH):
                    pop, fit = order_by_f(pop, fit)
                    fitness = maxmin_norm(fit)
                    hist_fit, _ = np.histogram(fitness, BINS)
                    mean_past_histo = np.mean(past_histo, 0)
                    input_net = np.hstack((fitness.reshape(1, -1), hist_fit.reshape(1, -1), mean_past_histo.reshape(1, -1)))
                    sf_cr, h_, c_ = MN.sampler(torch.FloatTensor(input_net[None, :]), h0, c0)  # parameter controller

                    sf_cr = np.squeeze(sf_cr, axis=0)  # scale factor
                    sf = sf_cr[:, 0:POP_SIZE]
                    cr = sf_cr[:, POP_SIZE:2*POP_SIZE]  # crossover rate
                    sf_mat = con2mat_rand2pbest_Nw(sf, nfes)
                    mu_pop = add_random(np.dot(sf_mat, pop), pop, sf)

                    pop_next, fit_next, nfes = de_crosselect(pop, mu_pop, fit, cr, nfes, p+1, train_flag=1)  # DE
                    bsf = np.min(fit)
                    bsf_next = np.min(fit_next)

                    reward = (bsf - bsf_next)/bsf  # reward
                    mean_reward += reward

                    inputs.append(input_net)
                    sf_crs.append(sf_cr)
                    hs.append(np.squeeze(h0.data.numpy(), axis=0))
                    cs.append(np.squeeze(c0.data.numpy(), axis=0))
                    rewards.append(reward)

                    fit = fit_next.copy()
                    pop = pop_next.copy()
                    past_histo = np.vstack((past_histo, hist_fit))
                    h0 = h_
                    c0 = c_
                mean_fit += np.min(fit)
            if p == 0:
                seq_bst_fit1_mean.append(mean_fit/TRAJECTORY_NUM)
                seq_reward_meanepi.append(mean_reward/TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 1:
                seq_bst_fit2_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward2_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 2:
                seq_bst_fit3_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward3_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 3:
                seq_bst_fit4_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward4_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 4:
                seq_bst_fit5_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward5_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 5:
                seq_bst_fit6_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward6_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 6:
                seq_bst_fit7_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward7_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 7:
                seq_bst_fit8_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward8_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 8:
                seq_bst_fit9_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward9_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 9:
                seq_bst_fit10_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward10_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 10:
                seq_bst_fit11_mean.append(mean_fit/TRAJECTORY_NUM)
                seq_reward11_meanepi.append(mean_reward/TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 11:
                seq_bst_fit12_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward12_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 12:
                seq_bst_fit13_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward13_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 13:
                seq_bst_fit14_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward14_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 14:
                seq_bst_fit15_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward15_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 15:
                seq_bst_fit16_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward16_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 16:
                seq_bst_fit17_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward17_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 17:
                seq_bst_fit18_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward18_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 18:
                seq_bst_fit19_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward19_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 19:
                seq_bst_fit20_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward20_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", np.min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 20:
                seq_bst_fit21_mean.append(mean_fit/TRAJECTORY_NUM)
                seq_reward21_meanepi.append(mean_reward/TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 21:
                seq_bst_fit22_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward22_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 22:
                seq_bst_fit23_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward23_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 23:
                seq_bst_fit24_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward24_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 24:
                seq_bst_fit25_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward25_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 25:
                seq_bst_fit26_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward26_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 26:
                seq_bst_fit27_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward27_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
            elif p == 27:
                seq_bst_fit28_mean.append(mean_fit / TRAJECTORY_NUM)
                seq_reward28_meanepi.append(mean_reward / TRAJECTORY_NUM)
                print("epoch:", epoch + 1, "data:", iter_data + 1, "f", p + 1, "initial min fit:", min(fit_ini),
                      " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward:", mean_reward / TRAJECTORY_NUM)
        # update network parameters
        all_eps_mean, all_eps_std, all_eps_h, all_eps_c = MN.forward(torch.FloatTensor(np.vstack(inputs)[None, :]),
                                                                     torch.Tensor(np.vstack(hs)[None, :]),
                                                                     torch.Tensor(np.vstack(cs)[None, :]))
        sf_crs = torch.FloatTensor(np.vstack(sf_crs))
        all_eps_mean = torch.squeeze(all_eps_mean, 0)
        all_eps_std = torch.squeeze(all_eps_std, 0)
        normal_dis = torch.distributions.Normal(all_eps_mean, all_eps_std)
        log_prob = torch.sum(normal_dis.log_prob(sf_crs + 1e-8), 1)
        all_eps_dis_reward = discounted_norm_rewards(rewards)
        loss = - torch.mean(log_prob * torch.FloatTensor(all_eps_dis_reward))
        loss.backward()
        optimizer.step()
print("PG done")
path = os.path.abspath('.') + "/model/pg_net"
torch.save(MN.state_dict(), path)  # save model
print("All cec13-funcs model parameters are saved")
x = np.linspace(0, num_train_data*max_epoch-1, num_train_data*max_epoch)
save_plot(x, seq_bst_fit2_mean, seq_reward2_meanepi, 1)
# LDE
print("test starts")
last_bsf1 = np.zeros(NUM_RUNS)
last_bsf2 = np.zeros(NUM_RUNS)
last_bsf3 = np.zeros(NUM_RUNS)
last_bsf4 = np.zeros(NUM_RUNS)
last_bsf5 = np.zeros(NUM_RUNS)
last_bsf6 = np.zeros(NUM_RUNS)
last_bsf7 = np.zeros(NUM_RUNS)
last_bsf8 = np.zeros(NUM_RUNS)
last_bsf9 = np.zeros(NUM_RUNS)
last_bsf10 = np.zeros(NUM_RUNS)
last_bsf11 = np.zeros(NUM_RUNS)
last_bsf12 = np.zeros(NUM_RUNS)
last_bsf13 = np.zeros(NUM_RUNS)
last_bsf14 = np.zeros(NUM_RUNS)
last_bsf15 = np.zeros(NUM_RUNS)
last_bsf16 = np.zeros(NUM_RUNS)
last_bsf17 = np.zeros(NUM_RUNS)
last_bsf18 = np.zeros(NUM_RUNS)
last_bsf19 = np.zeros(NUM_RUNS)
last_bsf20 = np.zeros(NUM_RUNS)
last_bsf21 = np.zeros(NUM_RUNS)
last_bsf22 = np.zeros(NUM_RUNS)
last_bsf23 = np.zeros(NUM_RUNS)
last_bsf24 = np.zeros(NUM_RUNS)
last_bsf25 = np.zeros(NUM_RUNS)
last_bsf26 = np.zeros(NUM_RUNS)
last_bsf27 = np.zeros(NUM_RUNS)
last_bsf28 = np.zeros(NUM_RUNS)
last_bsf29 = np.zeros(NUM_RUNS)  # still test 30 funcs in the older version
last_bsf30 = np.zeros(NUM_RUNS)
all_fs_bsf = []
all_fs_bsf_T = []
for repeat in range(NUM_RUNS):
    test_norm_pop = -1 + np.random.uniform(size=(POP_SIZE, PROBLEM_SIZE)) * (1 - (-1))
    test_pop = test_norm_pop * X_MAX
    for test_p in range(TEST_PROBLEM):
        test_fit = cec17(test_pop, test_p + 1)
        test_nfes = POP_SIZE
        test_h0 = torch.zeros(LAYERS_NUM, 1, CELL_SIZE)
        test_c0 = torch.zeros(LAYERS_NUM, 1, CELL_SIZE)
        test_past_histo = (POP_SIZE / BINS) * np.ones((1, BINS))
        for t in range(int((MAXFE-POP_SIZE)/POP_SIZE)):
            if t == 0:
                pop_ = test_pop.copy()
                fit_ = test_fit.copy()
            pop_, fit_ = order_by_f(pop_, fit_)
            fitness_ = maxmin_norm(fit_)
            hist_fit_, _ = np.histogram(fitness_, BINS)
            test_mean_past_histo = np.mean(test_past_histo, 0)
            test_input = np.hstack((fitness_.reshape(1, -1), hist_fit_.reshape(1, -1),
                                    test_mean_past_histo.reshape(1, -1)))
            sf_cr_, test_h, test_c = MN.sampler(torch.FloatTensor(test_input[None, :]), test_h0, test_c0)
            sf_cr_ = np.squeeze(sf_cr_, axis=0)
            sf_ = sf_cr_[:, 0:POP_SIZE]
            cr_ = sf_cr_[:, POP_SIZE:2 * POP_SIZE]
            sf_mat_ = con2mat_rand2pbest_Nw(sf_, test_nfes)
            mu_pop_ = add_random(np.dot(sf_mat_, pop_), pop_, sf_)
            pop_next_, fit_next_, test_nfes = de_crosselect(pop_, mu_pop_, fit_, cr_, test_nfes, test_p + 1,
                                                            train_flag=0)
            pop_ = pop_next_.copy()
            fit_ = fit_next_.copy()
            test_h0 = test_h
            test_c0 = test_c
            test_past_histo = np.vstack((test_past_histo, hist_fit_))
            bst_test = min(fit_)
            if t+1 == TRAJECTORY_LENGTH:
                all_fs_bsf_T.append(bst_test)
            if np.min(fit_) < EPSILON: break
        all_fs_bsf.append(bst_test)
        if test_p+1 == 1:
            last_bsf1[repeat] = bst_test
        elif test_p+1 == 2:
            last_bsf2[repeat] = bst_test
        elif test_p+1 == 3:
            last_bsf3[repeat] = bst_test
        elif test_p+1 == 4:
            last_bsf4[repeat] = bst_test
        elif test_p+1 == 5:
            last_bsf5[repeat] = bst_test
        elif test_p+1 == 6:
            last_bsf6[repeat] = bst_test
        elif test_p+1 == 7:
            last_bsf7[repeat] = bst_test
        elif test_p+1 == 8:
            last_bsf8[repeat] = bst_test
        elif test_p+1 == 9:
            last_bsf9[repeat] = bst_test
        elif test_p+1 == 10:
            last_bsf10[repeat] = bst_test
        elif test_p+1 == 11:
            last_bsf11[repeat] = bst_test
        elif test_p+1 == 12:
            last_bsf12[repeat] = bst_test
        elif test_p+1 == 13:
            last_bsf13[repeat] = bst_test
        elif test_p+1 == 14:
            last_bsf14[repeat] = bst_test
        elif test_p+1 == 15:
            last_bsf15[repeat] = bst_test
        elif test_p+1 == 16:
            last_bsf16[repeat] = bst_test
        elif test_p+1 == 17:
            last_bsf17[repeat] = bst_test
        elif test_p+1 == 18:
            last_bsf18[repeat] = bst_test
        elif test_p+1 == 19:
            last_bsf19[repeat] = bst_test
        elif test_p+1 == 20:
            last_bsf20[repeat] = bst_test
        elif test_p+1 == 21:
            last_bsf21[repeat] = bst_test
        elif test_p+1 == 22:
            last_bsf22[repeat] = bst_test
        elif test_p+1 == 23:
            last_bsf23[repeat] = bst_test
        elif test_p+1 == 24:
            last_bsf24[repeat] = bst_test
        elif test_p+1 == 25:
            last_bsf25[repeat] = bst_test
        elif test_p+1 == 26:
            last_bsf26[repeat] = bst_test
        elif test_p+1 == 27:
            last_bsf27[repeat] = bst_test
        elif test_p+1 == 28:
            last_bsf28[repeat] = bst_test
        elif test_p+1 == 29:
            last_bsf29[repeat] = bst_test
        elif test_p+1 == 30:
            last_bsf30[repeat] = bst_test

all_fs_bsf_T = np.transpose(np.array(all_fs_bsf_T).reshape(NUM_RUNS, TEST_PROBLEM))
np.savetxt('CEC17_all_51run_'+str(PROBLEM_NUM)+'DN'+str(POP_SIZE)+'size'+str(CELL_SIZE)+'L'+str(TRAJECTORY_NUM)
           +'_PG_MAXFE.txt', np.transpose(np.array(all_fs_bsf).reshape(NUM_RUNS, TEST_PROBLEM)))
