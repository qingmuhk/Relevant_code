
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Figure
def plot_ies_results(P_mt, P_e1, P_wind, P_pv, P_er, P_pl,
                     P_h, P_mth, P_hl,
                     P_gs, P_s, P_mts, P_gl,
                     P_mtc, P_erc, P_cl,
                     P_e2, C_cc):
    T = len(P_mt)
    x = np.arange(T)

    plt.figure(figsize=(10,6))
    bottom = np.zeros(T)
    for data, label in zip([P_mt, P_e1, P_wind, P_pv, P_er],
                           ['GT', 'CHP', 'Wind', 'PV', 'Re']):
        plt.bar(x, data, bottom=bottom, label=label)
        bottom += data
    plt.plot(x, P_pl, 'k-o', label='E_load', linewidth=2)
    plt.xlabel("Time(h)"); plt.ylabel("E_power(MW)"); plt.title("Electricity balance")
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    bottom = np.zeros(T)
    for data, label in zip([P_mth, P_h], ['GT', 'CHP']):
        plt.bar(x, data, bottom=bottom, label=label)
        bottom += data
    plt.plot(x, P_hl, 'k-o', label='H_load', linewidth=2)
    plt.xlabel("Time(h)"); plt.ylabel("H_power(MW)"); plt.title("Heat Balance")
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.show()

# Model 4
T = 24
Pre_wind = np.array([37.07,38.69,37.65,36.5,30.61,15.59,10.39,7.85,8.77,13.28,13.97,16.4,15.47,17.21,13.63,15.13,16.05,14.66,17.21,17.55,18.48,31.072,36.61,36.38])
Pre_pv = np.array([0,0,0,0,0,0.79,4.46,8.88,13.73,17.68,22.79,24.89,26.74,25.39,15.76,8.28,1,0,0,0,0,0,0,0])
P_pl = np.array([35.57,35.78,37.015,37.079,42.86,45.059,45.275,46.066,47.22,46.13,47.92,48.29,50.23,47.21,46.35,48.10,52.36,54.26,55.208,55.099,53.46,41.93,36.17,35.11])
P_hl = np.array([34.24,37.18,35.98,37.44,37.088,36.64,34.62,34.24,34.63,34.308,35.22,32.45,32.34,32.22,32.34,32.57,33.40,33.60,33.53,33.96,34.31,38.10,37.80,36.50])
P_cl = np.array([16.402,16.402,15.414,16.341,16.286,16.175,15.285,15.362,18.549,20.269,22.225,24.257,24.254,24.062,22.399,17.295,16.511,16.325,15.308,16.395,16.395,16.202,15.204,16.287])
P_gl = np.array([10.627,12.426,12.027,11.588,12.944,14.795,14.577,14.208,12.382,11.322,12.235,15.133,15.476,15.351,14.068,13.066,12.334,13.44,14.12,14.97,14.134,12.921,12.208,11.32])

# Variables
P_e1 = cp.Variable(T)
P_e2 = cp.Variable(T)
P_e3 = cp.Variable(T)
P_h = cp.Variable(T)
P_gs = cp.Variable(T)
C_cc = cp.Variable(T)
P_mt = cp.Variable(T)
P_mts = cp.Variable(T)
P_mth = cp.Variable(T)
P_mtc = cp.Variable(T)
P_erc = cp.Variable(T)
P_er = cp.Variable(T)
P_wind = cp.Variable(T)
P_cwind = cp.Variable(T)
P_pv = cp.Variable(T)
P_cpv = cp.Variable(T)
P_s = cp.Variable(T)

# Constraints
constraints = []
constraints += [
    P_e1 >= 10 - P_e2 - P_e3,
    P_e1 <= 35 - P_e2 - P_e3,
    P_e2 >= 0, P_e2 <= 15,
    P_e3 >= 0, P_e3 <= 10,
    P_e1 >= 0,
    P_h >= 0, P_h <= 40,
    P_gs == 0.55 * P_e2,
    C_cc == 1.02 * P_e2,
    P_e3 == 0.5 * C_cc,
    cp.abs(P_e1[1:] + P_e2[1:] + P_e3[1:] - (P_e1[:-1] + P_e2[:-1] + P_e3[:-1])) <= 20
]
for t in range(T):
    base = P_e1[t] + P_e2[t] + P_e3[t] + 0.15 * P_h[t]
    constraints.append(C_cc[t] <= 0.89 * base + 26.15)

constraints += [
    P_mt == 0.6 * P_mts,
    P_mth == 0.95 * 1.9 * P_mt * (1 - 0.6 - 0.05) / 0.6,
    P_mtc == 0.95 * 2.4 * P_mt * (1 - 0.6 - 0.05) / 0.6,
    P_mt >= 5, P_mt <= 30,
    cp.abs(P_mt[1:] - P_mt[:-1]) <= 20,
    P_erc == 3 * P_er,
    P_er >= 0, P_er <= 4,
    P_wind + P_cwind == Pre_wind,
    P_wind >= 0, P_cwind >= 0,
    P_pv + P_cpv == Pre_pv,
    P_pv >= 0, P_cpv >= 0,
    P_wind + P_pv + P_e1 + P_mt == P_pl + P_er,
    P_h + P_mth >= P_hl - 0.1 * P_hl,
    P_h + P_mth <= P_hl + 0.1 * P_hl,
    P_erc + P_mtc >= P_cl - 0.1 * P_cl,
    P_erc + P_mtc <= P_cl + 0.1 * P_cl,
    P_gl + P_mts <= P_gs + P_s,
    P_s >= 0, P_s <= 30
]

C3 = 13.29 * cp.sum(P_e1 + P_e2 + P_e3) + 0.004 * cp.sum_squares(P_e1 + P_e2 + P_e3) + 39 * 24 + 22 * cp.sum(P_e2 + P_e3)
base_vector = P_e1 + P_e2 + P_e3 + 0.15 * P_h
E_co2 = 0.6 * cp.sum(base_vector) + 0.0017 * cp.sum_squares(base_vector) + 26.15 * 24 - cp.sum(C_cc) + 1.09 * cp.sum(P_mt)
E_0 = 0.798 * cp.sum(P_e1 + P_e2 + P_e3 + P_mt + P_pv + P_wind)
C4 = 30 * (E_co2 - E_0)
C5 = 60 * cp.sum(P_mts)
C6 = 120 * cp.sum(P_cwind)
C7 = 120 * cp.sum(P_cpv)
C8 = 26 * cp.sum(P_er)
objective = cp.Minimize(C3 + C4 + C5 + C6 + C7 + C8)

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI, verbose=True)

# Print
if problem.status == cp.OPTIMAL:
    print("Success！")
    print(f"Minimum cost：{problem.value:.2f} USD")
    plot_ies_results(P_mt.value, P_e1.value, P_wind.value, P_pv.value, P_er.value, P_pl,
                     P_h.value, P_mth.value, P_hl,
                     P_gs.value, P_s.value, P_mts.value, P_gl,
                     P_mtc.value, P_erc.value, P_cl,
                     P_e2.value, C_cc.value)
else:
    print("Fail")
