import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
T = 24

dt = 1
lambda_E = np.ones(T) * 600
lambda_G = np.ones(T) * 2.5
lambda_C = np.ones(T) * 90
mu_C = 260
mu_pen = 600
mu_ES = 83
sigma_CO2 = 0.8

alpha_CHP1 = 0.15
alpha_CHP2 = 0.2
alpha_CHP3 = 0.8
beta_CHP = 18.18
gamma_CO2 = 0.002
P_CHP_min = 0.01
P_CHP_max = 2

chi_GT = 0.004
P_GT_min = 0.01
P_GT_max = 2

omega_CCS = 0.55
eta_CCS = 0.9

phi_P2G = 0.98
psi_P2G = 0.539
zeta_P2G = 274.4
P_CCS_fix = 0

P_chs_max = 0.1
P_dis_max = 0.1
eta_ES = 0.9
SoC_max = 2
SoC_0 = SoC_max / 2

# ----------------------------
# Variables
# ----------------------------
P_CHP_E = cp.Variable(T)
P_CHP_H = cp.Variable(T)
P_GT_E = cp.Variable(T)
G_CHP = cp.Variable(T)
G_GT = cp.Variable(T)
C_CHP = cp.Variable(T)
C_GT = cp.Variable(T)

P_buy = cp.Variable(T)
G_buy = cp.Variable(T)
P_W = cp.Parameter(T)
P_cur = cp.Variable(T)
P_dis = cp.Variable(T)
P_chs = cp.Variable(T)

u_chs = cp.Variable(T, boolean=True)
u_dis = cp.Variable(T, boolean=True)
SoC = cp.Variable(T)

P_CCS = cp.Variable(T)
C_cap = cp.Variable(T)
C_emi = cp.Variable(T)

P_P2G_E = cp.Variable(T)
P_P2G_H = cp.Variable(T)
G_P2G = cp.Variable(T)

P_load = cp.Parameter(T)
P_load_H = cp.Parameter(T)
C_quota = cp.Variable(T)

# ----------------------------
# Data input (example values)
# ----------------------------
P_load_H.value = np.array([0.0445, 0.0466, 0.04984, 0.04804, 0.04475, 0.05048, 0.04415, 0.04323, 0.03206, 0.0366, 0.02967, 0.02608, 0.03289, 0.03438, 0.04687, 0.05356, 0.0494, 0.05407, 0.04907, 0.05483, 0.05387, 0.05049, 0.04493, 0.04387])
P_load.value = np.array([0.0408, 0.03932, 0.04244, 0.04414, 0.04276, 0.0414, 0.04324, 0.04418, 0.04304, 0.04546, 0.04344, 0.04432, 0.04516, 0.04372, 0.0437, 0.04122, 0.04226, 0.04214, 0.0389, 0.0425, 0.04328, 0.04524, 0.0435, 0.04148])
P_W.value = np.array([0.01658, 0.01673, 0.01625, 0.01575, 0.01604, 0.01677, 0.01898, 0.02158, 0.02409, 0.02592, 0.03004, 0.03036, 0.03364, 0.02461, 0.02424, 0.02243, 0.02026, 0.01964, 0.01814, 0.01805, 0.01787, 0.01727, 0.01681, 0.01731])

# ----------------------------
# Constraints
# ----------------------------
Constraints = []
Constraints += [
    P_CHP_E <= P_CHP_max - alpha_CHP1 * P_CHP_H,
    P_CHP_E >= P_CHP_min - alpha_CHP2 * P_CHP_H,
    P_CHP_E >= alpha_CHP3 * P_CHP_H,
    G_CHP == beta_CHP * P_CHP_H,
    C_CHP == gamma_CO2 * G_CHP,
    P_GT_E >= P_GT_min,
    P_GT_E <= P_GT_max,
    P_GT_E == chi_GT * G_GT,
    C_GT == gamma_CO2 * G_GT,
    P_CCS == P_CCS_fix + omega_CCS * C_cap,
    C_cap == eta_CCS * (C_CHP + C_GT),
    C_emi == (1 - eta_CCS) * (C_CHP + C_GT),
    P_P2G_E == phi_P2G * C_cap,
    P_P2G_H == psi_P2G * C_cap,
    G_P2G == zeta_P2G * C_cap,
    SoC[0] == SoC_0 + dt * (eta_ES * P_chs[0] - P_dis[0] / eta_ES),
    SoC <= SoC_max,
    SoC >= 0,
    P_chs <= P_chs_max * u_chs,
    P_dis <= P_dis_max * u_dis,
    u_chs + u_dis <= 1,
    P_cur >= 0,
    P_buy >= 0,
    P_CHP_E >= 0,
    P_CHP_H >= 0,
    P_GT_E >= 0,
    G_CHP >= 0,
    G_GT >= 0,
    C_CHP >= 0,
    C_GT >= 0,
    C_cap >= 0,
    C_emi >= 0,
    P_P2G_E >= 0,
    P_P2G_H >= 0,
    G_P2G >= 0,
    P_chs >= 0,
    P_dis >= 0
]

for t in range(1, T):
    Constraints.append(SoC[t] == SoC[t-1] + dt * (eta_ES * P_chs[t] - P_dis[t] / eta_ES))

Constraints += [
    P_buy + P_W - P_cur + P_CHP_E + P_GT_E + P_dis == P_load + P_CCS + P_P2G_E + P_chs,
    G_buy + G_P2G == G_CHP + G_GT,
    C_CHP + C_GT == C_cap + C_emi,
    P_CHP_H + P_P2G_H >= P_load_H - 0.005,
    P_CHP_H + P_P2G_H <= P_load_H + 0.005
]

# Carbon quota
Constraints += [C_quota == sigma_CO2 * (P_buy + P_W + P_CHP_E + P_GT_E)]

# ----------------------------
# Objective
# ----------------------------
E_E = lambda_E @ P_buy
E_G = lambda_G @ G_buy
E_C = mu_C * cp.sum(C_cap + C_emi) + lambda_C @ (C_emi - C_quota)
E_W = mu_pen * cp.sum(P_cur)
E_ES = mu_ES * cp.sum(P_chs + P_dis)
Objective = cp.Minimize(E_E + E_G + E_C + E_W + E_ES)

# ----------------------------
# Solve
# ----------------------------
problem = cp.Problem(Objective, Constraints)
problem.solve(solver=cp.GUROBI, verbose=True)

# ----------------------------
# Results
# ----------------------------
if problem.status == cp.OPTIMAL:
    print("Success！")
    print("Minimum cost：{:.2f}".format(47.7*problem.value))
else:
    print("Else：", problem.status)

# Result
result_df = pd.DataFrame({
    "P_CHP_E": P_CHP_E.value,
    "P_buy": P_buy.value
})
result_df.to_excel("results.xlsx", index=False)
