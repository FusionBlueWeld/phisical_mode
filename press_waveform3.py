import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
# 共通パラメータ
d0 = 0.0       # パンチがワークに接触する位置 [mm]
t = 5.0        # ワークの厚さ [mm]
A = 100.0      # パンチの断面積 [mm^2]

# 追加パラメータ
cl = 0.1       # クリアランス [mm]
pr = 0.05      # パンチのエッジのR [mm]
dr = 0.05      # ダイのエッジのR [mm]
h = 2000.0     # ワーク材料の硬さ [MPa]

# 基準値と係数（例）
pr0 = 0.01     # 基準パンチエッジR [mm]
cl0 = 0.05     # 基準クリアランス [mm]
dr0 = 0.01     # 基準ダイエッジR [mm]
h0 = 1500.0    # 基準硬さ [MPa]

k_pr = 10.0    # パンチエッジRの影響係数
k_cl = 5.0     # クリアランスの影響係数
k_dr = 5.0     # ダイエッジRの影響係数

# モード1のパラメータ（ヘルツの接触理論）
E_punch = 210e3     # パンチのヤング率 [MPa]
E_work = 210e3      # ワークのヤング率 [MPa]
nu_punch = 0.3      # パンチのポアソン比
nu_work = 0.3       # ワークのポアソン比

# 有効弾性率と有効曲率半径の計算
E_star = 1 / ((1 - nu_punch**2) / E_punch + (1 - nu_work**2) / E_work)
R_eff = pr  # パンチ先端の曲率半径

# モード2のパラメータ（塑性変形）
n = 0.2          # ひずみ硬化指数

# モード3のパラメータ（亀裂の発生と進展）
F_max = 100.0e3  # 最大荷重 [N]
beta = 50.0      # モデル定数

# モード4のパラメータ（破断と打ち抜き）
gamma = 100.0    # モデル定数

# モード5のパラメータ（減衰振動）
F_vib = 10.0e3   # 振動の初期振幅 [N]
lambda_decay = 50.0  # 減衰定数 [1/mm]
omega = 200.0        # 振動の角周波数 [rad/mm]
phi = 0.0            # 初期位相 [rad]

# 新しいパラメータ（モード3bのため）
F_min = 20.0e3    # 破断直前の荷重 [N]

# モードの境界変位
d1 = d0 + 0.1                  # 弾性から塑性への移行点 [mm]
d2 = d1 + 0.5                  # 塑性から亀裂発生への移行点 [mm]
d_peak = d2 + 0.1              # 荷重が最大値に達する点 [mm]
d3 = d_peak + 0.5              # 亀裂進展から破断への移行点 [mm] （亀裂がワーク厚さに達する点）
d4 = d3 + 0.1                  # パンチ貫通の開始点 [mm]
d5 = d4 + 0.5                  # 振動が収束する点 [mm]

# 荷重減少率の計算（モード3bのため）
k_lin = (F_max - F_min) / (d3 - d_peak)

# 経験的な係数の計算
f_pr = 1 + k_pr * (pr - pr0)
f_cl = 1 + k_cl * (cl - cl0)
f_dr = 1 + k_dr * (dr - dr0)
f_h = h / h0

# gamma_effとdelta_paramの計算
gamma_eff = gamma * f_cl * f_dr
delta_param = gamma_eff  # モード4の減少率

lambda_eff = lambda_decay * f_dr  # モード5の減衰定数

# 変位の範囲を設定
d = np.linspace(d0, d5 + 0.5, 1000)

# モードごとのプロット表示制御変数
plot_mode1 = True
plot_mode2 = True
plot_mode3 = True
plot_mode4 = True
plot_mode5 = True
plot_mode6 = True

# 各モードの物理式を関数として定義
def mode1(d):
    delta = d - d0
    F = (4/3) * E_star * np.sqrt(R_eff) * delta**(1.5)
    return F

def mode2(d):
    delta = d - d1
    strain = delta / t
    sigma = h * f_pr * strain**n
    F = sigma * A
    return F

def mode3a(d):
    delta = d - d2
    beta_eff = beta * f_cl * f_dr * f_h
    F = F_max * (1 - np.exp(-beta_eff * delta))
    return F

def mode3b(d):
    F = F_max - k_lin * (d - d_peak)
    return F

def mode4(d):
    delta = d - d3
    F_residual = F_min  # 破断直前の荷重を初期値とする
    F = F_residual * np.exp(-delta_param * delta)
    return F

def mode5(d):
    delta = d - d4
    F = F_vib * np.exp(-lambda_eff * delta) * np.sin(omega * delta + phi)
    return F

def mode6(d):
    F = np.zeros_like(d)
    return F

# 各モードの範囲に応じて荷重を計算
F_total = np.zeros_like(d)

if plot_mode1:
    idx1 = np.where((d >= d0) & (d < d1))
    F_total[idx1] = mode1(d[idx1])

if plot_mode2:
    idx2 = np.where((d >= d1) & (d < d2))
    F_total[idx2] = mode2(d[idx2])

if plot_mode3:
    idx3a = np.where((d >= d2) & (d < d_peak))
    F_total[idx3a] = mode3a(d[idx3a])

    idx3b = np.where((d >= d_peak) & (d <= d3))
    F_total[idx3b] = mode3b(d[idx3b])

if plot_mode4:
    idx4 = np.where((d > d3) & (d < d4))
    F_total[idx4] = mode4(d[idx4])

if plot_mode5:
    idx5 = np.where((d >= d4) & (d < d5))
    F_total[idx5] = mode5(d[idx5])

if plot_mode6:
    idx6 = np.where(d >= d5)
    F_total[idx6] = mode6(d[idx6])

# プロット
plt.figure(figsize=(10, 6))

# モードごとに色を変えてプロット
if plot_mode1:
    plt.plot(d[idx1], F_total[idx1], label='Mode 1: Elastic Deformation', color='blue')

if plot_mode2:
    plt.plot(d[idx2], F_total[idx2], label='Mode 2: Plastic Deformation', color='green')

if plot_mode3:
    plt.plot(d[idx3a], F_total[idx3a], label='Mode 3a: Crack Initiation', color='red')
    plt.plot(d[idx3b], F_total[idx3b], label='Mode 3b: Crack Propagation', color='red', linestyle='--')

if plot_mode4:
    plt.plot(d[idx4], F_total[idx4], label='Mode 4: Fracture and Punching', color='orange')

if plot_mode5:
    plt.plot(d[idx5], F_total[idx5], label='Mode 5: Damped Vibration', color='purple')

if plot_mode6:
    plt.plot(d[idx6], F_total[idx6], label='Mode 6: Punch Return', color='black')

plt.xlabel('Punch Displacement d [mm]')
plt.ylabel('Load F [N]')
plt.title('Load-Displacement Curve of the Punching Process')
plt.legend()
plt.grid(True)
plt.show()
