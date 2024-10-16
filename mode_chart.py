import numpy as np
import matplotlib.pyplot as plt

# 定数の定義
E = 210e9  # ヤング率（Pa）
nu = 0.3   # ポアソン比
E_prime = E / (1 - nu**2)  # 有効弾性率（平面ひずみの場合）

# ヘルツ接触理論のための有効弾性率
E_star = E / (2 * (1 - nu**2))

R = 0.01  # 曲率半径（m）
k = 1e8   # ばね定数（N/m）
A_cross_section = 1e-4  # 断面積（m^2）
K_hardening = 1e9       # 硬化係数（N/m^n）
n_hardening = 0.2       # 硬化指数
Y = 1.0                 # 形状係数

# 境界位置の定義
x_12 = 0.001  # m
x_23 = 0.002  # m
x_34 = 0.003  # m
x_45 = 0.004  # m
x_56 = 0.005  # m
x_67 = 0.006  # m
x_78 = 0.007  # m
x_89 = 0.008  # m
x_910 = 0.009 # m
x_max = 0.01  # m

# xの配列を作成
x = np.linspace(0, x_max, 1000)

# F(x)をゼロで初期化
F = np.zeros_like(x)

# 領域1: F(x) = 0
region1 = x <= x_12
F[region1] = 0

# 領域2: ヘルツ接触理論
region2 = (x > x_12) & (x <= x_23)
x_region2 = x[region2] - x_12
F2 = (4/3) * E_star * np.sqrt(R) * x_region2**1.5
F[region2] = F2

# 領域3: フックの法則による弾性変形
region3 = (x > x_23) & (x <= x_34)
x_region3 = x[region3] - x_23
F3 = k * x_region3 + F[region2][-1]
F[region3] = F3

# 領域4: 硬化を考慮した塑性変形
region4 = (x > x_34) & (x <= x_45)
x_region4 = x[region4] - x_34
F4 = A_cross_section * K_hardening * (x_region4 / x_34)**n_hardening + F[region3][-1]
F[region4] = F4

# 領域5: 硬化領域
region5 = (x > x_45) & (x <= x_56)
x_region5 = x[region5] - x_45
F5 = A_cross_section * K_hardening * (x_region5 / x_45)**n_hardening + F[region4][-1]
F[region5] = F5

# 領域6: 亀裂・せん断の進行（モデル1: 応力拡大係数を使用）
region6 = (x > x_56) & (x <= x_67)
x_region6 = x[region6] - x_56
K_x = np.linspace(1e7, 1e6, len(x_region6))  # 応力拡大係数を減少させる
a_x = np.linspace(1e-5, 5e-5, len(x_region6))  # 亀裂長さはそのまま増加
F_model1 = (K_x * np.sqrt(np.pi * a_x)) / Y + F[region5][-1]

# 領域7: エネルギー解放領域（線形変化を仮定）
region7 = (x > x_67) & (x <= x_78)
F[region7] = np.linspace(F_model1[-1], F_model1[-1]*0.8, len(x[region7]))

# 領域8: 残留エネルギー解放（指数関数的減衰）
region8 = (x > x_78) & (x <= x_89)
F[region8] = F[region7][-1] * np.exp(-5 * (x[region8] - x_78) / (x_89 - x_78))

# 領域9: 振動領域（減衰サインカーブ）
region9 = (x > x_89) & (x <= x_910)
omega = 20  # 角周波数を減少
zeta = 0.05  # 減衰比を緩和
F[region9] = F[region8][-1] * np.exp(-zeta * omega * (x[region9] - x_89)) * np.sin(omega * (x[region9] - x_89))

# 領域10: F(x) = 0
region10 = x > x_910
F[region10] = 0

# 各領域の力を分離して描画
plt.figure(figsize=(12, 8))

# 領域1: F(x) = 0
plt.plot(x[region1], F[region1], label='領域1: F(x) = 0')

# 領域2: ヘルツ接触理論
plt.plot(x[region2], F[region2], label='領域2: ヘルツ接触理論')

# 領域3: フックの法則
plt.plot(x[region3], F[region3], label='領域3: フックの法則')

# 領域4: 硬化を考慮した塑性変形
plt.plot(x[region4], F[region4], label='領域4: 硬化塑性変形')

# 領域5: 硬化領域
plt.plot(x[region5], F[region5], label='領域5: 硬化領域')

# 領域6: モデル1（応力拡大係数）
plt.plot(x[region6], F_model1, label='領域6: モデル1（応力拡大係数）')

# 領域7: エネルギー解放領域
plt.plot(x[region7], F[region7], label='領域7: エネルギー解放領域')

# 領域8: 残留エネルギー解放
plt.plot(x[region8], F[region8], label='領域8: 残留エネルギー解放')

# 領域9: 振動領域
plt.plot(x[region9], F[region9], label='領域9: 振動領域')

# 領域10: F(x) = 0
plt.plot(x[region10], F[region10], label='領域10: F(x) = 0')

plt.xlabel('変位 x (m)')
plt.ylabel('力 F(x) (N)')
plt.title('打ち抜き加工における力と変位の関係（モデル1 - 調整後）')
plt.legend()
plt.grid(True)
plt.show()