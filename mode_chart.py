# コード全体の表示

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# パラメータの設定
E_star = 200  # 合成ヤング率（初期値、単位は GPa）
R = 0.01  # 相対曲率半径（単位は m）
x23 = 0.0003  # 境界 x23 の値
x34 = 0.0006  # 境界 x34 の値

# 変位 (x) の範囲を設定
x = np.linspace(0, 0.001, 1000)  # 0から1mmまでの変位

# 2. ヘルツ接触理論の荷重の傾き (x < x23)
def hertz_slope_at_x1(E_star, R, x1):
    return (2/3) * (4/3) * E_star * np.sqrt(R) * (x1 ** (1/2))

# 3. フックの法則による弾性変形の傾き (x >= x23)
def elastic_slope_at_x1(k):
    return k

# 目的関数: 2.と3.の傾きの差を少数5桁まで一致させる
def objective_fine(params, x23):
    E_star, k = params
    return abs(hertz_slope_at_x1(E_star, R, x23) - elastic_slope_at_x1(k))

# 初期パラメータの設定
initial_guess_fine = [200, 100000]  # E_star, k の初期値

# 最適化実行（少数5桁まで）
result_fine = minimize(objective_fine, initial_guess_fine, args=(x23,), method='Nelder-Mead', options={'xatol': 1e-5})

# 最適化結果の取得
E_star_opt_fine, k_opt_fine = result_fine.x

# 2.のヘルツ接触理論による荷重 F2 (x < x23)
F2_hertz_2_opt_fine = (4/3) * E_star_opt_fine * np.sqrt(R) * (x ** (3/2))

# 3. の弾性変形領域 F3 (x >= x23) の計算
F3_elastic_3_opt_fine = k_opt_fine * (x - x23)

# x23 における 2. の領域での F2 の値を取得
F2_x23_hertz_fine = (4/3) * E_star_opt_fine * np.sqrt(R) * (x23 ** (3/2))

# 3. の領域 (弾性変形) F3 をオフセットして滑らかに接続
F3_elastic_3_offset_fine = F3_elastic_3_opt_fine + F2_x23_hertz_fine

# 4. の塑性変形領域追加
def objective_fine_plastic(params, x34):
    k_hardened = params[0]
    return abs(k_hardened - elastic_slope_at_x1(k_opt_fine))  # 傾きの差を最小化

# 初期パラメータの設定
initial_guess_plastic = [100000]  # k_hardened の初期値

# 最適化実行（少数5桁まで）
result_plastic = minimize(objective_fine_plastic, initial_guess_plastic, args=(x34,), method='Nelder-Mead', options={'xatol': 1e-5})

# 最適化結果の取得
k_hardened_opt = result_plastic.x[0]

# 4.の塑性変形領域 F4 (x >= x34) の計算
F4_plastic_4_opt = k_hardened_opt * (x - x34)

# x34 における 3. の領域での F3 の値を取得
F3_x34_elastic_fine = k_opt_fine * (x34 - x23) + F2_x23_hertz_fine

# 4. の領域 (塑性変形) F4 をオフセットして滑らかに接続
F4_plastic_4_offset_fine = F4_plastic_4_opt + F3_x34_elastic_fine

# グラフ描画の準備
plt.figure(figsize=(8, 6))

# 2. のヘルツ接触理論 F2 (x < x23) の描画
plt.plot(x[x < x23] * 1000, F2_hertz_2_opt_fine[x < x23], label=f"Initial Contact (Hertz Theory, E*={E_star_opt_fine:.5f})")

# 3. の弾性変形領域 F3 (x23 <= x < x34, オフセット後) の描画
plt.plot(x[(x >= x23) & (x < x34)] * 1000, F3_elastic_3_offset_fine[(x >= x23) & (x < x34)], label=f"Elastic Deformation (Offset, k={k_opt_fine:.5f})")

# 4. の塑性変形領域 F4 (x >= x34, オフセット後) の描画
plt.plot(x[x >= x34] * 1000, F4_plastic_4_offset_fine[x >= x34], label=f"Plastic Deformation (Offset, k_hardened={k_hardened_opt:.5f})")

# グラフの設定
plt.title("Force vs Displacement with Smooth Connections (Hertz, Elastic, Plastic)")
plt.xlabel("Displacement (mm)")
plt.ylabel("Force (N)")
plt.axvline(x=x23 * 1000, color='r', linestyle='--', label='Boundary at $x_{23}$')
plt.axvline(x=x34 * 1000, color='g', linestyle='--', label='Boundary at $x_{34}$')
plt.grid(True)
plt.legend()
plt.show()

# 最適化されたパラメータの出力
E_star_opt_fine, k_opt_fine, k_hardened_opt