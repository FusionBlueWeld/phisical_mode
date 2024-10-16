import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
E_star = 210e9  # ヘルツ接触理論の係数
R = 0.05  # 曲率半径
k = 1e6  # フックの法則の弾性係数
A = 1e4  # 硬化係数に関する定数
K = 200  # 硬化係数
n = 0.2  # 硬化係数の指数

# 境界の設定
x12 = 0.01
x23 = 0.02
x34 = 0.03
x45 = 0.04
x56 = 0.05

# 領域ごとのF(x)関数
def F1(x):
    return np.zeros_like(x)

def F2(x):
    return (4/3) * E_star * np.sqrt(R) * (x**(3/2))

def F3(x):
    return k * (x - x23) + F2(x23)

def F4(x):
    return A * K * ((x - x34) / x34)**n + F3(x34)

def F5(x):
    return A * K * ((x - x45) / x45)**n + F4(x45)

# x範囲の設定
x = np.linspace(0, 0.06, 1000)

# 各領域のF(x)を計算
F_values = np.piecewise(x, 
    [x < x12, (x >= x12) & (x < x23), (x >= x23) & (x < x34), (x >= x34) & (x < x45), (x >= x45) & (x < x56)],
    [F1, F2, F3, F4, F5])

# 境界点の描画
boundaries = [x12, x23, x34, x45, x56]

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(x, F_values, label='Force F(x)', color='blue')

# 境界を示す縦線を描画
for b in boundaries:
    plt.axvline(x=b, color='red', linestyle='--', label=f'Boundary x={b:.2f}')

# グラフの装飾
plt.xlabel('x (Displacement)')
plt.ylabel('F(x) (Force)')
plt.title('Force vs Displacement in Punch Process')
plt.grid(True)
plt.show()