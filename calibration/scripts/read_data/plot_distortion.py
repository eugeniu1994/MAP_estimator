import numpy as np
import matplotlib.pyplot as plt

# Example distortion parameters
k1, k2, k3 = -0.00182508, 0.00917698, 0

# --- 1D Plot: distortion factor vs radius ---
r = np.linspace(0, 1, 500)   # normalized radius
distortion_factor = 1 + k1*r**2 + k2*r**4 + k3*r**6

plt.figure(figsize=(6,4))
plt.plot(r, distortion_factor)
plt.title("Radial Distortion Factor vs Normalized Radius")
plt.xlabel("Normalized radius r")
plt.ylabel("Distortion factor (1 + k1*r^2 + k2*r^4 + k3*r^6)")
plt.grid(True)
plt.draw()

# --- 2D Plot: distorted grid in normalized camera coordinates ---
grid_size = 21
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

R2 = X**2 + Y**2
dist_factor = 1 + k1*R2 + k2*R2**2 + k3*R2**3

X_dist = X * dist_factor
Y_dist = Y * dist_factor

plt.figure(figsize=(6,6))
plt.plot(X, Y, 'k.', alpha=0.3)            # original grid
plt.plot(X_dist, Y_dist, 'r.', alpha=0.6)  # distorted grid
plt.title("2D Radial Distortion Effect")
plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.axis('equal')
plt.grid(True)
plt.show()
