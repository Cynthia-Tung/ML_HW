import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

# load data
data = np.loadtxt('data1.txt')          # shape (N, 2)
print(f"Data shape: {data.shape}")

# 4 components, kmeans init, full covariance
gmm = GaussianMixture(n_components=4, covariance_type='full',
                      init_params='kmeans', random_state=42)
gmm.fit(data)

means       = gmm.means_          # (4, 2)
covariances = gmm.covariances_    # (4, 2, 2)
weights     = gmm.weights_        # (4,)

print("\n=== Estimated Parameters ===")
for k in range(4):
    print(f"\nComponent {k+1}:")
    print(f"  Weight : {weights[k]:.4f}")
    print(f"  Mean   : {means[k]}")
    print(f"  Cov    :\n{covariances[k]}")

# log likelihood
total_ll = gmm.score(data) * len(data)
print(f"\nTotal log-likelihood : {total_ll:.4f}")
print(f"Converged in {gmm.n_iter_} iterations")
print(f"BIC: {gmm.bic(data):.4f}  |  AIC: {gmm.aic(data):.4f}")

# drawing grid for contour plot
margin = 1.0
xmin, xmax = data[:, 0].min() - margin, data[:, 0].max() + margin
ymin, ymax = data[:, 1].min() - margin, data[:, 1].max() + margin

xx, yy  = np.meshgrid(np.linspace(xmin, xmax, 300),
                      np.linspace(ymin, ymax, 300))
grid    = np.c_[xx.ravel(), yy.ravel()]          # (300*300, 2)
Z       = np.exp(gmm.score_samples(grid)).reshape(xx.shape)  # density

# figure 1: scatter plot with component colors
labels = gmm.predict(data)

fig1, ax1 = plt.subplots(figsize=(7, 5))
scatter = ax1.scatter(data[:, 0], data[:, 1],
            c=labels, cmap='tab10', s=15, alpha=0.7, zorder=2)
ax1.scatter(means[:, 0], means[:, 1],
            marker='X', s=200, c='red', edgecolors='black',
            linewidths=0.8, label='Component means', zorder=3)

ax1.set_title('GMM Component Assignments (K=4)')
ax1.set_xlabel('X1'); ax1.set_ylabel('X2')
plt.colorbar(scatter, ax=ax1, label='Component')
ax1.legend()
plt.tight_layout()
plt.savefig('fig1_scatter.png', dpi=150)
plt.show()

# figure 2: contour plot (with cluster colors)
fig2, ax2 = plt.subplots(figsize=(7, 5))

# background color: cluster results
ax2.scatter(data[:, 0], data[:, 1],
            c=labels, cmap='tab10', s=10, alpha=0.4, zorder=1)

# contour lines: density distribution
contour_filled = ax2.contourf(xx, yy, Z, levels=20,
                               cmap='Blues', alpha=0.55, zorder=2)
contour_lines  = ax2.contour(xx, yy, Z, levels=20,
                              colors='navy', linewidths=0.6, alpha=0.7, zorder=3)

# mark means
ax2.scatter(means[:, 0], means[:, 1],
            marker='X', s=200, c='red', edgecolors='black',
            linewidths=0.8, zorder=4, label='Component means')

plt.colorbar(contour_filled, ax=ax2, label='Density')
ax2.set_title('GMM Density Contours (K=4)')
ax2.set_xlabel('X1'); ax2.set_ylabel('X2')
ax2.legend()
plt.tight_layout()
plt.savefig('fig2_contour.png', dpi=150)
plt.show()

# figure 3: 3D density
fig3  = plt.figure(figsize=(9, 6))
ax3   = fig3.add_subplot(111, projection='3d')
surf  = ax3.plot_surface(xx, yy, Z,
                         cmap='viridis', linewidth=0, antialiased=True, alpha=0.85)
fig3.colorbar(surf, ax=ax3, shrink=0.5, label='Density')
ax3.set_title('3D Probability Density of Fitted GMM (K=4)')
ax3.set_xlabel('X1'); ax3.set_ylabel('X2'); ax3.set_zlabel('Density')
plt.tight_layout()
plt.savefig('fig3_3d_density.png', dpi=150)
plt.show()

# different n_components
bic_scores, aic_scores = [], []
n_range = [2, 3, 4, 5, 6]

for n in n_range:
    g = GaussianMixture(n_components=n, covariance_type='full',
                        init_params='kmeans', random_state=42)
    g.fit(data)
    bic_scores.append(g.bic(data))
    aic_scores.append(g.aic(data))

best_n_bic = n_range[np.argmin(bic_scores)]
best_n_aic = n_range[np.argmin(aic_scores)]
print(f"\nBest n_components by BIC: {best_n_bic}")
print(f"Best n_components by AIC: {best_n_aic}")

# different covariance_type
cov_types   = ['full', 'tied', 'diag', 'spherical']
cov_bic     = {}
cov_aic     = {}

print("\n=== Covariance Type Comparison (K=4) ===")
for cov_type in cov_types:
    g = GaussianMixture(n_components=4, covariance_type=cov_type,
                        init_params='kmeans', random_state=42)
    g.fit(data)
    cov_bic[cov_type] = g.bic(data)
    cov_aic[cov_type] = g.aic(data)
    print(f"  {cov_type:10s}  BIC={g.bic(data):.2f}  AIC={g.aic(data):.2f}")

# figure 4: BIC / AIC vs n_components
fig4, axes = plt.subplots(1, 2, figsize=(13, 5))

# left：n_components
ax_n = axes[0]
ax_n.plot(n_range, bic_scores, 'o-', color='steelblue', label='BIC')
ax_n.plot(n_range, aic_scores, 's--', color='darkorange', label='AIC')
ax_n.axvline(best_n_bic, color='steelblue', linestyle=':', alpha=0.7,
             label=f'Best BIC n={best_n_bic}')
ax_n.set_xlabel('Number of Components')
ax_n.set_ylabel('Score')
ax_n.set_title('Model Selection: n_components')
ax_n.legend(); ax_n.grid(True, alpha=0.3)

# right：covariance_type
ax_c = axes[1]
x_pos = np.arange(len(cov_types))
ax_c.bar(x_pos - 0.2, [cov_bic[c] for c in cov_types], 0.4,
         label='BIC', color='steelblue', alpha=0.8)
ax_c.bar(x_pos + 0.2, [cov_aic[c] for c in cov_types], 0.4,
         label='AIC', color='darkorange', alpha=0.8)
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(cov_types)
ax_c.set_ylabel('Score')
ax_c.set_title('Model Selection: covariance_type (K=4)')
ax_c.legend(); ax_c.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig4_model_selection.png', dpi=150)
plt.show()

print("\nDone. Figures saved: fig1_scatter.png, fig2_contour.png, "
      "fig3_3d_density.png, fig4_model_selection.png")