import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 確保 3D 繪圖功能正常

# input data
try:
    data = np.loadtxt('Data_Exercise_2.txt')
except FileNotFoundError:
    print("Data_Exercise_2.txt not found.")
    exit()

# 2. 設定初始質心 Init (a)
init_a = np.array([
    [0.0, -0.3, -2.0],   # m1 (blue)
    [-1.3, 1.5, 4.0],    # m2 (black)
    [11.3, 3.0, 0.2],  # m3 (red)
    [5.7, 3.0, -2.0],    # m4 (green)
    [10.0, -1.0, 1.2]   # m5 (magenta)
])

colors = ['blue', 'black', 'red', 'green', 'magenta']

# functions
def compute_distortion(data, centroids, labels):
    distortion = 0
    for k in range(len(centroids)):
        cluster_points = data[labels == k]
        if len(cluster_points) > 0:
            # || xi - ci ||^2
            distortion += np.sum(np.linalg.norm(cluster_points - centroids[k], axis=1)**2)
    return distortion

def kmeans_b(data, initial_centroids):
    centroids = initial_centroids.copy()
    history_d = []
    labels = np.zeros(len(data))
    
    iteration = 0
    prev_centroids = None
    
    while True:
        # --- E-step: Assign points to nearest centroid ---
        # 計算每個點到所有質心的距離
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 紀錄 E-step 後的 Distortion
        d_e = compute_distortion(data, centroids, labels)
        history_d.append(d_e)
        
        # --- M-step: Update centroids ---
        new_centroids = np.array([data[labels == k].mean(axis=0) if len(data[labels == k]) > 0 
                                  else centroids[k] for k in range(len(centroids))])
        
        # 紀錄 M-step 後的 Distortion
        d_m = compute_distortion(data, new_centroids, labels)
        history_d.append(d_m)
        
        iteration += 1
        print(f"Iteration {iteration}: Distortion = {d_m:.4f}")
        
        # 檢查收斂 (質心不再變動)
        if prev_centroids is not None and np.allclose(centroids, new_centroids):
            break
            
        prev_centroids = centroids.copy()
        centroids = new_centroids
        
    return centroids, labels, history_d, iteration

# 執行算法
final_centroids, final_labels, distortion_history, total_iters = kmeans_b(data, init_a)

# --- 繪圖 1: Distortion Curve ---
plt.figure(figsize=(10, 5))
steps = np.arange(len(distortion_history))
plt.plot(steps, distortion_history, marker='o', linestyle='-', color='b')
# 標註 E-step (偶數索引) 與 M-step (奇數索引)
for i in range(len(distortion_history)):
    label = 'E' if i % 2 == 0 else 'M'
    plt.text(i, distortion_history[i], label, verticalalignment='bottom', fontsize=9)

plt.title("Distortion Curve - Initialization (b)")
plt.xlabel("Step Index (E-steps and M-steps)")
plt.ylabel("Total Distortion (D)")
plt.grid(True)
plt.show()

# --- 繪圖 2: 3-D Scatter Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for k in range(len(final_centroids)):
    cluster_data = data[final_labels == k]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], 
               c=colors[k], label=f'Cluster {k+1}', s=1, alpha=0.3)
    # 標記最終質心 (星星記號)
    ax.scatter(final_centroids[k, 0], final_centroids[k, 1], final_centroids[k, 2], 
               c=colors[k], marker='*', s=200, edgecolors='white', linewidth=1.5)

ax.set_title(f"Final Clustering (b) - {total_iters} Iterations")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

print(f"Final Distortion: {distortion_history[-1]}")
print(f"Converged in {total_iters} iterations.")