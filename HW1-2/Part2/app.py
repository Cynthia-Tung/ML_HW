import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

st.set_page_config(layout="wide")
st.title("Image Quantization")

# ================= 讀圖 =================
img = np.array(Image.open("flower.jpg"))
h, w, c = img.shape

# ================= Functions =================
def quantize_image(img, k):
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_

    distortion = np.sum((pixels - centroids[labels])**2)

    new_pixels = centroids[labels].astype(np.uint8)
    new_img = new_pixels.reshape(img.shape)

    return new_img, distortion, centroids


def nubs(img, target_k):
    clusters = [img.reshape(-1,3)]

    while len(clusters) < target_k:
        variances = [np.var(c) for c in clusters]
        idx = np.argmax(variances)

        data = clusters.pop(idx)

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(data)

        clusters.append(data[labels == 0])
        clusters.append(data[labels == 1])

    centroids = np.array([c.mean(axis=0) for c in clusters])

    pixels = img.reshape(-1,3)
    distances = np.linalg.norm(pixels[:,None] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    new_pixels = centroids[labels].astype(np.uint8)
    new_img = new_pixels.reshape(img.shape)

    distortion = np.sum((pixels - centroids[labels])**2)

    return new_img, distortion


# ================= UI =================
left, right = st.columns([1, 1.5])

# ===== 左側 =====
with left:
    st.header("Controls")

    k = st.selectbox("Choose K", [2,4,8,16,32])
    
    show_rgb = st.checkbox("Show RGB Scatter", False)
    show_elbow = st.checkbox("Show Elbow Plot", False)
    show_nubs = st.checkbox("Show NUBS (K=8)", False)
    
    # ===== NUBS =====
    if show_nubs:
        nubs_img, nubs_d = nubs(img, 8)

        fig6, ax6 = plt.subplots(figsize=(4,4))
        ax6.imshow(nubs_img)
        ax6.set_title("NUBS (K=8)")
        ax6.axis('off')

        st.pyplot(fig6)
        plt.close(fig6)

        st.write(f"NUBS Distortion: {nubs_d:.2e}")
    
    # ===== Elbow Plot =====
    if show_elbow:
        Ks = range(1,17)
        distortions = []

        for k_val in Ks:
            _, d_val, _ = quantize_image(img, k_val)
            distortions.append(d_val)

        fig5, ax5 = plt.subplots(figsize=(5,4))
        ax5.plot(Ks, distortions, marker='o')
        ax5.set_xlabel("K")
        ax5.set_ylabel("Distortion")
        ax5.set_title("Elbow Plot")
        ax5.grid()

        st.pyplot(fig5)
        plt.close(fig5)
    
    # ===== Comparison RGB Scatter =====
    if show_rgb:
        col_orig_rgb, col_k_rgb = st.columns(2)
        
        # ===== RGB Scatter =====
        with col_orig_rgb:
            pixels = img.reshape(-1,3)
            idx = np.random.choice(len(pixels), 5000, replace=False)
            sample = pixels[idx]

            fig4 = plt.figure(figsize=(5,4))
            ax4 = fig4.add_subplot(111, projection='3d')

            ax4.scatter(sample[:,0], sample[:,1], sample[:,2],
                        c=sample/255.0, s=2)

            ax4.set_xlabel("R")
            ax4.set_ylabel("G")
            ax4.set_zlabel("B")
            ax4.set_title("Original")

            st.pyplot(fig4)
            plt.close(fig4)
        
        # ===== RGB Scatter with k-means =====
        with col_k_rgb:
            pixels = img.reshape(-1,3)

            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(pixels)
            centroids = kmeans.cluster_centers_

            sample_idx = np.random.choice(len(pixels), 5000, replace=False)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                pixels[sample_idx,0],
                pixels[sample_idx,1],
                pixels[sample_idx,2],
                c=centroids[labels][sample_idx]/255,
                s=2
            )
            ax.set_title(f"K={k}")

            st.pyplot(fig)
            plt.close(fig)

# ===== 右側 =====
with right:
    st.header("Visualization")

    # ===== 原圖 vs 壓縮圖 =====
    new_img, d, centroids = quantize_image(img, k)
    
    col_orig_img, col_k_img = st.columns(2)
    
    st.write(f"Image: flower | Image size: {w} x {h}")
    
    # ===== 原圖 =====
    with col_orig_img:
        fig1, ax1 = plt.subplots(figsize=(4,4))
        ax1.imshow(img)
        ax1.set_title("Original")
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)

    # ===== 壓縮圖 =====
    with col_k_img:
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(new_img)
        ax2.set_title(f"K={k}")
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)

    st.write(f"Distortion: {d:.2e}")
    
    # ===== Color Palette =====
    fig3, ax3 = plt.subplots(figsize=(6,1))
    for i, color in enumerate(centroids):
        ax3.add_patch(plt.Rectangle((i,0), 1, 1, color=color/255))
    ax3.set_xlim(0, k)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title("Color Palette")
    st.pyplot(fig3)
    plt.close(fig3)
    
    
    
    
    
    
    