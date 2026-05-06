"""
原型向量可视化：t-SNE 降维 + 余弦相似度热力图
用法：python plot_prototypes.py --path ./results/FullModel_prototypes.npy
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def main(path):
    M = np.load(path)  # (K, D)
    K, D = M.shape
    print(f"Prototypes: K={K}, D={D}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左图：t-SNE 降维散点 ----
    ax = axes[0]
    perp = min(5, K - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(M)
    ax.scatter(coords[:, 0], coords[:, 1], c=range(K), cmap='tab10',
               s=120, edgecolors='k', linewidths=0.5, zorder=3)
    for i in range(K):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]),
                    fontsize=8, ha='center', va='bottom', fontweight='bold')
    ax.set_title('Prototype t-SNE Embedding', fontsize=13)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.3)

    # ---- 右图：余弦相似度热力图 ----
    ax = axes[1]
    sim = cosine_similarity(M)  # (K, K)
    im = ax.imshow(sim, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xlabel('Prototype Index')
    ax.set_ylabel('Prototype Index')
    ax.set_title('Pairwise Cosine Similarity', fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 打印统计
    off_diag = sim[np.triu_indices(K, k=1)]
    print(f"Off-diagonal cosine similarity: mean={off_diag.mean():.4f}, "
          f"max={off_diag.max():.4f}, min={off_diag.min():.4f}")

    plt.tight_layout()
    out = path.replace('.npy', '_vis.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,default='/home/ubuntu/wyh/cicdis/secondPaper/script/results/FullModel_prototypes.npy')
    main(parser.parse_args().path)