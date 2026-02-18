"""
能動学習デモ用 2次元データセット生成スクリプト

2つの入力パラメータ (x1, x2) から1つの出力 y を予測する回帰問題を作成する。
真の関数は非線形で、領域によって複雑さが異なるため、
能動学習の効果（どこをサンプリングすべきか）が視覚的にわかりやすい。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data"


def true_function(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """真の関数: 領域によって複雑さが異なる非線形関数"""
    return (
        np.sin(3 * x1) * np.cos(3 * x2)
        + 0.5 * np.sin(6 * x1 * x2)
    )


def generate_grid(n_grid: int = 100) -> dict:
    """評価用の密なグリッドデータを生成"""
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    Y = true_function(X1, X2)
    return {"X1": X1, "X2": X2, "Y": Y}


def generate_initial_samples(n_samples: int = 20, noise_std: float = 0.05) -> dict:
    """初期学習用のランダムサンプルを生成"""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 1, size=(n_samples, 2))
    y_true = true_function(X[:, 0], X[:, 1])
    y = y_true + rng.normal(0, noise_std, size=n_samples)
    return {"X": X, "y": y, "y_true": y_true}


def generate_candidate_pool(n_candidates: int = 500, noise_std: float = 0.05) -> dict:
    """能動学習の候補プール（ラベル未取得）を生成"""
    rng = np.random.default_rng(SEED + 1)
    X = rng.uniform(0, 1, size=(n_candidates, 2))
    y_true = true_function(X[:, 0], X[:, 1])
    y = y_true + rng.normal(0, noise_std, size=n_candidates)
    return {"X": X, "y": y, "y_true": y_true}


def plot_dataset(grid: dict, initial: dict, candidates: dict) -> None:
    """データセットを可視化"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 真の関数の等高線
    ax = axes[0]
    c = ax.contourf(grid["X1"], grid["X2"], grid["Y"], levels=20, cmap="RdBu_r")
    fig.colorbar(c, ax=ax)
    ax.set_title("True Function")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # 初期サンプル
    ax = axes[1]
    ax.contourf(grid["X1"], grid["X2"], grid["Y"], levels=20, cmap="RdBu_r", alpha=0.3)
    ax.scatter(initial["X"][:, 0], initial["X"][:, 1],
               c=initial["y"], cmap="RdBu_r", edgecolors="k", s=60, zorder=5)
    ax.set_title(f"Initial Samples (n={len(initial['y'])})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # 候補プール
    ax = axes[2]
    ax.contourf(grid["X1"], grid["X2"], grid["Y"], levels=20, cmap="RdBu_r", alpha=0.3)
    ax.scatter(candidates["X"][:, 0], candidates["X"][:, 1],
               c="gray", s=5, alpha=0.5, label="Candidates (unlabeled)")
    ax.scatter(initial["X"][:, 0], initial["X"][:, 1],
               c="red", edgecolors="k", s=60, zorder=5, label="Initial samples")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Candidate Pool (n={len(candidates['y'])})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_overview.png'}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    grid = generate_grid(n_grid=100)
    initial = generate_initial_samples(n_samples=20, noise_std=0.05)
    candidates = generate_candidate_pool(n_candidates=500, noise_std=0.05)

    # npz形式で保存
    np.savez(OUTPUT_DIR / "grid.npz", **grid)
    np.savez(OUTPUT_DIR / "initial_samples.npz", **initial)
    np.savez(OUTPUT_DIR / "candidate_pool.npz", **candidates)
    print(f"Initial samples: {initial['X'].shape}")
    print(f"Candidate pool:  {candidates['X'].shape}")
    print(f"Grid:            {grid['X1'].shape}")

    plot_dataset(grid, initial, candidates)


if __name__ == "__main__":
    main()
