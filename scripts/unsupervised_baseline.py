"""Train-only GLCM + LBP / StandardScaler / PCA / K-Means baseline.

Run from any directory: PYTHONNOUSERSITE=1 python3 scripts/unsupervised_baseline.py
"""

import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sklearn
import skimage
from skimage.feature import local_binary_pattern
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # Older scikit-image uses British spelling.
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "unsupervised_baseline"
FIGURES = ROOT / "results" / "figures"
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation")
SEED = 42


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    """Read existing manifests; labels stay separate from numeric features."""
    rows = {}
    for split in ("train", "val", "test"):
        with (ROOT / "splits" / (split + ".csv")).open(newline="", encoding="utf-8") as stream:
            rows[split] = list(csv.DictReader(stream))
        if not rows[split]:
            raise ValueError("Empty split: " + split)
        if any(row["label"] not in CLASSES for row in rows[split]):
            raise ValueError("Unknown class in " + split)
    paths = [row["path"] for values in rows.values() for row in values]
    if len(paths) != len(set(paths)):
        raise ValueError("Overlapping image paths in manifests")
    return rows


def extract_glcm_features(image):
    """72 GLCM statistics with the project's fixed parameters."""
    glcm = graycomatrix(
        image // 8, distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    return np.concatenate([graycoprops(glcm, prop).ravel() for prop in PROPERTIES])


def extract_lbp_features(image):
    """54 normalized multiscale uniform LBP bins."""
    features = []
    for points, radius in ((8, 1), (16, 2), (24, 3)):
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        histogram = np.bincount(lbp.astype(np.int64).ravel(), minlength=points + 2)
        features.append(histogram / histogram.sum())
    return np.concatenate(features)


def extract_features(image):
    return np.concatenate([extract_glcm_features(image), extract_lbp_features(image)])


def feature_matrices(rows):
    matrices, audit = {}, {}
    for split, records in rows.items():
        sizes, modes, channels, formats = Counter(), Counter(), Counter(), Counter()
        feature_rows = []
        for row in records:
            with Image.open(ROOT / row["path"]) as image:
                image.load()  # Fully decode, so corrupt images fail immediately.
                sizes["{}x{}".format(*image.size)] += 1
                modes[image.mode] += 1
                channels[str(len(image.getbands()))] += 1
                formats[image.format or "unknown"] += 1
                gray = image.convert("L")
                if gray.size != (200, 200):
                    gray = gray.resize((200, 200), Image.BILINEAR)
                feature_rows.append(extract_features(np.asarray(gray, dtype=np.uint8)))
        matrices[split] = np.asarray(feature_rows, dtype=np.float64)
        if not np.isfinite(matrices[split]).all():
            raise ValueError("Non-finite features in " + split)
        audit[split] = {
            "images": len(records), "classes": dict(Counter(row["label"] for row in records)),
            "original_sizes": dict(sizes), "original_modes": dict(modes),
            "original_channels": dict(channels), "original_formats": dict(formats),
            "loaded_shape": [200, 200], "loaded_channels": 1,
        }
        print("{}: {} images, features {}".format(split, len(records), matrices[split].shape), flush=True)
    return matrices, audit


def scale_features(features):
    scaler = StandardScaler()
    scaled = {"train": scaler.fit_transform(features["train"])}
    for split in ("val", "test"):
        scaled[split] = scaler.transform(features[split])
    return scaler, scaled


def apply_pca(scaled):
    pca = PCA(n_components=0.95, svd_solver="full", random_state=SEED)
    projected = {"train": pca.fit_transform(scaled["train"])}
    for split in ("val", "test"):
        projected[split] = pca.transform(scaled[split])
    return pca, projected


def train_kmeans(train_features):
    kmeans = KMeans(n_clusters=6, random_state=SEED, n_init=10, max_iter=300)
    kmeans.fit(train_features)
    return kmeans


def evaluate_clustering(features, clusters, labels):
    count = len(np.unique(clusters))
    silhouette = float(silhouette_score(features, clusters)) if 1 < count < len(features) else None
    return {
        "silhouette": silhouette,
        "silhouette_note": "Computed in retained PCA space; null if fewer than two occupied clusters.",
        "ARI": float(adjusted_rand_score(labels, clusters)),
        "NMI": float(normalized_mutual_info_score(labels, clusters)),
        "occupied_clusters": count,
        "cluster_counts": {str(k): int(v) for k, v in Counter(clusters).items()},
    }


def visualize_clusters(val_projected, pca, clusters, labels):
    """Use the first two components of the existing training-fitted PCA."""
    coordinates = val_projected[:, :2]
    if coordinates.shape[1] != 2:
        raise ValueError("The fitted PCA must retain at least two components for plotting")
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    for name, values, categories, title in (
        ("validation_clusters.png", clusters, range(6), "Validation: K-Means clusters"),
        ("validation_classes.png", labels, CLASSES, "Validation: ground-truth defect classes"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, category in enumerate(categories):
            mask = values == category
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=35,
                       alpha=0.8, color=colors(index), label=str(category))
        variance = pca.explained_variance_ratio_
        ax.set_xlabel("PC1 ({:.1%} variance)".format(variance[0]))
        ax.set_ylabel("PC2 ({:.1%} variance)".format(variance[1]))
        ax.set_title(title + "\nFirst two components of training-fitted PCA")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    rows = load_data()
    # Fingerprint all raw data and manifests before and after execution.
    inputs = sorted(path for path in (ROOT / "data").rglob("*") if path.is_file())
    inputs += [ROOT / "splits" / (split + ".csv") for split in rows]
    before = {str(path.relative_to(ROOT)): file_hash(path) for path in inputs}
    features, audit = feature_matrices(rows)

    scaler, scaled = scale_features(features)
    pca, projected = apply_pca(scaled)
    kmeans = train_kmeans(projected["train"])
    val_clusters = kmeans.predict(projected["val"])
    # Labels enter only external evaluation and the ground-truth plot.
    val_labels = np.array([row["label"] for row in rows["val"]])
    metrics = evaluate_clustering(projected["val"], val_clusters, val_labels)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    visualize_clusters(projected["val"], pca, val_clusters, val_labels)

    after = {str(path.relative_to(ROOT)): file_hash(path) for path in inputs}
    if before != after:
        raise RuntimeError("Input files changed during execution")
    summary = {
        "random_state": SEED, "dataset": audit,
        "feature_parameters": {"GLCM": {"levels": 32, "distances": [1, 2, 4],
            "angles_degrees": [0, 45, 90, 135], "properties": list(PROPERTIES), "dimensions": 72},
            "LBP": {"method": "uniform", "points_radius": [[8, 1], [16, 2], [24, 3]],
                    "histogram_normalization": "L1 per scale", "dimensions": 54}},
        "feature_shapes": {split: list(matrix.shape) for split, matrix in features.items()},
        "pca_dimensions": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "projected_shapes": {split: list(matrix.shape) for split, matrix in projected.items()},
        "validation_metrics": metrics,
        "original_dimension": int(features["train"].shape[1]),
        "visualization_variance": pca.explained_variance_ratio_[:2].tolist(),
        "visualization_method": "First two components of the existing training-fitted PCA.",
        "fit_scope": "Training features only for scaler, PCA and K-Means.",
        "test_scope": "Integrity audit, extraction and transform only; no test metrics or predictions.",
        "input_files_unchanged": before == after,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "input_hashes.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    joblib.dump({"scaler": scaler, "pca": pca, "kmeans": kmeans}, OUTPUT / "baseline.joblib")
    np.savez_compressed(OUTPUT / "features.npz", **features)
    with (OUTPUT / "validation_clusters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "cluster"])
        writer.writerows((row["path"], row["label"], int(cluster))
                         for row, cluster in zip(rows["val"], val_clusters))
    print("\nDataset Summary")
    for split, inspection in audit.items():
        print(split, json.dumps(inspection))
    print("\nFeature Extraction: GLCM=72, LBP=54, Combined=126")
    for split, matrix in features.items():
        print(split, matrix.shape)
    print("\nPCA original dimension: 126")
    print("PCA dimensions: {}; explained variance: {:.6f}".format(
        pca.n_components_, pca.explained_variance_ratio_.sum()))
    print("Validation metrics:", json.dumps(metrics, indent=2))
    print("Visualization:", FIGURES / "validation_clusters.png", FIGURES / "validation_classes.png")
    print("Input files unchanged:", before == after)
    print("Outputs:", OUTPUT)


if __name__ == "__main__":
    main()
