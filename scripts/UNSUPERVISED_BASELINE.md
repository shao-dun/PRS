# Unsupervised baseline

在项目目录运行：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/unsupervised_baseline.py
```

`PYTHONNOUSERSITE=1` 用于隔离当前机器用户目录中的不兼容 Python 包。

流程：现有 split → 灰度读取 → GLCM + LBP → StandardScaler → PCA → K-Means → validation evaluation → visualization。

- 输入来自 `splits/train.csv`、`splits/val.csv` 和 `splits/test.csv`，不重新划分。
- 图像读取为 uint8 单通道 200×200；如果原尺寸不同，仅在内存中 resize，不写回图片。
- GLCM：32 灰度级，距离 1/2/4，角度 0/45/90/135 度，六种统计量，共 72 维。
- Uniform LBP：三个尺度 (P,R)=(8,1)/(16,2)/(24,3)，每尺度直方图 L1 归一化，共 54 维。
- 拼接特征共 126 维。Scaler、保留至少 95% 方差的 PCA、K=6 K-Means 均只在训练特征上 fit。
- K-Means 固定 random_state=42、n_init=10、max_iter=300，不进行参数搜索。
- Silhouette 在验证集的保留 PCA 空间计算；ARI/NMI 使用验证标签，仅作为外部聚类评价。这些指标不是分类准确率。
- 可视化直接取已在训练集拟合的 PCA representation 的前两个分量；两个图使用相同坐标，不额外拟合 PCA。
- Test 仅检查、提取特征和 transform，不进行测试集预测或评价。
- 运行前后校验所有原始数据和 split 文件的 SHA-256，确认输入文件未改变。

输出位于 `results/unsupervised_baseline/`：

| 文件 | 内容 |
|---|---|
| summary.json | 数据审计、特征 shape、PCA 维度/方差、validation 指标、软件版本 |
| ../figures/validation_clusters.png | 按 cluster 着色的 validation PCA 图 |
| ../figures/validation_classes.png | 按真实类别着色的 validation PCA 图 |
| validation_clusters.csv | 验证集路径、真实标签、cluster ID |
| features.npz | 按 CSV 行顺序保存的三个原始特征矩阵 |
| baseline.joblib | 训练拟合的 scaler、PCA 和 K-Means |
| input_hashes.json | 输入文件 SHA-256 哈希 |

Cluster ID 为任意编号，不等同于缺陷类别。模型文件应在记录的软件环境中加载。
