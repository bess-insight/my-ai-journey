OK，Jason。下面这版尽量“短句 + 有料 + 结构清晰”。你要的是心智图 + 关键概念 + 常见坑 + 上手示例，核心围绕你在做的 SoC/SoH 序列建模。

# 一、整条技术栈怎么串起来

| 层    | 工具                     | 解决什么问题    | 核心数据结构               | 典型动作               |
| ---- | ---------------------- | --------- | -------------------- | ------------------ |
| 语言   | Python                 | 胶水语言      | list / dict          | 脚本、调用库             |
| 数值底座 | **NumPy**              | 快速数组运算    | `ndarray`            | 线性代数、广播            |
| 表格处理 | **pandas**             | 结构化/时序数据  | `DataFrame/Series`   | 清洗、分组、重采样          |
| 可视化  | Matplotlib             | 画图        | figure/axes          | 曲线、直方图             |
| 传统ML | **scikit-learn**       | 基准模型/特征工程 | `Estimator/Pipeline` | 标准化、CV、基线          |
| 深度学习 | **TensorFlow (Keras)** | 构建/训练NN   | `Tensor/Layer/Model` | LSTM/CNN/优化器       |
| 数据管道 | **tf.data**            | 高效喂数据     | `Dataset`            | map/batch/prefetch |
| 备选   | PyTorch/Lightning      | 另一路深度学习栈  | tensor/module        | 研究友好、灵活            |
| 放大   | Dask/Polars            | 大数据/更快表格  | 表/惰性计算               | 批量/并行              |

> 电池项目常见流程：CSV/Parquet → **pandas** 清洗/对齐 → `numpy`/`tf.data` 组序列 → **TensorFlow/Keras** LSTM 训练 → **Matplotlib** 看损失/预测 → **sklearn** 做基线与对比。

---

# 二、pandas（表格/时序数据的主力）

**它是什么**：基于 NumPy 的表格计算库，擅长 CSV/Parquet、时间索引、分组聚合、缺失值处理。

**核心概念**

* `DataFrame`（二维表）、`Series`（一列）、`Index`（标签/时间戳）。
* `dtype`：数值/分类/时间戳要明确，否则会掉进 `object` 慢坑。
* 向量化、分组（`groupby`）、重采样（`resample`）、窗口（`rolling`）。

**常用操作（够用即好）**

```python
import pandas as pd

df = pd.read_csv("cycles.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# 1) 缺失/异常
df = df.dropna()
df = df[(df['voltage']>2.5) & (df['voltage']<4.3)]

# 2) 重采样到统一步长（比如1s）
df1s = df.resample("1S").mean().interpolate()

# 3) 派生特征
df1s["delta_soc"] = df1s["soc"].diff().fillna(0)
df1s["current_abs"] = df1s["current"].abs()

# 4) 分组（按工况/温度）
g = df1s.groupby("temperature_bucket")["delta_soc"].mean()

# 5) 导出到 numpy（喂给深度学习）
X = df1s[["voltage","current","temperature"]].to_numpy()
y = df1s[["soc"]].to_numpy()
```

**常见坑**

* `SettingWithCopyWarning`：链式索引引发的浅拷贝/视图问题。用 `.loc` 明确写法。
* `object` 类型拖慢一切：及时 `astype` 到数值/分类/时间戳。
* 时区和重采样：先统一时区，再 `resample`，缺失用 `interpolate/ffill`。

**性能要点**

* 列操作 > 行循环；尽量向量化。
* 分类变量用 `Categorical`。
* 大文件用 `read_csv(chunksize=...)` 或改用 Parquet。

---

# 三、TensorFlow / Keras（深度学习主力）

**它是什么**：Google 的深度学习框架，Keras 是其高级 API；当前默认“动态图”易调试，也能图编译加速。

**核心概念**

* `Tensor`（张量）、`Layer`（层）、`Model`（网络）、`Loss`、`Optimizer`、`Metrics`。
* `tf.data.Dataset`：高效输入流水线（map → cache → shuffle → batch → prefetch）。
* 回调：`ModelCheckpoint`、`EarlyStopping`、`TensorBoard`。

**典型：把时序数据打包成 LSTM 输入**

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_windows(X, y, win=100, stride=1):
    # X: (N, F), y: (N, 1) → (M, win, F), (M, win, 1)
    n = (len(X) - win) // stride + 1
    Xw = np.lib.stride_tricks.sliding_window_view(X, (win, X.shape[1]))[::stride,0]
    yw = np.lib.stride_tricks.sliding_window_view(y, (win, y.shape[1]))[::stride,0]
    return Xw[:n], yw[:n]

# X:(N,3), y:(N,1) 来自 pandas.to_numpy()
Xw, yw = make_windows(X, y, win=100)

ds = tf.data.Dataset.from_tensor_slices((Xw, yw))\
     .shuffle(4096).batch(64).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(100, X.shape[1])),
    layers.LSTM(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mae", metrics=["mse"])

ckpt = keras.callbacks.ModelCheckpoint("ckpt.keras",
                                       save_best_only=True, monitor="val_loss")
es = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

history = model.fit(ds.take(800), validation_data=ds.skip(800).take(200),
                    epochs=100, callbacks=[ckpt, es])
```

> 形状心法：**(batch, seq_len, features)**。你日志里出现的 `(12222, 100, 3)` 就是 12,222 个样本、每个样本 100 步、每步 3 个特征。

**常见坑**

* 形状不匹配：`return_sequences=True` 才能做逐步回归（SoC 序列）。
* dtype：混用 `float64`/`float32` 会掉速，统一 `float32`。
* 学习率过大/过小：先 1e-3，曲线抖/不降再调。
* 数据泄露：时间序列要按时间切分训练/验证，避免乱序。

**性能要点**

* `tf.data`：`cache→shuffle→batch→prefetch`。
* GPU 有就用混合精度：`tf.keras.mixed_precision.set_global_policy("mixed_float16")`。
* 频繁磁盘读取 → `TFRecord`/`cache`。

---

# 四、scikit-learn（强力基线/特征工程）

**它是什么**：传统机器学习工具箱。优点是快出基线、管道清晰。

**常用套路**

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), shuffle=False)
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

用途：快速 sanity-check、做非深度学习的 SoH 回归、与 LSTM 做对照（避免“深度学习的错觉提升”）。

---

# 五、这些库在 BESS 场景的分工

1. **数据侧（pandas）**

   * 合并多温度/多工况数据；重采样到统一频率；异常/缺失处理；构造派生特征（dV/dt、I_abs、温度桶）。
2. **样本构造（NumPy + tf.data）**

   * 滑窗成序列（seq_len=100/200）；按“早期周期→训练，后期周期→验证/测试”的时间切分。
3. **建模（TensorFlow/Keras）**

   * LSTM/GRU/TCN 做 SoC；SoH 可用序列 + 统计特征；多任务（SoC+温度）共享骨干。
4. **基线与解释（scikit-learn）**

   * Ridge/XGBoost 基线；`PermutationImportance`/SHAP（可选）看特征贡献。
5. **评估**

   * MAE/RMSE/温区分层误差；不同循环段（充/放/静置）分段评估；OOS 工况泛化。

---

# 六、如何选型（给你一句话版）

* **先 pandas 把数据打干净**；
* 想要“快有结果”→ **sklearn** 基线；
* 需要序列记忆/长依赖 → **TensorFlow(Keras) LSTM/GRU**；
* 偏研究/想自由度更高 → 再考虑 **PyTorch**；
* 数据更大/更快表格 → 看 **Polars/Dask**。

---

# 七、最小可用脚手架（把 CSV 直接喂到 LSTM）

```python
import pandas as pd, numpy as np, tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers

# 1) 读 & 整理
df = pd.read_csv("data.csv", parse_dates=["ts"]).sort_values("ts").set_index("ts")
df = df.dropna()
df = df.resample("1S").mean().interpolate()

feats = ["voltage","current","temperature"]
X = df[feats].astype("float32").to_numpy()
y = df[["soc"]].astype("float32").to_numpy()

# 2) 滑窗
def windows(X, y, win=100, step=1):
    n = (len(X) - win)//step + 1
    Xw = np.stack([X[i:i+win] for i in range(0, n*step, step)])
    yw = np.stack([y[i:i+win] for i in range(0, n*step, step)])
    return Xw, yw

Xw, yw = windows(X, y, win=100)

# 时间序列切分（不洗牌）
n_train = int(len(Xw)*0.7); n_val = int(len(Xw)*0.85)
ds_train = tf.data.Dataset.from_tensor_slices((Xw[:n_train], yw[:n_train]))\
          .batch(64).prefetch(tf.data.AUTOTUNE)
ds_val   = tf.data.Dataset.from_tensor_slices((Xw[n_train:n_val], yw[n_train:n_val]))\
          .batch(64).prefetch(tf.data.AUTOTUNE)

# 3) 模型
model = keras.Sequential([
  layers.Input((100, len(feats))),
  layers.LSTM(128, return_sequences=True),
  layers.TimeDistributed(layers.Dense(1))
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae", metrics=["mse"])
model.fit(ds_train, validation_data=ds_val, epochs=30)
```

---

# 八、易错点清单（你现在的项目最相关）

* 序列维度：**(batch, seq_len, features)**，别混到 `(batch, features, seq_len)`。
* 训练/验证划分必须按**时间**；不要 `shuffle=True` 地把未来信息泄露到过去。
* pandas 重采样后要 **统一频率**，否则窗口步长不一致。
* 电压/电流量纲差异大时，**标准化/归一化**，并把 scaler 与模型一起保存。
* 不同温度/循环的覆盖面要够，否则泛化会“翻车”。

---

需要我把这套内容做成一页 A4 的“口袋卡”（含图示 + 代码模板）吗？你可以直接贴到项目 Wiki 给团队用。
