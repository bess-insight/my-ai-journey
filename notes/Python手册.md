明白了，Jason 👍。我会结合你目前的基础和目标（跑通 AI 项目 → 看懂代码 → 能改参数/函数 → 形成管理者视角），整理一份 **“Jason 专属 Python 语法手册”**。
它不会像教材那样从零开始，而是挑出 **AI 项目里高频、必须会的部分**，每条都配**示例代码 + 白话解释**。

---

# 📘 Jason 专属 Python 语法手册

## 1. 函数（function）

### 定义和默认参数

```python
def add(a, b=1):
    return a + b

print(add(5))      # 没传 b，默认 b=1 → 输出 6
print(add(5, 10))  # 显式传 b=10 → 输出 15
```

* `b=1` 表示默认参数，不写时自动用。
* 项目里常见：`normalize=True`、`output_capacity=False`，就是可选开关。

---

## 2. 类（class）和 `self`

```python
class Battery:
    def __init__(self, capacity):
        self.capacity = capacity  # 保存属性

    def discharge(self, amount):
        self.capacity -= amount   # 用 self 访问对象的属性
        return self.capacity

cell = Battery(100)
print(cell.discharge(20))  # 输出 80
```

* `__init__` → 初始化函数（创建对象时自动执行）。
* `self` → 指代“这个对象自己”。
* AI 项目里的 `LgData()` 就是这样：`self.path` 保存数据路径。

---

## 3. 列表（list）和切片

```python
nums = [10, 20, 30, 40, 50]

print(nums[0])      # 第一个元素 → 10
print(nums[-1])     # 最后一个元素 → 50
print(nums[1:4])    # 切片，取索引1到3 → [20, 30, 40]

nums.append(60)     # 加一个元素
print(nums)         # [10,20,30,40,50,60]
```

* 列表是最常见的数据结构。
* 在 AI 项目里，`train_names = ['25degC/551_LA92', '25degC/551_Mixed1']` 就是一个字符串列表。

---

## 4. 字典（dict）

```python
info = {"voltage": 3.7, "current": 2.0}
print(info["voltage"])  # 3.7

info["temperature"] = 25
print(info)  # {"voltage":3.7, "current":2.0, "temperature":25}
```

* 类似 Excel 的“键-值”映射。
* 项目里常见：配置参数、metrics 结果、history.history（保存训练曲线）。

---

## 5. Numpy（数组和矩阵操作）

```python
import numpy as np

a = np.array([1, 2, 3, 4])
print(a.shape)         # (4,)

b = a.reshape(2, 2)
print(b)
# [[1 2]
#  [3 4]]

print(b[:, 0])         # 取所有行的第0列 → [1, 3]
print(np.mean(a))      # 平均值 → 2.5
```

* **重点**：Numpy 的 `.shape` 就是“维度”，LSTM 输入常见 `(samples, steps, features)`。
* 所以 `train_x.shape = (12222, 100, 3)` 意思是：

  * 12222 个样本
  * 每个样本 100 个时间步
  * 每个时间步有 3 个特征（V/I/T）。

---

## 6. Pandas（数据表）

```python
import pandas as pd

df = pd.read_csv("data.csv")  # 读 CSV 文件
print(df.head())              # 显示前5行

print(df["Voltage"])          # 取某一列
df = df[df["Status"] == "DCH"]  # 过滤行：只要放电状态
```

* 项目里 `cycle = pd.read_csv(..., skiprows=30)` 就是读实验数据表。

---

## 7. 循环和条件

```python
for i in range(3):
    print("step", i)

x = 5
if x > 0:
    print("positive")
else:
    print("not positive")
```

* `for` → 遍历序列。
* `if/else` → 条件判断。
* 在项目里：循环遍历 cycle 文件，if 判断 NaN。

---

## 8. 导入模块和路径

```python
import os, sys
sys.path.append("../../")  # 把上两级目录加到搜索路径
from data_processing.lg_dataset import LgData
```

* `sys.path.append` → 为了能导入自定义模块。
* 在 notebook 里，这样才能用 `LgData()`。

---

## 9. 错误处理

```python
try:
    val = int("abc")  # 会报错
except ValueError as e:
    print("出错了:", e)
```

* `try/except` 捕获错误，避免程序崩溃。
* 在项目里，常见用法是：发现 NaN 就打印 warning，而不是让程序挂掉。

---

## 10. 常用工具（AI 项目高频）

### 打印日志

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Data loaded successfully")
```

### 保存/加载模型

```python
from tensorflow.keras.models import load_model
model.save("best_model.h5")
model = load_model("best_model.h5")
```

### 回调（callbacks）

```python
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5)
```

---

# 🎯 Jason 要记住的核心

你 **不用全盘掌握 Python 的所有语法**，只要掌握：

1. **函数/类**（能看懂参数、返回值、`self`）
2. **Numpy/Pandas**（能操作矩阵和表格）
3. **路径和导入**（能在 notebook 里加载自己的模块）
4. **循环/条件**（能跟随逻辑读懂代码）
5. **训练过程常用写法**（model.save, callbacks, logging）

这 20% 的内容覆盖你项目里 80% 的实际需求。

---

要不要我帮你把这份手册**整理成 Markdown 文件**，这样你可以放进你的学习 repo，每次遇到不会的语法就直接加进去，慢慢变成你的“私人工具书”？
