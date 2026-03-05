# 模块文档：Interpretation (热力图归因引擎)

## 1. 概述 (Overview)
Interpretation 模块是 DriveCritic-VL 的核心组件，实现了 EAGLE (Efficient Attribution for Grounded Language Explanations) 算法。该模块通过迭代搜索图像或视频中的关键区域，量化不同视觉输入对模型特定 Token 输出的贡献度，最终生成显著性热力图。

## 2. 组件说明 (Components)

### 2.1 InterpretationEngine (控制器)
- **职责**: 协调适配器、处理器与策略类，提供统一的 `explain()` 接口。
- **关键方法**: 
    - `explain(path, prompt, target_phrase)`: 自动判断输入类型（图/视）并触发归因流程。

### 2.2 Qwen3Adaptor (模型适配)
- **职责**: 封装模型推理接口，处理 Teacher Forcing 逻辑。
- **技术细节**: 
    - 通过覆盖 `input_ids` 确保归因上下文的稳定性。
    - 优化 GPU 上的通道转换，减少 CPU-GPU 拷贝。

### 2.3 VisualProcessor (视觉处理)
- **职责**: 视觉区域划分与掩码图像合成。
- **支持模式**: 
    - `2D Grid/SLIC`: 图像分割。
    - `3D Spatio-Temporal`: 视频时空体素划分。

### 2.4 EagleStrategy (核心算法)
- **职责**: 执行子模优化贪婪搜索。
- **目标函数**: $f(S) = \lambda_1 P(z_i | S) + \lambda_2 (1 - P(z_i | V \setminus S))$。
- **性能**: 采用批处理（Batching）与张量广播加速。

## 3. 核心逻辑流 (Internal Logic)
1. **初始推理**: 获取模型原始响应，并根据 `target_phrase` 定位目标 Token 及其在序列中的位置。
2. **区域划分**: 将视觉输入切分为 $N$ 个独立候选区域。
3. **贪婪搜索**:
    - 在每一步中，评估剩余区域 $c$ 加入当前集合 $S$ 后目标 Token 概率的增益。
    - 并行计算“插入（Insertion）”与“删除（Deletion）”分数。
    - 选择增益最大的区域。
4. **可视化**: 累计各步骤的选择增益，生成归因图并与原图叠加。

## 4. 使用示例 (Usage)

```python
from src.interpretation.engine import InterpretationEngine

# 初始化引擎
engine = InterpretationEngine(model_path="models/Qwen3-VL-2B-Instruct")

# 运行图像解释
result = engine.explain(
    path="scene.jpg", 
    prompt_text="前方的交通状况如何？",
    target_phrase="红灯"
)

# 结果包含响应文本、选择序列及热力图保存路径
print(f"解释结果已保存至: {result['save_path']}")
```

## 5. 注意事项
- **计算开销**: 视频归因步数建议控制在 32 步以内，以平衡生成时间与解释质量。
- **Token 对齐**: 若模型输出包含大量特殊字符，建议通过 `target_phrase` 显式指定归因目标。
