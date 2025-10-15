# 模态协作知识提取模块使用指南

## 概述

模态协作知识提取模块（Modal Collaborative Knowledge Extractor）是一个先进的多模态融合组件，专为多模态知识图谱补全任务设计。

## 核心功能

### 1. 跨模态注意力机制
- **目的**：过滤模态间的噪声，提取互补信息
- **实现**：多头交叉注意力，每个模态作为query从其他模态中提取有用特征
- **优势**：自适应地关注不同模态的相关信息，降低噪声影响

### 2. Transformer编码器
- **目的**：深度提取模态内的语义特征
- **实现**：可配置的多层Transformer编码层
- **优势**：捕捉长距离依赖和复杂语义关系

### 3. 多任务辅助损失
- **MSE相似性损失**：促进不同模态在语义空间的对齐
- **对比学习损失**：增强模态表示的区分性
- **联合优化**：通过辅助损失与主任务损失共同训练

### 4. 精细化嵌入输出
- 输出三种精细化的模态嵌入（结构、视觉、文本）
- 可用于下游任务：实体对齐、跨模态检索、知识融合等

## 使用方法

### 基础使用

#### 1. 初始化模型（启用知识提取器）

```python
import torch
from mmkgc.module.model.AdvRelRotatE import AdvRelRotatE

# 加载预训练的多模态嵌入
img_emb = torch.load('embeddings/DB15K_ImageEmbeddings.pth')
text_emb = torch.load('embeddings/DB15K_TextEmbeddings.pth')

# 初始化模型，启用模态协作知识提取
model = AdvRelRotatE(
    ent_tot=14951,
    rel_tot=279,
    dim=250,
    margin=12.0,
    epsilon=2.0,
    img_emb=img_emb,
    text_emb=text_emb,
    # 知识提取器配置
    use_knowledge_extractor=True,  # 启用知识提取模块
    num_heads=4,                    # 注意力头数
    num_layers=2,                   # Transformer层数
    dropout=0.1,                    # Dropout率
    lambda_auxiliary=0.1            # 辅助损失权重
)
```

#### 2. 训练时计算总损失

```python
# 前向传播
data = {
    'batch_h': batch_h,
    'batch_t': batch_t,
    'batch_r': batch_r,
    'mode': 'normal'
}

# 计算主任务损失（链接预测）
score = model(data)
main_loss = loss_function(score, labels)

# 计算正则化损失
regul_loss = model.regularization(data)

# 计算辅助损失（模态对齐）
auxiliary_loss, loss_info = model.get_auxiliary_loss()

# 总损失
total_loss = main_loss + mu * regul_loss + auxiliary_loss

# 打印损失信息（可选）
print(f"Main Loss: {main_loss.item():.4f}")
print(f"Regularization Loss: {regul_loss.item():.4f}")
print(f"Auxiliary Loss: {auxiliary_loss.item():.4f}")
print(f"Loss Details: {loss_info}")

# 反向传播
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

#### 3. 获取精细化嵌入（用于下游任务）

```python
# 获取特定实体的精细化嵌入
entity_ids = torch.tensor([0, 1, 2, 3, 4])  # 实体ID

refined_embeddings = model.get_refined_embeddings(entity_ids)

# 访问不同模态的精细化嵌入
struct_refined = refined_embeddings['struct']   # [batch_size, dim]
visual_refined = refined_embeddings['visual']   # [batch_size, dim]
text_refined = refined_embeddings['text']       # [batch_size, dim]

# 应用场景：实体对齐
similarity = torch.cosine_similarity(struct_refined, visual_refined)

# 应用场景：跨模态检索
query_emb = struct_refined[0]  # 查询嵌入
doc_embs = torch.stack([visual_refined, text_refined])  # 文档嵌入
retrieval_scores = torch.matmul(query_emb, doc_embs.T)
```

### 高级配置

#### 1. 禁用知识提取器（使用原始模型）

```python
model = AdvRelRotatE(
    ent_tot=14951,
    rel_tot=279,
    dim=250,
    margin=12.0,
    epsilon=2.0,
    img_emb=img_emb,
    text_emb=text_emb,
    use_knowledge_extractor=False  # 禁用，使用原始模型
)
```

#### 2. 调整模块参数

```python
model = AdvRelRotatE(
    # ... 基础参数 ...
    use_knowledge_extractor=True,
    num_heads=8,              # 增加注意力头数（更细粒度的特征提取）
    num_layers=3,             # 增加Transformer层数（更深的语义理解）
    dropout=0.2,              # 调整dropout（防止过拟合）
    lambda_auxiliary=0.2      # 增加辅助损失权重（更强的模态对齐）
)
```

#### 3. 自定义辅助损失权重

```python
# 在知识提取器中直接计算
auxiliary_loss, loss_info = model.knowledge_extractor.compute_auxiliary_loss(
    auxiliary_outputs=model.last_h_auxiliary,
    lambda_mse=0.3  # 自定义MSE损失权重（0-1之间）
)
```

## 完整训练示例

```python
import torch
import torch.optim as optim
from mmkgc.module.model.AdvRelRotatE import AdvRelRotatE
from mmkgc.module.loss.SigmoidLoss import SigmoidLoss

# 1. 加载数据
img_emb = torch.load('embeddings/DB15K_ImageEmbeddings.pth')
text_emb = torch.load('embeddings/DB15K_TextEmbeddings.pth')

# 2. 初始化模型
model = AdvRelRotatE(
    ent_tot=14951,
    rel_tot=279,
    dim=250,
    margin=12.0,
    img_emb=img_emb,
    text_emb=text_emb,
    use_knowledge_extractor=True,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    lambda_auxiliary=0.1
).cuda()

# 3. 定义损失和优化器
criterion = SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        # 准备数据
        data = {
            'batch_h': batch['h'].cuda(),
            'batch_t': batch['t'].cuda(),
            'batch_r': batch['r'].cuda(),
            'mode': 'normal'
        }

        # 前向传播
        score = model(data)

        # 计算损失
        main_loss = criterion(score, batch['labels'].cuda())
        regul_loss = model.regularization(data)
        auxiliary_loss, loss_info = model.get_auxiliary_loss()

        total_loss = main_loss + 0.0001 * regul_loss + auxiliary_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录（每100个batch）
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"  Main: {main_loss.item():.4f}")
            print(f"  Aux: {auxiliary_loss.item():.4f}")
            print(f"  MSE(S-V): {loss_info.get('head_mse_sv', 0):.4f}")
            print(f"  MSE(S-T): {loss_info.get('head_mse_st', 0):.4f}")
            print(f"  MSE(V-T): {loss_info.get('head_mse_vt', 0):.4f}")
```

## 模块架构详解

### 组件层次结构

```
AdvRelRotatE
├── 原始嵌入层
│   ├── ent_embeddings (结构化嵌入)
│   ├── img_embeddings (视觉嵌入)
│   └── text_embeddings (文本嵌入)
│
├── 投影层
│   ├── img_proj (视觉投影)
│   └── text_proj (文本投影)
│
└── ModalCollaborativeKnowledgeExtractor (新增)
    ├── 模态特定编码器
    │   ├── struct_encoder (结构Transformer)
    │   ├── visual_encoder (视觉Transformer)
    │   └── text_encoder (文本Transformer)
    │
    ├── 跨模态注意力
    │   ├── cross_attn_sv (结构→视觉)
    │   ├── cross_attn_st (结构→文本)
    │   ├── cross_attn_vt (视觉→文本)
    │   ├── cross_attn_vs (视觉→结构)
    │   ├── cross_attn_ts (文本→结构)
    │   └── cross_attn_tv (文本→视觉)
    │
    ├── 融合层
    │   ├── fusion_struct
    │   ├── fusion_visual
    │   └── fusion_text
    │
    └── 精细化层
        └── refinement
```

### 数据流

```
输入：(结构嵌入, 视觉嵌入, 文本嵌入)
  ↓
模态特定编码（Transformer）
  ↓
跨模态注意力（噪声过滤）
  ├─ 结构 ← 视觉、文本
  ├─ 视觉 ← 结构、文本
  └─ 文本 ← 结构、视觉
  ↓
特征融合（拼接+投影）
  ↓
精细化处理
  ↓
输出：精细化嵌入 + 辅助信息
```

## 参数调优建议

### num_heads（注意力头数）
- **推荐值**：4-8
- **小数据集**：4
- **大数据集**：8
- **影响**：更多头能捕捉更多样的特征关系，但增加计算成本

### num_layers（Transformer层数）
- **推荐值**：2-3
- **简单任务**：2
- **复杂任务**：3
- **影响**：更深的网络能学习更复杂的模式，但可能过拟合

### dropout
- **推荐值**：0.1-0.2
- **大数据集**：0.1
- **小数据集**：0.2-0.3
- **影响**：防止过拟合，提高泛化能力

### lambda_auxiliary
- **推荐值**：0.1-0.3
- **弱对齐需求**：0.1
- **强对齐需求**：0.3
- **影响**：控制模态对齐的强度

## 性能优化

### 1. 内存优化
```python
# 如果显存不足，可以：
# - 减少batch_size
# - 减少num_layers
# - 减少num_heads
# - 使用梯度累积
```

### 2. 速度优化
```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    score = model(data)
    loss = criterion(score)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 渐进式训练
```python
# 先训练基础模型，再启用知识提取器
# 阶段1：预训练（不使用知识提取器）
model.use_knowledge_extractor = False
for epoch in range(50):
    train_epoch()

# 阶段2：微调（启用知识提取器）
model.use_knowledge_extractor = True
for epoch in range(50):
    train_epoch()
```

## 常见问题

### Q1: 辅助损失为0怎么办？
```python
# 检查forward是否被调用
# 确保在调用get_auxiliary_loss()之前先调用forward()
score = model(data)  # 必须先执行
aux_loss, info = model.get_auxiliary_loss()  # 再获取辅助损失
```

### Q2: 显存溢出怎么办？
```python
# 方案1：减小参数
model = AdvRelRotatE(..., num_heads=2, num_layers=1)

# 方案2：使用梯度检查点
torch.utils.checkpoint.checkpoint(model.knowledge_extractor, ...)
```

### Q3: 如何可视化注意力权重？
```python
# 获取注意力权重
_, auxiliary = model.knowledge_extractor(struct_emb, visual_emb, text_emb)
attn_weights = auxiliary['attention_weights']

# 可视化结构→视觉的注意力
import matplotlib.pyplot as plt
plt.imshow(attn_weights['sv'][0, 0].detach().cpu().numpy())
plt.colorbar()
plt.show()
```

## 引用

如果您使用了这个模块，请引用：

```bibtex
@inproceedings{MMKGC-ModalCollaborative,
  title={Modal Collaborative Knowledge Extraction for Multi-modal Knowledge Graph Completion},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## 版本历史

- **v1.0** (2024-10): 初始版本
  - 跨模态注意力机制
  - Transformer编码器
  - 多任务辅助损失
  - 精细化嵌入输出
