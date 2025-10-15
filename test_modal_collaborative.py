"""
测试模态协作知识提取模块
验证模块的功能和兼容性
"""

import torch
import sys
sys.path.append('.')

from mmkgc.module.model.ModalCollaborativeKnowledgeExtractor import (
    ModalCollaborativeKnowledgeExtractor,
    CrossModalAttention,
    TransformerEncoderLayer
)


def test_cross_modal_attention():
    """测试跨模态注意力机制"""
    print("=" * 60)
    print("测试 1: 跨模态注意力机制")
    print("=" * 60)

    batch_size = 4
    seq_len = 5
    dim = 128
    num_heads = 4

    # 初始化模块
    cross_attn = CrossModalAttention(dim=dim, num_heads=num_heads)

    # 创建测试数据
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)

    # 前向传播
    output, attn_weights = cross_attn(query, key, value)

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, dim), f"输出形状错误: {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"注意力权重形状错误: {attn_weights.shape}"

    print(f"✓ 输出形状正确: {output.shape}")
    print(f"✓ 注意力权重形状正确: {attn_weights.shape}")
    print(f"✓ 注意力权重和: {attn_weights.sum(dim=-1).mean().item():.4f} (应接近1.0)")
    print()


def test_transformer_encoder_layer():
    """测试Transformer编码层"""
    print("=" * 60)
    print("测试 2: Transformer编码层")
    print("=" * 60)

    batch_size = 4
    seq_len = 5
    dim = 128
    num_heads = 4

    # 初始化模块
    encoder_layer = TransformerEncoderLayer(dim=dim, num_heads=num_heads)

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, dim)

    # 前向传播
    output = encoder_layer(x)

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, dim), f"输出形状错误: {output.shape}"

    print(f"✓ 输出形状正确: {output.shape}")
    print()


def test_modal_collaborative_knowledge_extractor():
    """测试模态协作知识提取模块"""
    print("=" * 60)
    print("测试 3: 模态协作知识提取模块")
    print("=" * 60)

    batch_size = 8
    dim = 200
    num_heads = 4
    num_layers = 2

    # 初始化模块
    extractor = ModalCollaborativeKnowledgeExtractor(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        use_modal_specific=True
    )

    # 创建测试数据（模拟三种模态的嵌入）
    struct_emb = torch.randn(batch_size, dim)
    visual_emb = torch.randn(batch_size, dim)
    text_emb = torch.randn(batch_size, dim)

    # 前向传播
    refined_embeddings, auxiliary_outputs = extractor(struct_emb, visual_emb, text_emb)

    # 验证精细化嵌入
    print("精细化嵌入:")
    for modal_name, emb in refined_embeddings.items():
        assert emb.shape == (batch_size, dim), f"{modal_name} 形状错误: {emb.shape}"
        print(f"  ✓ {modal_name}: {emb.shape}")

    # 验证辅助输出
    print("\n辅助输出:")
    for key in ['struct_proj', 'visual_proj', 'text_proj']:
        assert key in auxiliary_outputs, f"缺少 {key}"
        assert auxiliary_outputs[key].shape == (batch_size, dim)
        print(f"  ✓ {key}: {auxiliary_outputs[key].shape}")

    # 验证注意力权重
    print("\n注意力权重:")
    attn_weights = auxiliary_outputs['attention_weights']
    for key in ['sv', 'st', 'vt', 'vs', 'ts', 'tv']:
        assert key in attn_weights, f"缺少注意力权重 {key}"
        print(f"  ✓ {key}: {attn_weights[key].shape}")

    # 计算辅助损失
    print("\n辅助损失:")
    aux_loss, loss_dict = extractor.compute_auxiliary_loss(auxiliary_outputs, lambda_mse=0.1)
    print(f"  总辅助损失: {aux_loss.item():.4f}")
    print(f"  MSE损失: {loss_dict['mse_loss']:.4f}")
    print(f"    - 结构-视觉: {loss_dict['mse_sv']:.4f}")
    print(f"    - 结构-文本: {loss_dict['mse_st']:.4f}")
    print(f"    - 视觉-文本: {loss_dict['mse_vt']:.4f}")
    print(f"  对比损失: {loss_dict['contrastive_loss']:.4f}")
    print()


def test_integration_with_model():
    """测试与AdvRelRotatE模型的集成"""
    print("=" * 60)
    print("测试 4: 与AdvRelRotatE模型集成")
    print("=" * 60)

    # 这里只做基础测试，不加载实际数据
    from mmkgc.module.model.AdvRelRotatE import AdvRelRotatE

    # 创建模拟的多模态嵌入
    ent_tot = 100
    img_dim = 4096
    text_dim = 768
    dim = 200

    img_emb = torch.randn(ent_tot, img_dim)
    text_emb = torch.randn(ent_tot, text_dim)

    # 初始化模型（启用知识提取器）
    print("初始化模型（启用知识提取器）...")
    model = AdvRelRotatE(
        ent_tot=ent_tot,
        rel_tot=10,
        dim=dim // 2,  # dim_e = dim * 2
        margin=6.0,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb,
        use_knowledge_extractor=True,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        lambda_auxiliary=0.1
    )
    print("✓ 模型初始化成功")

    # 创建测试数据
    batch_size = 16
    data = {
        'batch_h': torch.randint(0, ent_tot, (batch_size,)),
        'batch_t': torch.randint(0, ent_tot, (batch_size,)),
        'batch_r': torch.randint(0, 10, (batch_size,)),
        'mode': 'normal'
    }

    # 前向传播
    print("\n执行前向传播...")
    score = model(data)
    print(f"✓ 前向传播成功，得分形状: {score.shape}")

    # 计算正则化损失
    regul_loss = model.regularization(data)
    print(f"✓ 正则化损失: {regul_loss.item():.4f}")

    # 计算辅助损失
    aux_loss, loss_info = model.get_auxiliary_loss()
    print(f"✓ 辅助损失: {aux_loss.item():.4f}")
    print(f"  详细信息: {list(loss_info.keys())[:5]}...")  # 只打印前5个键

    # 获取精细化嵌入
    print("\n获取精细化嵌入...")
    entity_ids = torch.randint(0, ent_tot, (5,))
    refined = model.get_refined_embeddings(entity_ids)
    print(f"✓ 结构嵌入: {refined['struct'].shape}")
    print(f"✓ 视觉嵌入: {refined['visual'].shape}")
    print(f"✓ 文本嵌入: {refined['text'].shape}")

    # 测试禁用知识提取器的情况
    print("\n" + "=" * 60)
    print("测试禁用知识提取器...")
    model_no_extractor = AdvRelRotatE(
        ent_tot=ent_tot,
        rel_tot=10,
        dim=dim // 2,
        margin=6.0,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb,
        use_knowledge_extractor=False
    )
    print("✓ 模型初始化成功（无知识提取器）")

    score_no_extractor = model_no_extractor(data)
    print(f"✓ 前向传播成功，得分形状: {score_no_extractor.shape}")

    aux_loss_no_extractor, _ = model_no_extractor.get_auxiliary_loss()
    print(f"✓ 辅助损失: {aux_loss_no_extractor.item():.4f} (应为0.0)")

    print()


def test_backward_pass():
    """测试反向传播和梯度"""
    print("=" * 60)
    print("测试 5: 反向传播和梯度")
    print("=" * 60)

    batch_size = 8
    dim = 200
    num_heads = 4
    num_layers = 2

    # 初始化模块
    extractor = ModalCollaborativeKnowledgeExtractor(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )

    # 创建测试数据
    struct_emb = torch.randn(batch_size, dim, requires_grad=True)
    visual_emb = torch.randn(batch_size, dim, requires_grad=True)
    text_emb = torch.randn(batch_size, dim, requires_grad=True)

    # 前向传播
    refined_embeddings, auxiliary_outputs = extractor(struct_emb, visual_emb, text_emb)

    # 计算损失
    loss = refined_embeddings['struct'].sum()
    aux_loss, _ = extractor.compute_auxiliary_loss(auxiliary_outputs)
    total_loss = loss + aux_loss

    # 反向传播
    total_loss.backward()

    # 检查梯度
    assert struct_emb.grad is not None, "结构嵌入没有梯度"
    assert visual_emb.grad is not None, "视觉嵌入没有梯度"
    assert text_emb.grad is not None, "文本嵌入没有梯度"

    print(f"✓ 反向传播成功")
    print(f"✓ 结构嵌入梯度范数: {struct_emb.grad.norm().item():.4f}")
    print(f"✓ 视觉嵌入梯度范数: {visual_emb.grad.norm().item():.4f}")
    print(f"✓ 文本嵌入梯度范数: {text_emb.grad.norm().item():.4f}")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("模态协作知识提取模块 - 测试套件")
    print("=" * 60)
    print()

    try:
        # 运行各项测试
        test_cross_modal_attention()
        test_transformer_encoder_layer()
        test_modal_collaborative_knowledge_extractor()
        test_backward_pass()
        test_integration_with_model()

        # 总结
        print("=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        print("\n模块已成功集成，可以开始训练！")
        print("\n使用建议:")
        print("1. 查看 MODAL_COLLABORATIVE_USAGE.md 了解详细使用方法")
        print("2. 调整 num_heads, num_layers, lambda_auxiliary 等参数")
        print("3. 在训练脚本中添加辅助损失计算")
        print("4. 监控损失信息以了解模态对齐情况")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
