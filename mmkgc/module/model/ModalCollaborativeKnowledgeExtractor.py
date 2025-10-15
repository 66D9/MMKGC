import torch
import torch.nn as nn
import torch.nn.functional as F
from ..BaseModule import BaseModule


class CrossModalAttention(BaseModule):
    """
    跨模态注意力机制，用于过滤噪声并提取精细的模态间交互特征
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, dim] - 目标模态
            key: [batch_size, seq_len_k, dim] - 源模态
            value: [batch_size, seq_len_v, dim] - 源模态 (seq_len_k == seq_len_v)
            mask: optional attention mask
        Returns:
            output: [batch_size, seq_len_q, dim]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # Linear projections and reshape to multi-head
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_proj(context)

        return output, attention_weights


class TransformerEncoderLayer(BaseModule):
    """
    Transformer编码层，包含多头自注意力和前馈网络
    """
    def __init__(self, dim, num_heads=4, ff_dim=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        if ff_dim is None:
            ff_dim = dim * 4

        self.self_attn = CrossModalAttention(dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: optional attention mask
        Returns:
            output: [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class ModalCollaborativeKnowledgeExtractor(BaseModule):
    """
    模态协作知识提取模块
    功能：
    1. 使用加权嵌入作为输入
    2. 通过跨注意力机制过滤模态间噪声
    3. 使用Transformer提取精细特征
    4. 输出精细嵌入供对齐和检索使用
    """
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1, use_modal_specific=True):
        super(ModalCollaborativeKnowledgeExtractor, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_modal_specific = use_modal_specific

        # 模态特定的Transformer编码器
        if use_modal_specific:
            self.struct_encoder = nn.ModuleList([
                TransformerEncoderLayer(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.visual_encoder = nn.ModuleList([
                TransformerEncoderLayer(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.text_encoder = nn.ModuleList([
                TransformerEncoderLayer(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])

        # 跨模态注意力层 - 用于模态间协作和噪声过滤
        self.cross_attn_sv = CrossModalAttention(dim, num_heads, dropout)  # struct -> visual
        self.cross_attn_st = CrossModalAttention(dim, num_heads, dropout)  # struct -> text
        self.cross_attn_vt = CrossModalAttention(dim, num_heads, dropout)  # visual -> text
        self.cross_attn_vs = CrossModalAttention(dim, num_heads, dropout)  # visual -> struct
        self.cross_attn_ts = CrossModalAttention(dim, num_heads, dropout)  # text -> struct
        self.cross_attn_tv = CrossModalAttention(dim, num_heads, dropout)  # text -> visual

        # 融合层 - 整合跨模态信息
        self.fusion_struct = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_visual = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_text = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 最终的精细化层
        self.refinement = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        # 用于计算辅助损失的投影层
        self.similarity_proj = nn.Linear(dim, dim)

    def forward(self, struct_emb, visual_emb, text_emb):
        """
        Args:
            struct_emb: [batch_size, dim] - 结构化嵌入
            visual_emb: [batch_size, dim] - 视觉嵌入
            text_emb: [batch_size, dim] - 文本嵌入
        Returns:
            refined_embeddings: dict with keys 'struct', 'visual', 'text'
            auxiliary_outputs: dict with intermediate outputs for computing auxiliary loss
        """
        batch_size = struct_emb.size(0)

        # 扩展维度以适应Transformer: [batch_size, 1, dim]
        struct_emb = struct_emb.unsqueeze(1)
        visual_emb = visual_emb.unsqueeze(1)
        text_emb = text_emb.unsqueeze(1)

        # 1. 模态特定编码 (可选)
        if self.use_modal_specific:
            for layer in self.struct_encoder:
                struct_emb = layer(struct_emb)
            for layer in self.visual_encoder:
                visual_emb = layer(visual_emb)
            for layer in self.text_encoder:
                text_emb = layer(text_emb)

        # 2. 跨模态注意力 - 噪声过滤和特征增强
        # 每个模态作为query，从其他模态中提取互补信息
        struct_from_visual, attn_sv = self.cross_attn_sv(struct_emb, visual_emb, visual_emb)
        struct_from_text, attn_st = self.cross_attn_st(struct_emb, text_emb, text_emb)

        visual_from_struct, attn_vs = self.cross_attn_vs(visual_emb, struct_emb, struct_emb)
        visual_from_text, attn_vt = self.cross_attn_vt(visual_emb, text_emb, text_emb)

        text_from_struct, attn_ts = self.cross_attn_ts(text_emb, struct_emb, struct_emb)
        text_from_visual, attn_tv = self.cross_attn_tv(text_emb, visual_emb, visual_emb)

        # 3. 融合跨模态信息
        struct_fused = self.fusion_struct(
            torch.cat([struct_emb, struct_from_visual, struct_from_text], dim=-1)
        )
        visual_fused = self.fusion_visual(
            torch.cat([visual_emb, visual_from_struct, visual_from_text], dim=-1)
        )
        text_fused = self.fusion_text(
            torch.cat([text_emb, text_from_struct, text_from_visual], dim=-1)
        )

        # 4. 精细化处理
        struct_refined = self.refinement(struct_fused).squeeze(1)  # [batch_size, dim]
        visual_refined = self.refinement(visual_fused).squeeze(1)
        text_refined = self.refinement(text_fused).squeeze(1)

        # 5. 准备输出
        refined_embeddings = {
            'struct': struct_refined,
            'visual': visual_refined,
            'text': text_refined
        }

        # 用于辅助损失的特征投影
        auxiliary_outputs = {
            'struct_proj': self.similarity_proj(struct_refined),
            'visual_proj': self.similarity_proj(visual_refined),
            'text_proj': self.similarity_proj(text_refined),
            'attention_weights': {
                'sv': attn_sv, 'st': attn_st,
                'vs': attn_vs, 'vt': attn_vt,
                'ts': attn_ts, 'tv': attn_tv
            }
        }

        return refined_embeddings, auxiliary_outputs

    def compute_auxiliary_loss(self, auxiliary_outputs, lambda_mse=0.1):
        """
        计算多任务辅助损失 (MSE相似性损失)
        目标：促进不同模态之间的语义对齐

        Args:
            auxiliary_outputs: forward方法返回的auxiliary_outputs
            lambda_mse: MSE损失的权重系数
        Returns:
            total_loss: 总辅助损失
            loss_dict: 各项损失的详细信息
        """
        struct_proj = auxiliary_outputs['struct_proj']
        visual_proj = auxiliary_outputs['visual_proj']
        text_proj = auxiliary_outputs['text_proj']

        # 计算模态间的MSE相似性损失
        # 目标：使不同模态的投影在语义空间中相互接近
        mse_sv = F.mse_loss(struct_proj, visual_proj)
        mse_st = F.mse_loss(struct_proj, text_proj)
        mse_vt = F.mse_loss(visual_proj, text_proj)

        # 总MSE损失
        mse_loss = (mse_sv + mse_st + mse_vt) / 3.0

        # 可选：添加对比学习损失来增强区分性
        # 这里使用余弦相似度来衡量模态内的一致性
        struct_norm = F.normalize(struct_proj, p=2, dim=-1)
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        text_norm = F.normalize(text_proj, p=2, dim=-1)

        # 计算模态间的余弦相似度
        cos_sv = torch.mean(torch.sum(struct_norm * visual_norm, dim=-1))
        cos_st = torch.mean(torch.sum(struct_norm * text_norm, dim=-1))
        cos_vt = torch.mean(torch.sum(visual_norm * text_norm, dim=-1))

        # 对比损失：最大化模态间的相似度
        contrastive_loss = -(cos_sv + cos_st + cos_vt) / 3.0

        # 总辅助损失
        total_loss = lambda_mse * mse_loss + (1 - lambda_mse) * contrastive_loss

        loss_dict = {
            'mse_loss': mse_loss.item(),
            'mse_sv': mse_sv.item(),
            'mse_st': mse_st.item(),
            'mse_vt': mse_vt.item(),
            'contrastive_loss': contrastive_loss.item(),
            'cos_sv': cos_sv.item(),
            'cos_st': cos_st.item(),
            'cos_vt': cos_vt.item(),
            'total_auxiliary_loss': total_loss.item()
        }

        return total_loss, loss_dict
