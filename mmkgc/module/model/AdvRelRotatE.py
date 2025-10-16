import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class AdvRelRotatE(Model):

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
            margin=6.0,
            epsilon=2.0,
            img_emb=None,  # 单实例图像嵌入：(ent_tot, img_dim)
            text_emb=None  # 单实例文本嵌入：(ent_tot, text_dim)
    ):

        super(AdvRelRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None and text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2  # 实体结构嵌入维度（RotatE分实部虚部）
        self.dim_r = dim      # 关系嵌入维度

        # --------------------------
        # 1. 基础嵌入层（结构+单实例模态）
        # --------------------------
        # 结构嵌入（RotatE标准初始化）
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(self.ent_embeddings.weight, -self.ent_embedding_range.item(), self.ent_embedding_range.item())
        nn.init.uniform_(self.rel_embeddings.weight, -self.rel_embedding_range.item(), self.rel_embedding_range.item())

        # 单实例模态嵌入（图像+文本）
        self.img_dim = img_emb.shape[1]    # 图像原始维度（如2048）
        self.text_dim = text_emb.shape[1]  # 文本原始维度（如768）
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        # 模态投影层：映射到结构嵌入维度
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )

        # --------------------------
        # 2. 单实例链接感知融合模块（核心修改）
        # --------------------------
        self.num_heads = 2      # 多头部注意力数量
        self.threshold = 0.1     # 模态权重过滤阈值（低于此值则该模态被弱化）
        
        # 关系投影层：将关系嵌入(dim_r)映射到实体维度(dim_e)，解决拼接维度不匹配
        self.rel_proj_to_e = nn.Linear(self.dim_r, self.dim_e)
        # 链接信息编码层：生成当前三元组的语义向量
        self.link_encoder = nn.Linear(3 * self.dim_e, self.dim_e)
        
        # 模态重要性计算层：计算文本/图像相对于当前链接的权重
        self.text_weight_layer = nn.Linear(self.dim_e * 2, 1)  # 输入：文本嵌入+链接向量
        self.img_weight_layer = nn.Linear(self.dim_e * 2, 1)   # 输入：图像嵌入+链接向量
        


        # --------------------------
        # 3. 原模型其他必要组件
        # --------------------------
        self.margin = nn.Parameter(torch.Tensor([margin]), requires_grad=False)
        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(self.rel_gate.weight, -self.ent_embedding_range.item(), self.ent_embedding_range.item())
        self.adv_scores = nn.Sequential(
            nn.Linear(self.dim_e, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, 1)
        )

    def _modal_fusion(self, entity_struct, entity_text, entity_image, link_vector):
        """
        单实例模态融合：计算文本/图像相对于结构模态的动态权重
        input:
            entity_struct: (batch_size, dim_e) → 结构嵌入（基础模态）
            entity_text: (batch_size, dim_e) → 单实例文本嵌入
            entity_image: (batch_size, dim_e) → 单实例图像嵌入
            link_vector: (batch_size, dim_e) → 链接信息向量（当前三元组语义）
        output:
            final_fusion: (batch_size, dim_e) → 融合后嵌入
        """
        batch_size = entity_struct.shape[0]

        # --------------------------
        # 子步骤1：计算模态重要性权重（核心修改）
        # --------------------------
        # 文本模态权重：基于文本嵌入与链接向量的相关性
        text_cat = torch.cat([entity_text, link_vector], dim=1)  # (batch_size, 2*dim_e)
        text_raw_weight = self.text_weight_layer(text_cat).squeeze(-1)  # (batch_size,)
        text_weight = torch.sigmoid(text_raw_weight)  # 映射到[0,1]，表示文本相对重要性
        
        # 图像模态权重：基于图像嵌入与链接向量的相关性
        img_cat = torch.cat([entity_image, link_vector], dim=1)  # (batch_size, 2*dim_e)
        img_raw_weight = self.img_weight_layer(img_cat).squeeze(-1)  # (batch_size,)
        img_weight = torch.sigmoid(img_raw_weight)  # 映射到[0,1]，表示图像相对重要性
        
        # 过滤低重要性模态（低于阈值则权重置为0）
        text_weight = torch.where(text_weight > self.threshold, text_weight, torch.zeros_like(text_weight))
        img_weight = torch.where(img_weight > self.threshold, img_weight, torch.zeros_like(img_weight))

        # --------------------------
        # 子步骤2：加权融合三模态（结构为基础，文本/图像按权重增强）
        # --------------------------
        # 结构模态权重固定为1（基础模态），文本和图像按动态权重调整
        fused = entity_struct + (text_weight.unsqueeze(1) * entity_text) + (img_weight.unsqueeze(1) * entity_image)

        
        # 返回融合结果和权重（可选，用于调试）
        return fused

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        
        # 1. 获取各模态嵌入
        h_struct = self.ent_embeddings(batch_h)  # 结构嵌入 (batch_size, dim_e)
        t_struct = self.ent_embeddings(batch_t)
        r_struct = self.rel_embeddings(batch_r)  # 关系嵌入 (batch_size, dim_r)
        
        h_text = self.text_proj(self.text_embeddings(batch_h))  # 文本嵌入 (batch_size, dim_e)
        t_text = self.text_proj(self.text_embeddings(batch_t))
        h_img = self.img_proj(self.img_embeddings(batch_h))     # 图像嵌入 (batch_size, dim_e)
        t_img = self.img_proj(self.img_embeddings(batch_t))
        
        # 2. 生成链接信息向量（修复维度不匹配）
        r_proj = self.rel_proj_to_e(r_struct)  # 关系嵌入映射到实体维度 (batch_size, dim_e)
        link_concat = torch.cat([h_struct, r_proj, t_struct], dim=1)  # (batch_size, 3*dim_e)
        link_vector = self.link_encoder(link_concat)
        link_vector = F.relu(link_vector)  # (batch_size, dim_e)
        
        # 3. 单实例模态融合（头实体+尾实体）
        h_fused = self._modal_fusion(
            entity_struct=h_struct,
            entity_text=h_text,
            entity_image=h_img,
            link_vector=link_vector
        )
        t_fused= self._modal_fusion(
            entity_struct=t_struct,
            entity_text=t_text,
            entity_image=t_img,
            link_vector=link_vector
        )
        
        # 4. 计算RotatE得分
        score = self.margin - self._calc(h_fused, t_fused, r_struct, mode)
        return score


    def predict(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        if mode == "head_batch":
            batch_size = batch_h.shape[0]
            batch_r = batch_r.repeat(batch_size)
            batch_t = batch_t.repeat(batch_size) 
        else:
            batch_size = batch_t.shape[0]
            batch_r = batch_r.repeat(batch_size)
            batch_h = batch_h.repeat(batch_size)            
        
        # 1. 获取各模态嵌入
        h_struct = self.ent_embeddings(batch_h)  # 结构嵌入 (batch_size, dim_e)
        t_struct = self.ent_embeddings(batch_t)
        r_struct = self.rel_embeddings(batch_r)  # 关系嵌入 (batch_size, dim_r)
        
        h_text = self.text_proj(self.text_embeddings(batch_h))  # 文本嵌入 (batch_size, dim_e)
        t_text = self.text_proj(self.text_embeddings(batch_t))
        h_img = self.img_proj(self.img_embeddings(batch_h))     # 图像嵌入 (batch_size, dim_e)
        t_img = self.img_proj(self.img_embeddings(batch_t))
        
        # 2. 生成链接信息向量（修复维度不匹配）
        r_proj = self.rel_proj_to_e(r_struct)  # 关系嵌入映射到实体维度 (batch_size, dim_e)
        link_concat = torch.cat([h_struct, r_proj, t_struct], dim=1)  # (batch_size, 3*dim_e)
        link_vector = self.link_encoder(link_concat)
        link_vector = F.relu(link_vector)  # (batch_size, dim_e)
        
        # 3. 单实例模态融合（头实体+尾实体）
        h_fused = self._modal_fusion(
            entity_struct=h_struct,
            entity_text=h_text,
            entity_image=h_img,
            link_vector=link_vector
        )
        t_fused = self._modal_fusion(
            entity_struct=t_struct,
            entity_text=t_text,
            entity_image=t_img,
            link_vector=link_vector
        )
        
        # 4. 计算RotatE得分
        score = self.margin - self._calc(h_fused, t_fused, r_struct, mode)
        score = -score
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        # 融合模块参数加入正则化（新增）
        regul = (torch.mean(h **2) + torch.mean(t** 2) + torch.mean(r **2) +
                 torch.mean(self.link_encoder.weight** 2) + 
                 torch.mean(self.text_weight_layer.weight **2) +
                 torch.mean(self.img_weight_layer.weight** 2)) / 6
        return regul
    