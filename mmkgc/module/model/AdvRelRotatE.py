import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import faiss  # 新增：FAISS用于高效相似性检索

class AdvRelRotatE(Model):

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
            margin=6.0,
            epsilon=2.0,
            img_emb=None,  # 单实例图像嵌入：(ent_tot, img_dim)
            text_emb=None,  # 单实例文本嵌入：(ent_tot, text_dim)
            lambda_interp=0.5,  # 新增：CMR插值超参数 λ (KS与ES概率混合权重)
            k_neighbors=10,     # 新增：检索的语义邻居数量 k
            temperature=0.1     # 新增：对比学习温度 τ
    ):

        super(AdvRelRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None and text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2  # 实体结构嵌入维度（RotatE分实部虚部）
        self.dim_r = dim      # 关系嵌入维度

        # --------------------------
        # 1. 基础嵌入层（结构+单实例模态） - 无变化
        # --------------------------
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
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb, freeze=True)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb, freeze=True)
        # 模态投影层：映射到结构嵌入维度 - 可选升级为CMR的VMN前缀（若用PLM）
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
        # 2. 单实例链接感知融合模块 - 无变化，但将在对比阶段使用
        # --------------------------
        self.threshold = 0.1   # 模态权重过滤阈值

        self.rel_proj_to_e = nn.Linear(self.dim_r, self.dim_e)
        self.link_encoder = nn.Linear(3 * self.dim_e, self.dim_e)
        self.text_weight_layer = nn.Linear(self.dim_e * 2, 1)
        self.img_weight_layer = nn.Linear(self.dim_e * 2, 1)

        # --------------------------
        # 3. 新增：CMR语义邻居检索组件
        # --------------------------
        self.lambda_interp = lambda_interp  # λ：KS与ES概率混合权重
        self.k_neighbors = k_neighbors      # 检索邻居数
        self.temperature = temperature      # 对比学习温度
        self.ks = None  # Knowledge Store: 查询嵌入 (N_train, dim_e) 和目标实体ID (N_train,)
        self.es = None  # Entity Store: 实体嵌入 (ent_tot, dim_e) 和实体ID (ent_tot,)
        self.faiss_index_ks = None  # FAISS索引 for KS (L2距离)
        self.faiss_index_es = None  # FAISS索引 for ES

        # --------------------------
        # 4. 原模型其他必要组件 - 无变化
        # --------------------------
        self.margin = nn.Parameter(torch.Tensor([margin]), requires_grad=False)
        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(self.rel_gate.weight, -self.ent_embedding_range.item(), self.ent_embedding_range.item())
        self.adv_scores = nn.Sequential(
            nn.Linear(self.dim_e, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, 1)
        )
        self.pi_const = 3.141592653589793  # 新增：RotatE的pi常量

    def _modal_fusion(self, entity_struct, entity_text, entity_image, link_vector):
        """
        单实例模态融合：计算文本/图像相对于结构模态的动态权重 - 无变化
        """
        batch_size = entity_struct.shape[0]

        text_cat = torch.cat([entity_text, link_vector], dim=1)
        text_raw_weight = self.text_weight_layer(text_cat).squeeze(-1)
        text_weight = torch.sigmoid(text_raw_weight)
        text_weight = torch.where(text_weight > self.threshold, text_weight, torch.zeros_like(text_weight))
        
        img_cat = torch.cat([entity_image, link_vector], dim=1)
        img_raw_weight = self.img_weight_layer(img_cat).squeeze(-1)
        img_weight = torch.sigmoid(img_raw_weight)
        img_weight = torch.where(img_weight > self.threshold, img_weight, torch.zeros_like(img_weight))

        fused = entity_struct + (text_weight.unsqueeze(1) * entity_text) + (img_weight.unsqueeze(1) * entity_image)
        return fused

    def _calc(self, h, t, r, mode):
        # RotatE计算 - 无变化
        pi = self.pi_const
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        phase_relation = r / (self.rel_embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

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

    # 新增：CMR对比学习阶段（预训练）
    def contrastive_train(self, train_data_loader, epochs=5, queue_size=8192):
        """
        对比阶段：使用InfoNCE损失优化嵌入，使相同目标的查询嵌入更接近
        输入：train_data_loader（提供{'batch_h', 'batch_t', 'batch_r'}的批次）
        这是一个独立预训练步骤，在主训练前调用
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        queue = torch.randn(queue_size, self.dim_e).to(next(self.parameters()).device)
        queue = F.normalize(queue, dim=1)

        for epoch in range(epochs):
            for data in train_data_loader:
                optimizer.zero_grad()
                batch_h, batch_t, batch_r = data['batch_h'], data['batch_t'], data['batch_r']
                h_struct = self.ent_embeddings(batch_h)
                t_struct = self.ent_embeddings(batch_t)
                r_struct = self.rel_embeddings(batch_r)
                h_text = self.text_proj(self.text_embeddings(batch_h))
                t_text = self.text_proj(self.text_embeddings(batch_t))
                h_img = self.img_proj(self.img_embeddings(batch_h))
                t_img = self.img_proj(self.img_embeddings(batch_t))
                r_proj = self.rel_proj_to_e(r_struct)
                link_concat = torch.cat([h_struct, r_proj, t_struct], dim=1)
                link_vector = F.relu(self.link_encoder(link_concat))

                h_fused = self._modal_fusion(h_struct, h_text, h_img, link_vector)
                t_fused = self._modal_fusion(t_struct, t_text, t_img, link_vector)

                query_emb = F.normalize(h_fused + r_proj, dim=1)  # 查询：h_fused + r_proj
                entity_emb = F.normalize(t_fused, dim=1)  # 正样本实体

                sim_pos = (query_emb * entity_emb).sum(1) / self.temperature
                sim_neg = (query_emb @ queue.T) / self.temperature
                loss = -sim_pos + torch.logsumexp(sim_neg, dim=1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()

                queue = torch.cat([queue[batch_h.shape[0]:], entity_emb.detach()], dim=0)

            print(f"对比学习第 {epoch} 轮: 损失 {loss.item()}")

    # 新增：CMR记忆阶段（在对比训练后调用一次）
    def memorize_stores(self, train_data_loader, all_entities=torch.arange(0, self.ent_tot)):
        """
        记忆阶段：存储KS（查询嵌入 -> 目标实体ID）和ES（实体嵌入 -> 实体ID）
        输入：train_data_loader（全训练数据），all_entities（所有实体ID，支持未见实体）
        """
        device = next(self.parameters()).device
        query_embs = []
        targets = []
        entity_embs = []

        # 构建KS：从训练三元组提取查询嵌入和目标
        with torch.no_grad():
            for data in train_data_loader:
                batch_h, batch_t, batch_r = data['batch_h'], data['batch_t'], data['batch_r']
                h_struct = self.ent_embeddings(batch_h)
                r_struct = self.rel_embeddings(batch_r)
                h_text = self.text_proj(self.text_embeddings(batch_h))
                h_img = self.img_proj(self.img_embeddings(batch_h))
                r_proj = self.rel_proj_to_e(r_struct)
                link_concat = torch.cat([h_struct, r_proj, self.ent_embeddings(batch_t)], dim=1)
                link_vector = F.relu(self.link_encoder(link_concat))
                h_fused = self._modal_fusion(h_struct, h_text, h_img, link_vector)
                query_emb = F.normalize(h_fused + r_proj, dim=1)
                query_embs.append(query_emb)
                targets.append(batch_t)

        self.ks_queries = torch.cat(query_embs, dim=0).cpu().numpy()
        self.ks_targets = torch.cat(targets, dim=0).cpu().numpy()

        self.faiss_index_ks = faiss.IndexFlatL2(self.dim_e)
        self.faiss_index_ks.add(self.ks_queries)

        # 构建ES：所有实体融合嵌入
        with torch.no_grad():
            batch_size = 512
            for i in range(0, len(all_entities), batch_size):
                batch_ent = all_entities[i:i+batch_size].to(device)
                e_struct = self.ent_embeddings(batch_ent)
                e_text = self.text_proj(self.text_embeddings(batch_ent))
                e_img = self.img_proj(self.img_embeddings(batch_ent))
                link_vector = torch.zeros(len(batch_ent), self.dim_e).to(device)  # 简化：零链接向量
                e_fused = self._modal_fusion(e_struct, e_text, e_img, link_vector)
                entity_embs.append(F.normalize(e_fused, dim=1))

        self.es_embs = torch.cat(entity_embs, dim=0).cpu().numpy()
        self.es_ids = all_entities.cpu().numpy()

        self.faiss_index_es = faiss.IndexFlatL2(self.dim_e)
        self.faiss_index_es.add(self.es_embs)

    def forward(self, data):
        # 主训练forward - 无变化
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        
        h_struct = self.ent_embeddings(batch_h)
        t_struct = self.ent_embeddings(batch_t)
        r_struct = self.rel_embeddings(batch_r)
        
        h_text = self.text_proj(self.text_embeddings(batch_h))
        t_text = self.text_proj(self.text_embeddings(batch_t))
        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = self.img_proj(self.img_embeddings(batch_t))
        
        r_proj = self.rel_proj_to_e(r_struct)
        link_concat = torch.cat([h_struct, r_proj, t_struct], dim=1)
        link_vector = F.relu(self.link_encoder(link_concat))
        
        h_fused = self._modal_fusion(h_struct, h_text, h_img, link_vector)
        t_fused = self._modal_fusion(t_struct, t_text, t_img, link_vector)
        
        score = self.margin - self._calc(h_fused, t_fused, r_struct, mode)
        return score

    def predict(self, data):
        # 预测：添加CMR检索增强
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        
        if mode == "head_batch":
            batch_size = batch_h.shape[0]
            batch_r = batch_r.repeat_interleave(batch_size)
            batch_t = batch_t.repeat_interleave(batch_size)
            pred_entities = batch_h
        else:
            batch_size = batch_t.shape[0]
            batch_r = batch_r.repeat_interleave(batch_size)
            batch_h = batch_h.repeat_interleave(batch_size)
            pred_entities = batch_t
        
        h_struct = self.ent_embeddings(batch_h)
        t_struct = self.ent_embeddings(batch_t)
        r_struct = self.rel_embeddings(batch_r)
        
        h_text = self.text_proj(self.text_embeddings(batch_h))
        t_text = self.text_proj(self.text_embeddings(batch_t))
        h_img = self.img_proj(self.img_embeddings(batch_h))
        t_img = self.img_proj(self.img_embeddings(batch_t))
        
        r_proj = self.rel_proj_to_e(r_struct)
        link_concat = torch.cat([h_struct, r_proj, t_struct], dim=1)
        link_vector = F.relu(self.link_encoder(link_concat))
        
        h_fused = self._modal_fusion(h_struct, h_text, h_img, link_vector)
        t_fused = self._modal_fusion(t_struct, t_text, t_img, link_vector)
        
        rot_scores = -self._calc(h_fused, t_fused, r_struct, mode)
        
        if self.ks is None:
            raise ValueError("请先调用 memorize_stores()！")
        
        with torch.no_grad():
            if mode == "head_batch":
                query_emb = F.normalize(t_fused + r_proj, dim=1).cpu().numpy()  # 尾预测：(t,r,?)
            else:
                query_emb = F.normalize(h_fused + r_proj, dim=1).cpu().numpy()  # 头预测：(h,r,?)
            
            # p_ES：直接实体相似度
            distances_es, indices_es = self.faiss_index_es.search(query_emb, self.ent_tot)
            p_es = F.softmax(-torch.tensor(distances_es) / self.temperature, dim=1)
            
            # p_KS：KS检索k邻居，聚合目标概率
            distances_ks, indices_ks = self.faiss_index_ks.search(query_emb, self.k_neighbors)
            ks_targets = self.ks_targets[indices_ks]
            unique_targets, unique_indices = torch.unique(torch.tensor(ks_targets), dim=1, return_inverse=True)
            min_dist_per_target = torch.full((query_emb.shape[0], self.ent_tot), float('inf'))
            for b in range(query_emb.shape[0]):
                for tgt in unique_targets[b]:
                    mask = (ks_targets[b] == tgt)
                    min_dist = distances_ks[b][mask].min()
                    min_dist_per_target[b, tgt] = min_dist
            p_ks = F.softmax(-min_dist_per_target / self.temperature, dim=1)
            
            # 插值：p = λ p_ks + (1-λ) p_es
            p_final = self.lambda_interp * p_ks + (1 - self.lambda_interp) * p_es
            
            # 结合RotatE：加权分数
            enhanced_scores = rot_scores.view(-1, self.ent_tot) * p_final
        
        return enhanced_scores.cpu().numpy().flatten()

    def regularization(self, data):
        # 正则化 - 添加CMR组件
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h **2) + torch.mean(t** 2) + torch.mean(r **2) +
                 torch.mean(self.link_encoder.weight** 2) + 
                 torch.mean(self.text_weight_layer.weight **2) +
                 torch.mean(self.img_weight_layer.weight** 2)) / 6
        return regul