import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model
from .ModalCollaborativeKnowledgeExtractor import ModalCollaborativeKnowledgeExtractor


class AdvRelRotatE(Model):

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
            margin=6.0,
            epsilon=2.0,
            img_emb=None,
            text_emb=None,
            use_knowledge_extractor=True,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            lambda_auxiliary=0.1
    ):

        super(AdvRelRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        # self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        # self.text_proj = nn.Linear(self.text_dim, self.dim_e)
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

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )

        self.adv_scores = nn.Sequential(
            nn.Linear(self.dim_e, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, 1)
        )

        # 模态协作知识提取模块
        self.use_knowledge_extractor = use_knowledge_extractor
        self.lambda_auxiliary = lambda_auxiliary
        if self.use_knowledge_extractor:
            self.knowledge_extractor = ModalCollaborativeKnowledgeExtractor(
                dim=self.dim_e,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                use_modal_specific=True
            )

    def get_joint_embeddings(self, es, ev, et, rg):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1)  # Design of V8
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors

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
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        rg = self.rel_gate(batch_r)

        # 如果启用了模态协作知识提取模块
        if self.use_knowledge_extractor:
            # 对头实体应用知识提取
            h_refined, h_auxiliary = self.knowledge_extractor(h, h_img_emb, h_text_emb)
            h_struct_refined = h_refined['struct']
            h_img_refined = h_refined['visual']
            h_text_refined = h_refined['text']

            # 对尾实体应用知识提取
            t_refined, t_auxiliary = self.knowledge_extractor(t, t_img_emb, t_text_emb)
            t_struct_refined = t_refined['struct']
            t_img_refined = t_refined['visual']
            t_text_refined = t_refined['text']

            # 使用精细化后的嵌入进行联合嵌入
            h_joint = self.get_joint_embeddings(h_struct_refined, h_img_refined, h_text_refined, rg)
            t_joint = self.get_joint_embeddings(t_struct_refined, t_img_refined, t_text_refined, rg)

            # 存储辅助输出用于损失计算
            self.last_h_auxiliary = h_auxiliary
            self.last_t_auxiliary = t_auxiliary
        else:
            # 原始方法：直接使用投影后的嵌入
            h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, rg)
            t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, rg)

        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score


    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def get_auxiliary_loss(self):
        """
        获取模态协作知识提取模块的辅助损失
        该损失用于联合优化模型，促进模态间的对齐

        Returns:
            auxiliary_loss: 辅助损失值
            loss_info: 损失详细信息字典
        """
        if not self.use_knowledge_extractor:
            return torch.tensor(0.0), {}

        if not hasattr(self, 'last_h_auxiliary') or not hasattr(self, 'last_t_auxiliary'):
            return torch.tensor(0.0), {}

        # 计算头实体和尾实体的辅助损失
        h_aux_loss, h_loss_info = self.knowledge_extractor.compute_auxiliary_loss(
            self.last_h_auxiliary, lambda_mse=self.lambda_auxiliary
        )
        t_aux_loss, t_loss_info = self.knowledge_extractor.compute_auxiliary_loss(
            self.last_t_auxiliary, lambda_mse=self.lambda_auxiliary
        )

        # 平均头尾实体的辅助损失
        total_aux_loss = (h_aux_loss + t_aux_loss) / 2.0

        # 合并损失信息
        loss_info = {
            'head_' + k: v for k, v in h_loss_info.items()
        }
        loss_info.update({
            'tail_' + k: v for k, v in t_loss_info.items()
        })
        loss_info['total_auxiliary_loss'] = total_aux_loss.item()

        return total_aux_loss, loss_info

    def get_refined_embeddings(self, entity_ids):
        """
        获取实体的精细化嵌入，用于下游任务（对齐、检索等）

        Args:
            entity_ids: [batch_size] 实体ID

        Returns:
            refined_embeddings: dict with keys 'struct', 'visual', 'text'
        """
        if not self.use_knowledge_extractor:
            raise ValueError("Knowledge extractor is not enabled")

        # 获取原始嵌入
        struct_emb = self.ent_embeddings(entity_ids)
        img_emb = self.img_proj(self.img_embeddings(entity_ids))
        text_emb = self.text_proj(self.text_embeddings(entity_ids))

        # 通过知识提取器获取精细化嵌入
        refined, _ = self.knowledge_extractor(struct_emb, img_emb, text_emb)

        return refined
