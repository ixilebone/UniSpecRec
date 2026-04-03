import torch
from typing import Literal
import torch.nn.functional as F
    


def cal_bpr_loss(user_embedding: torch.Tensor, pos_item_embedding: torch.Tensor, neg_item_embeddings: torch.Tensor, similarity: Literal["cos", "dot"] = "dot"):
    """
    计算 BPR Loss 的最大版本（仅使用最难负样本）
    
    Args:
        user_embedding: [Batch, Dim] - 用户嵌入
        pos_item_embedding: [Batch, Dim] - 正样本物品嵌入
        neg_item_embeddings: [Batch, K, Dim] - K 个负样本物品嵌入
        similarity: "dot" 使用点积，"cos" 使用余弦相似度 (默认 "dot")
    
    Returns:
        loss: scalar - BPR loss (mean over batch)
    """

    match similarity:
        case "cos":
            user_norm = F.normalize(user_embedding, dim=-1)
            pos_norm = F.normalize(pos_item_embedding, dim=-1)
            neg_norm = F.normalize(neg_item_embeddings, dim=-1)

            pos_score = (user_norm * pos_norm).sum(-1)  # [Batch]
            neg_score = (user_norm.unsqueeze(1) * neg_norm).sum(-1)  # [Batch, K]
        case "dot":
            pos_score = (user_embedding * pos_item_embedding).sum(-1)  # [Batch]
            neg_score = (user_embedding.unsqueeze(1) * neg_item_embeddings).sum(-1)  # [Batch, K]
        case _:
            raise ValueError(f"Unsupported similarity type: {similarity}")

    hardest_neg_score, _ = neg_score.max(dim=1)  # [Batch]

    loss = -F.logsigmoid(pos_score - hardest_neg_score).mean()
    return loss

def cal_infonce_loss(user_embedding: torch.Tensor, pos_item_embedding: torch.Tensor, neg_item_embeddings: torch.Tensor = None, similarity: Literal["cos", "dot"] = "dot", temperature=0.5, is_inbatch: bool = False, pos_item_index: torch.Tensor = None):
    """
    计算 InfoNCE Loss
    
    Args:
        user_embedding: [Batch, Dim] - 用户嵌入
        pos_item_embedding: [Batch, Dim] - 正样本物品嵌入
        neg_item_embeddings: [Batch, K, Dim] - K 个负样本物品嵌入 (非 inbatch 模式必需)
        similarity: "cos" 使用余弦相似度, "dot" 使用点积 (默认 "dot")
        temperature: 温度系数 tau (默认 0.5)
        is_inbatch: 是否使用 inbatch 模式 (默认 False)
        pos_item_index: [Batch] - 正样本物品的全局唯一 ID (inbatch 模式必需)
    
    Returns:
        loss: scalar - InfoNCE loss (mean over batch)
    """
    if is_inbatch:
        match similarity:
            case "cos":
                normed_user = F.normalize(user_embedding, p=2, dim=-1)
                normed_pos = F.normalize(pos_item_embedding, p=2, dim=-1)
                logits = torch.matmul(normed_user, normed_pos.T) / temperature
            
            case "dot":
                logits = torch.matmul(user_embedding, pos_item_embedding.T) / temperature
            
            case _:
                raise ValueError(f"Unsupported similarity type: {similarity}")
        
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
    
    else:
        all_item_embeddings = torch.cat([pos_item_embedding.unsqueeze(1), neg_item_embeddings], dim=1)
        
        match similarity:
            case "cos":
                normed_user = F.normalize(user_embedding, dim=-1)  # [Batch, Dim]
                normed_all = F.normalize(all_item_embeddings, dim=-1)  # [Batch, K+1, Dim]
                logits = torch.bmm(normed_user.unsqueeze(1), normed_all.transpose(1, 2)).squeeze(1) / temperature
            
            case "dot":
                logits = torch.bmm(user_embedding.unsqueeze(1), all_item_embeddings.transpose(1, 2)).squeeze(1) / temperature
            
            case _:
                raise ValueError(f"Unsupported similarity type: {similarity}")
        
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
    
    return loss