from transformers import Trainer, TrainingArguments
import torch

class Direc_TrainingArguments(TrainingArguments):
    def __init__(self, ortho_lambda=4e-4, **kwargs):
        super().__init__(**kwargs)
        self.ortho_lambda = ortho_lambda

class Direc_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 首先调用父类的 compute_loss 获取原始任务损失
        task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # 计算正交约束损失
        ortho_loss = self.calc_ortho(model)
        
        if ortho_loss is not None:
            if int(self.args.ortho_lambda) == 1:
                # 帕累托优化实现
                # 使用一种动态权重调整方法来平衡两个损失
                # 计算两个损失的相对大小
                ratio = ortho_loss.detach() / (task_loss.detach() + 1e-8)  # 避免除以零
                
                # 使用softmax-like方法计算动态权重
                alpha_task = torch.exp(-ratio) / (torch.exp(-ratio) + torch.exp(-1/ratio + 1e-8))
                alpha_ortho = 1.0 - alpha_task
                
                # 计算帕累托优化后的总损失
                total_loss = alpha_task * task_loss + alpha_ortho * ortho_loss
            else:
                # 非帕累托优化的情况，保持原有加权方式
                total_loss = task_loss + self.args.ortho_lambda * ortho_loss
        else:
            total_loss = task_loss
            
        return (total_loss, outputs) if return_outputs else total_loss

    
    def calc_ortho(self, model):
        ortho_loss = 0.0
        den = 0
        for name, param in model.named_parameters():
            if "Direc_Ur" in name:
                u = param
                iu = torch.eye(u.shape[1], device=u.device)
                iu.requires_grad = False
                u_loss = torch.norm(u.T @ u - iu, p="fro")
                ortho_loss += u_loss
                den += 1
            if "Direc_Vhr" in name:
                vh = param
                ivh = torch.eye(vh.shape[0], device=vh.device)
                ivh.requires_grad = False
                vh_loss = torch.norm(vh @ vh.T - ivh, p="fro")
                ortho_loss += vh_loss
                den += 1
        if den != 0:
            return ortho_loss / den
        else:
            return None