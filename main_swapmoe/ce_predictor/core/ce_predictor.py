import torch
from torch import nn
import torch.nn.functional as F


class CEPredictor(nn.Module):
    def __init__(self, layers, input_dim, num_experts):
        super(CEPredictor, self).__init__()
        self.layers = layers
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim, num_experts * 2, bias=True)
                for _ in range(layers)
            ])
            for _ in range(layers)
        ])

    def forward(self, layer_inputs):
        L, N, _ = layer_inputs.shape
        assert L == self.layers, "layer_inputs 第一维必须等于 self.layers"

        outputs = []
        for s in range(L):
            x_s = layer_inputs[s]
            row = []
            for t in range(L):
                logits_st = self.predictors[s][t](x_s)
                row.append(logits_st)
            row = torch.stack(row, dim=0)
            outputs.append(row)

        outputs = torch.stack(outputs, dim=0)
        return outputs

    def predict(self, x, input_layer, target_layer, topk, expert_num):
        logits = self.predictors[input_layer][target_layer](x)
        output = CEPredictor.cal_pred_index(logits, topk, expert_num)
        return output

    @classmethod
    def cal_loss(
        cls,
        ce_loss,
        router_logits,
        pred_logits,
        expert_num=60,
        activated_expert_num=4,
        is_router_index=False,
    ):
        device = pred_logits.device
        L, L2, N, twoE = pred_logits.shape
        assert L == L2, "pred_logits 的前两维都应该是 layers"
        assert twoE == expert_num * 2, "最后一维应该是 num_experts * 2"

        pred_logits = pred_logits.view(L, L, N, expert_num, 2)

        if is_router_index:
            router_index = router_logits
        else:
            _, router_index = torch.topk(router_logits, activated_expert_num, dim=-1)

        router_mask = torch.zeros(L, N, expert_num, dtype=torch.long, device=device)
        router_mask.scatter_(2, router_index, 1)
        router_label_full = router_mask.unsqueeze(0).expand(L, -1, -1, -1).contiguous()

        pred_flat = pred_logits.reshape(-1, 2)
        label_flat = router_label_full.reshape(-1)
        loss = ce_loss(pred_flat, label_flat)
        return loss

    @classmethod
    def cal_pred_index(cls, pred_logits, topk=4, expert_num=60):
        pred_logits = pred_logits.reshape(-1, 2)
        pred_logits = F.softmax(pred_logits, dim=-1)
        pos_logits = pred_logits[:, 1]
        pos_logits = pos_logits.reshape(-1, expert_num)
        _, pred_index = torch.topk(pos_logits, topk, dim=-1)
        return pred_index

    @classmethod
    def get_loss_func(cls, reduction="mean"):
        return nn.CrossEntropyLoss(reduction=reduction)
