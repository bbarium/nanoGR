import torch
import torch.nn as nn

class MTP(nn.Module):
    def __init__(
            self, 
            in_model: nn.Module,
            tokenization: nn.Module,
            hid_channels: int, 
            num_token: int,
            top_k: int,
            prefix_dict: dict,
        ):
        super().__init__()
        if in_model is not None:
            self.predictive_head = in_model
        else:
            self.predictive_head = nn.Linear(hid_channels, num_token*hid_channels) # N*d -> N*(T*d)

        self.top_k = top_k
        self.num_token = num_token
        self.tokenization = tokenization
        self.prefix_dict = prefix_dict
        self.reset_parameters()

    def reset_parameters(self):
        self.predictive_head.reset_parameters()

    def forward(self, in_embed: torch.Tensor, out_weight: torch.Tensor, token_space: torch.Tensor):
        multi_embd = self.predictive_head(in_embed)

        outputs, accumulation, token_candidate = [], None, None
        for i in range(self.num_token):
            embd = multi_embd[i]

            token_weights = torch.dist(embd, token_space).softmax(dim=-1)
            top_k_weights, top_k_indices = torch.topk(token_weights, dim=-1)

            if token_candidate is None:
                token_candidate = [self.prefix_dict[ids] for ids in top_k_indices if ids in self.prefix_dict.keys()]

            token_weights = []
            for j, ids in enumerate(top_k_indices):
                tmp = [token[0] for token in token_candidate]
                if ids not in tmp:
                   continue 
                
                token_candidate = [token[1:] for token in token_candidate]
                if len(outputs) == 0:
                    outputs = outputs.append(ids)
                else:
                    outputs = [outputs] # TODO: 这里还需要改成prefix的beam search结构
                token_weights.append(top_k_weights[j])
            
            if accumulation is None:
                accumulation = torch.tensor(token_weights, dtype=torch.float32, device=in_embed.device)
            else:
                accumulation += torch.tensor(token_weights, dtype=torch.float32, device=in_embed.device)
        return outputs, accumulation
