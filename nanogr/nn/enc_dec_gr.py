from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache

from nanogr.functional.generation_trie import Trie


class SemanticIDEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        encoder: PreTrainedModel,
        decoder: PreTrainedModel,
        codebooks: Tensor,
        d_model: int,
        num_beams: int=10,
        loss_func: nn.Module=None,
        check_prefix: bool=True,
    ) -> None:
        super().__init__()
        self.num_beams = num_beams
        self.check_prefix = check_prefix
        self.loss_func = loss_func

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.total_vocab_size = num_hierarchies * num_embeddings_per_hierarchy

        self.encoder = encoder
        self.decoder = decoder

        self.token_embedding = nn.Embedding(self.total_vocab_size+1, d_model, padding_idx=0)
        self.pred_head = nn.ModuleList(
            [nn.Linear(d_model, num_embeddings_per_hierarchy) for _ in range(num_hierarchies)]
        )
        self.bos_token = nn.Parameter(torch.randn(1, d_model), requires_grad=True)

        if codebooks is not None:
            codebooks = codebooks.long()
        self.register_buffer("codebooks", codebooks, persistent=False)

        if self.check_prefix:
            sequences = sorted(codebooks.cpu().tolist())
            self.prefix_trie = Trie(sequences)
        else:
            self.prefix_trie = None

        self._strip_unused_t5_embeddings()

    def _strip_unused_t5_embeddings(self) -> None:
        """Drop unused token-embedding sub-modules from the wrapped T5 stacks."""      
        if hasattr(self.encoder, "shared"):
            self.encoder.shared = nn.Identity()
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "embed_tokens"):
            self.encoder.encoder.embed_tokens = nn.Identity()
        if hasattr(self.decoder, "embed_tokens"):
            self.decoder.embed_tokens = nn.Identity()

    def _hierarchy_offsets(self, device: torch.device) -> Tensor:
        return torch.arange(self.num_hierarchies, device=device)* self.num_embeddings_per_hierarchy + 1

    def _to_local_labels(self, token_ids: Tensor) -> Tensor:
        offsets = self._hierarchy_offsets(token_ids.device).unsqueeze(0)
        local_ids = token_ids - offsets
        if ((local_ids >= 0) & (local_ids < self.num_embeddings_per_hierarchy)).all():
            return local_ids
        return token_ids

    def _trie_allowed_mask(
        self,
        beam_ids_flat: Tensor,
        hierarchy: int,
    ) -> Tensor:
        device = beam_ids_flat.device
        batch_beam_size = beam_ids_flat.size(0)
        vocab_size = self.num_embeddings_per_hierarchy
        if self.prefix_trie is None:
            return torch.ones(batch_beam_size, vocab_size, dtype=torch.bool, device=device)

        offset = int(hierarchy) * vocab_size + 1
        prefix_list = beam_ids_flat.tolist()
        rows, cols = [], []
        for row_idx, prefix in enumerate(prefix_list):
            allowed_global_ids = self.prefix_trie.get(prefix)
            for global_id in allowed_global_ids:
                local_id = global_id - offset
                if 0 <= local_id < vocab_size:
                    rows.append(row_idx)
                    cols.append(local_id)

        mask_cpu = torch.zeros(batch_beam_size, vocab_size, dtype=torch.bool)
        if rows:
            mask_cpu[
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ] = True
        return mask_cpu.to(device=device, non_blocking=True)

    def encode(self, src: Tensor, src_mask: Tensor=None) -> Tuple[Tensor, Tensor]:
        src_embd = self.token_embedding(src)

        encoder_output = self.encoder(inputs_embeds=src_embd, attention_mask=src_mask)
        return encoder_output.last_hidden_state, src_mask

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_mask: Tensor,
        use_cache: bool=False,
        past_key_values: DynamicCache=None,
    ) -> Tensor:
        batch_size = memory.size(0)
        bos = self.bos_token.unsqueeze(0).expand(batch_size, 1, -1)

        if tgt is None or tgt.size(1) == 0:
            decoder_input = bos
        else:
            decoder_embeds = self.token_embedding(tgt)
            decoder_input = torch.cat([bos, decoder_embeds], dim=1)

        decoder_output = self.decoder(
            attention_mask=tgt_mask,
            inputs_embeds=decoder_input,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        if use_cache:
            return decoder_output.last_hidden_state, decoder_output.past_key_values
        
        return decoder_output.last_hidden_state

    def forward(
        self,
        input_ids: Tensor,
        encoder_attn_mask: Tensor,
        future_ids: Tensor=None,
        decoder_attn_mask: Tensor=None,
    ) -> Tensor:
        memory, memory_attn_mask = self.encode(
            src=input_ids,
            src_mask=encoder_attn_mask,
        )
        return self.decode(
            tgt=future_ids,
            memory=memory,
            tgt_mask=decoder_attn_mask,
            memory_mask=memory_attn_mask,
        )

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
        memory, memory_attn_mask = self.encode(src=input_ids, src_mask=attention_mask)

        batch_size = input_ids.size(0)
        beam_ids = torch.empty(batch_size, 1, 0, dtype=torch.long, device=input_ids.device)
        beam_scores = torch.zeros(batch_size, 1, device=input_ids.device)
        for hierarchy in range(self.num_hierarchies):
            beam_width = beam_ids.size(1)
            flat_prefix = beam_ids.reshape(batch_size * beam_width, hierarchy)
            repeated_memory = memory.repeat_interleave(beam_width, dim=0)
            repeated_mask = memory_attn_mask.repeat_interleave(beam_width, dim=0)
            hierarchy_offset = self._hierarchy_offsets(input_ids.device)[hierarchy]

            decoder_output = self.decode(
                tgt=flat_prefix if hierarchy > 0 else None,
                memory=repeated_memory,
                tgt_mask=None,
                memory_mask=repeated_mask,
                use_cache=False,
                past_key_values=None,
            )
            logits = self.pred_head[hierarchy](decoder_output[:, -1]).view(
                batch_size,
                beam_width,
                self.num_embeddings_per_hierarchy,
            )
            log_probs = F.log_softmax(logits, dim=-1)

            if self.check_prefix and self.prefix_trie is not None:
                V = self.num_embeddings_per_hierarchy
                if hierarchy == 0:
                    offset_0 = 0 * V + 1
                    allowed_root = [g - offset_0 for g in self.prefix_trie.get([]) if 0 <= g - offset_0 < V]
                    root_mask = torch.zeros(V, dtype=torch.bool, device=input_ids.device)
                    if allowed_root:
                        root_mask[
                            torch.tensor(
                                allowed_root, dtype=torch.long, device=input_ids.device
                            )
                        ] = True
                    log_probs = log_probs.masked_fill(~root_mask.view(1, 1, V), float("-inf"))
                else:
                    trie_mask = self._trie_allowed_mask(flat_prefix, hierarchy)
                    log_probs = log_probs.view(batch_size * beam_width, V).masked_fill(~trie_mask, float("-inf")).view(batch_size, beam_width, V)

            combined_scores = beam_scores.unsqueeze(-1) + log_probs
            flat_scores = combined_scores.view(batch_size, -1)
            next_beam_width = min(self.num_beams, flat_scores.size(1))
            next_scores, next_indices = torch.topk(flat_scores, k=next_beam_width, dim=-1)

            parent_indices = next_indices // self.num_embeddings_per_hierarchy
            token_indices = next_indices % self.num_embeddings_per_hierarchy
            if hierarchy > 0:
                gather_indices = parent_indices.unsqueeze(-1).expand(-1, -1, hierarchy)
                next_beam_ids = torch.gather(beam_ids, dim=1, index=gather_indices)
            else:
                next_beam_ids = beam_ids.new_empty(batch_size, next_beam_width, 0)

            global_token_indices = token_indices + hierarchy_offset
            beam_ids = torch.cat([next_beam_ids, global_token_indices.unsqueeze(-1)], dim=-1)
            beam_scores = next_scores

        return beam_ids, beam_scores.exp()
    
    def compute_loss(self, decoder_hidden_states: Tensor, future_ids: Tensor) -> Tensor:
        decoder_hidden_states = decoder_hidden_states[:, :-1]
        local_future_ids = self._to_local_labels(future_ids)
        loss = 0
        for hierarchy in range(self.num_hierarchies):
            logits = self.pred_head[hierarchy](decoder_hidden_states[:, hierarchy])
            loss += self.loss_func(logits, local_future_ids[:, hierarchy].long())
        return loss
