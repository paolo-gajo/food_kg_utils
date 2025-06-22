from typing import Optional, Tuple
import torch
from torch import nn
from transformers.models.gpt_neox import modeling_gpt_neox
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from model.llm_annotator.italia_config import ItaliaConfig

# inject a GPTNeoXLayer no post layer norm
class GPTNeoXLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = modeling_gpt_neox.GPTNeoXAttention(config, layer_idx)
        self.mlp = modeling_gpt_neox.GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        
        attn_output, attn_weights = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output_dropout = self.post_attention_dropout(attn_output[0])

        # self.use_parallel_residual: default true
        # x = x + attn(ln1(x)) + mlp(ln1(x))
        mlp_output = self.mlp(self.input_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output_dropout + hidden_states

        if use_cache:
            outputs = (hidden_states,) + attn_output  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states, attn_weights)  # hidden_states, (attn_weights)

        return outputs

modeling_gpt_neox.GPTNeoXLayer = GPTNeoXLayer

from transformers.models.gpt_neox.modeling_gpt_neox import  GPTNeoXForCausalLM, GPTNeoXModel

class ItaliaForCausalLM(GPTNeoXForCausalLM):


    config_class = ItaliaConfig

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig