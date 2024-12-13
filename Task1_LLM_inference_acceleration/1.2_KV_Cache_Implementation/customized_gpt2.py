# from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

prefix = {}
class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None, #
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False, #
        **kwargs,
    ):
        
        # print(f"hidden_states: {hidden_states.shape}")
        # print(f"query: {query.shape}")
        # Prepare query, key, value matrix
        
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # each of them has shape (batch_size, seq_len, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        # print(f"key.shape: {key.shape}")
        # KV Cache
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            # print(f"cache: {past_key.shape}")
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        # print(f"key.shape + cache: {key.shape}")
        present = None
        if use_cache is True:
            present = (key, value)
            
        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        # if layer_past is None:
        #     print(attn_weights.shape)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        return outputs 


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None, #
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False, #
        **kwargs,
    ):
        residual = hidden_states

        # self-attention (class `CustomizedGPT2AttentionWithFasterCache`)
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past, #
            attention_mask=attention_mask,
            use_cache=use_cache, # 
        )

        # 有一部分是KV Cache
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # residual connection
        hidden_states = attn_output + residual
        # print(hidden_states.shape)

        residual = hidden_states

        # feed-forward
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) 

        return outputs


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, #
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None, #
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Positional embedding
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        
        # Prepare input embeddings
        inputs_embeds = None
        output_shape = None
        hidden_states = None
        if past_key_values is not None:
            # 我们只需要新token和最后一个位置的PE
            # print(position_embeds.shape)
            inputs_embeds = self.wte(input_ids[:,-1])
            # print(position_embeds[:,-1,:].shape)
            hidden_states = (inputs_embeds + position_embeds[:,-1,:]).unsqueeze(1)
            output_shape = (-1,) + (1,) + (hidden_states.size(-1),)
        else:
            # 如要实现prefix cache，可以在这里检索一个prefix字典。如有重合，则将input_ids拆分成prefix_ids和other_ids
            # prefix_ids对应的KV赋值给past_key_values，other_ids做embedding然后作为hidden_states, 计算KV
            # 两者在CustomizedGPT2Attention中拼接起来
            inputs_embeds = self.wte(input_ids)
            hidden_states = inputs_embeds + position_embeds
            output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        

        # Prepare Attention mask.
        
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        
        attention_mask = attention_mask[:, None, None, :]
        
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        # print(attention_mask.shape)
        hidden_states = self.drop(hidden_states)
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        presents = () if use_cache else None
        # Iterate over all GPT2 layer, i.e. `block`
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past, #
                attention_mask=attention_mask,
                use_cache=use_cache, #
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        
        hidden_states = hidden_states.view(output_shape)
        outputs = (hidden_states, presents)
            
        return outputs


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, #
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None, #
    ):
        # if past_key_values is not None :
        #     print(len(past_key_values[0]))
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values, #
            attention_mask=attention_mask,
            use_cache=use_cache, #
        )

        hidden_states = transformer_outputs[0] #[batchsize, prompt_len or 1, hidden_size]
        # 如果要实现Prefix Cache，需要让transformer_outputs额外返回一个prefix KV，然后存到prefix字典中
        
        # 拼接了新token的每一层的KV Cache
        KV_Cache = transformer_outputs[1]
        # Prepare logits from last hidden state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'KV_Cache': KV_Cache,
        }