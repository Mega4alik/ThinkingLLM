# ThinkingLLM
 thinking models

# helpful info
Qwen2ForCausalLM(
  (model): Qwen2Model(
	(embed_tokens): Embedding(151936, 1536)
	(layers): ModuleList(
	  (0-27): 28 x Qwen2DecoderLayer(
		(self_attn): Qwen2Attention(
		  (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
		  (k_proj): Linear(in_features=1536, out_features=256, bias=True)
		  (v_proj): Linear(in_features=1536, out_features=256, bias=True)
		  (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
		)
		(mlp): Qwen2MLP(
		  (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
		  (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
		  (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
		  (act_fn): SiLU()
		)
		(input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
		(post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
	  )
	)
	(norm): Qwen2RMSNorm((1536,), eps=1e-06)
	(rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)



# Qwen2 modificaton for Looping
Qwen2DecoderLayer  vs Qwen2Model forward update to loop multiple times
https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen2/modeling_qwen2.py#L347
/home/mega4alik/Desktop/environments_venv/asr3.12/lib/python3.12/site-packages/transformers/models/qwen2/
! Qwen2Model added at line:434 add
		#temp edit        
		for loop in range(1): #meine1 looping the whole stack multiple times -- this does not seem to work            
			for decoder_layer in self.layers[: self.config.num_hidden_layers]:
				if output_hidden_states:
					all_hidden_states += (hidden_states,)                
				layer_outputs = decoder_layer(
					hidden_states,
					attention_mask=causal_mask if loop==0 else None, #meine1
					position_ids=position_ids,
					past_key_value=past_key_values,
					output_attentions=output_attentions,
					use_cache=use_cache,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
					**flash_attn_kwargs,
				)

				hidden_states = layer_outputs[0]

				if output_attentions:
					all_self_attns += (layer_outputs[1],)
		#./endOf temp edit



# Datasets for Looping:
- https://huggingface.co/datasets/allenai/openbookqa
- https://huggingface.co/datasets/ChilleD/MultiArith
- https://huggingface.co/datasets/mwpt5/MAWPS
- https://huggingface.co/datasets/Maurus/ToolBench

#===============================

# Qwen/Qwen2-0.5B-Instruct Classification on pminervini/HaluEval.qa_samples ft
## m1. layers=4(averaged from 24), Loop=4, prompt.tokens <= 200, best checkpoint-4000:
TP: 43, FP: 0, TN: 45, FN: 2
Precision: 1.0000
Recall:    0.9556
F1 Score:  0.9773

## m2. as m1,  Loop=1, epochs=10, checkpoint-4000 and 9500 (same results as m1)
TP: 43, FP: 0, TN: 45, FN: 2; Precision:1.0000; Recall:0.9556; F1 Score:0.9773


