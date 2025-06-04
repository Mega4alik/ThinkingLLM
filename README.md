# ThinkingLLM
 thinking models



Qwen2DecoderLayer  vs Qwen2Model forward update to loop multiple times
https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen2/modeling_qwen2.py#L347
/home/mega4alik/Desktop/environments_venv/asr3.12/lib/python3.12/site-packages/transformers/models/qwen2/



#===============================
LoopedLM(
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




Dataset for Looping:
- https://huggingface.co/datasets/allenai/openbookqa
- https://huggingface.co/datasets/ChilleD/MultiArith
- https://huggingface.co/datasets/mwpt5/MAWPS
- https://huggingface.co/datasets/Maurus/ToolBench



Qwen/Qwen2-0.5B-Instruct classification on pminervini/HaluEval layers=4(averaged), Loop=4, checkpoint-4000:
TP: 43, FP: 0, TN: 45, FN: 2
Precision: 1.0000
Recall:    0.9556
F1 Score:  0.9773