import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class LoopedLM(Qwen2ForCausalLM):

	def generate(self, input_ids=None, max_new_tokens=None, thinking_steps=50, **kwargs):
		# Step 1: Prime model with prompt
		outputs = super().forward(input_ids=input_ids, use_cache=True)
		past_key_values = outputs.past_key_values        
		last_token = input_ids[:, -1:]
		think_start_token = torch.tensor([[151648]], dtype=torch.long, device="cuda")  #220 - space, 151643-pad, 151648-<think>, 151649-</think>
		think_end_token = torch.tensor([[151649]], dtype=torch.long, device="cuda")  

		# Step 2: Run internal loop without generating tokens
		for _ in range(thinking_steps):
			outputs = super().forward(
				input_ids=last_token,  # Continue using cache
				past_key_values=past_key_values,
				use_cache=True
			)
			past_key_values = outputs.past_key_values
			#input_ids = torch.cat([input_ids, special_token], dim=1) #append

		# Step 3: Manual decoding loop (replace generate)
		"""
		input_ids = torch.cat([input_ids, special_token], dim=1)
		return super().generate(
			input_ids=input_ids,
			past_key_values=past_key_values,
			**kwargs
		)
		
		"""

		generated = []
		cur_token = think_end_token
		for _ in range(max_new_tokens):
			with torch.no_grad():
				outputs = super().forward(
					input_ids=cur_token,
					past_key_values=past_key_values,
					use_cache=True
				)
				logits = outputs.logits[:, -1, :]
				past_key_values = outputs.past_key_values

				next_token = torch.argmax(logits, dim=-1, keepdim=True)
				generated.append(next_token)
				cur_token = next_token

				if next_token.item() == 151643: #tokenizer.eos_token_id:
					break

		# Decode final output
		gen_ids = torch.cat(generated, dim=-1)
		return gen_ids
		
#========================================

# Alternative: Lightweight thinking without additional parameters
class LightweightThinkingModel(Qwen2ForCausalLM): #PreTrainedModel
	"""
	Zero-parameter thinking model that uses only the base model's capabilities
	This version works immediately without any training!

	def __init__(self, config, base_model_path: str, thinking_steps: int = 3):
		super().__init__(config)
		self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
		self.config = self.base_model.config
		self.thinking_steps = thinking_steps
	"""


	def lightweight_thinking(self, hidden_states: torch.Tensor, thinking_steps=30) -> torch.Tensor:
		"""
		Perform thinking using only attention mechanisms from the base model
		No additional parameters needed!
		"""
		batch_size, seq_len, hidden_size = hidden_states.shape
		
		# Use the model's own attention to create thinking steps
		for step in range(thinking_steps):
			# Self-attention for internal processing
			# This mimics what the model does naturally but in a loop
			attn_weights = torch.softmax(
				torch.matmul(hidden_states, hidden_states.transpose(-2, -1)) / (hidden_size ** 0.5),
				dim=-1
			)
			
			# Apply attention to create "thought"
			thought = torch.matmul(attn_weights, hidden_states)
			
			# Residual connection with decay
			decay_factor = 0.9 ** step  # Decreasing influence
			hidden_states = hidden_states * (1 - 0.1 * decay_factor) + thought * (0.1 * decay_factor)
		
		#print(hidden_states.shape, "thinking")
		return hidden_states
	


	def forward(self, input_ids=None, past_key_values=None, enable_thinking=True, **kwargs):
		# If past is given, only use the last token for decoding
		if past_key_values is not None:
			#input_ids = input_ids[:, -1:]  # Only last token for incremental generation
			pass

		# Call base forward with all kwargs (includes past_key_values)
		out = super().forward(
			input_ids=input_ids,
			past_key_values=past_key_values,
			output_hidden_states=True,    
			**kwargs
		)
		hidden_states = out.hidden_states[-1]

		# Apply lightweight thinking
		if enable_thinking:
			hidden_states = self.lightweight_thinking(hidden_states)
		#print(hidden_states.shape, " -- forward2", self) #[1, 60, 1536]

		logits = self.lm_head(hidden_states)

		return CausalLMOutputWithPast(
			logits=logits,
			past_key_values=out.past_key_values,
			hidden_states=hidden_states,  #out.hidden_states,  # Keep full list if needed
			attentions=out.attentions,
		)


	"""
	def generate(self, input_ids=None, max_new_tokens=256, **model_kwargs):
		# Use base model's generate but with our forward method
		outputs = super().generate(
			input_ids=input_ids,
			max_new_tokens=max_new_tokens,
			**model_kwargs,
		)
		return outputs
	"""


"""
def create_lightweight_thinking_model(model_path: str, thinking_steps: int = 3):	
	base_model = AutoModelForCausalLM.from_pretrained(model_path)
	config = base_model.config
	
	return LightweightThinkingModel(
		config=config,
		base_model_path=model_path,
		thinking_steps=thinking_steps
	)
"""
# ./EndOf Lightweight thinking without additional parameters