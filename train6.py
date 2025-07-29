# [ CNN(t1, t2 t3), t4, t5,t6,t7] -> [t5,t6,t7]
# preprocessing is don in train4 
# venv: US1-asr3.12

import json, os, random, time
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, TrainerCallback
from transformers import Qwen2Model,Qwen2ForCausalLM
from utils import Sonar, NLP, JinaAI, get_magnitudes
from modeling import get_averaged_layers, freeze_some_layers, find_subsequence

class ConvDataCollator:
	def __init__(self):
		self.kernel_size, self.stride = 3, 3		

	def apply_template(self, question, answer):
		return f"<|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n" + (f"{answer}<|im_end|>" if mode==1 else "") #<|endoftext|>?

	def __call__(self, batch):
		# conv part - sents
		out = tokenizer([" ".join(x['sentences']) for x in batch], return_tensors="pt", padding=True, return_attention_mask=True) #B,S
		sent_ids, att_s = out.input_ids, out.attention_mask		
		att_s = att_s.unsqueeze(1).float()  # (B, 1, T)
		pooled = nn.functional.max_pool1d(att_s, self.kernel_size, stride=self.stride)
		att_s = pooled.squeeze(1).bool()  # (B, T')
		S2 = att_s.size(1)

		#tokens
		st = [self.apply_template(x['question'], random.choice(x['answers'])) for x in batch]	
		out = tokenizer(st, return_tensors="pt", padding=True, return_attention_mask=True)
		token_ids, att_t = out.input_ids, out.attention_mask
		
		# labels
		labels = []
		for t_ids in token_ids:
			answer_start_id = find_subsequence(t_ids, [77091, 198]) #assistant\n			
			label = [-100] * S2 + [-100] * (answer_start_id+1) + [-100 if v==151643 else v for v in t_ids[answer_start_id+1+1:].tolist()] + [-100]
			#print(t_ids, label[inputs_embeds.size(1):], att_t[len(labels)])
			labels.append(label)
		labels = torch.tensor(labels, dtype=torch.long)

		#print("inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids,"\natt_t", att_t, "\nlabels", labels); exit()
		out = {'sent_ids': sent_ids, 'token_ids': token_ids, 'att_s':att_s, 'att_t':att_t, 'labels':labels}
		if mode==2: #test
			out["question"] = [x["question"] for x in batch]
			out["answers"] = [x["answers"] for x in batch]			
		return out


class MyModel(Qwen2ForCausalLM):
	def __init__(self, config):
		super().__init__(config)		
		self.hidden_dim = 896 #qwen:896
		self.kernel_size, self.stride = 3, 3
		self.embed_types = nn.Embedding(2, self.hidden_dim) #number of types, hidden_dim		
		self.conv = nn.Conv1d(
			in_channels=self.hidden_dim,  # C
			out_channels=self.hidden_dim, # or change if you want to map to different dim
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=0
		)		
	
	def trans(self, inputs_embeds, token_ids):
		tokens_embeds = self.model.embed_tokens(token_ids)
		if inputs_embeds is not None:			
			x = torch.cat([inputs_embeds, tokens_embeds], dim=1)
			#B, S, T = inputs_embeds.size(0), inputs_embeds.size(1), token_ids.size(1)
			#type_ids = torch.cat([torch.zeros((B, S), dtype=torch.long), torch.ones((B, T), dtype=torch.long)], dim=1).to(device)
		else: #not considering this case yet
			pass			

		#type_embeds = self.embed_types(type_ids)
		#x = x + type_embeds
		#print("\nmagnitudes:", get_magnitudes([inputs_embeds, tokens_embeds])); exit() #, type_embeds
		return x

	def forward(self, sent_ids=None, token_ids=None, att_s=None, att_t=None, labels=None, **kwargs):
		x1 = self.model.embed_tokens(sent_ids)
		x = x1.transpose(1, 2)
		x = self.conv(x)  # (B, C, T') where T' = (T - 3)//3 + 1
		inputs_embeds = x.transpose(1, 2)  # back to (B, T', C)
		print(get_magnitudes([x1, inputs_embeds]), sent_ids.shape); exit()
		#print("\nsent_ids", sent_ids.shape, sent_ids, "\ninputs_embeds", inputs_embeds.shape, "att_s", att_s.shape, att_s, "\ntoken_ids", token_ids, "\natt_t", att_t); exit()		
		attention_mask = torch.cat([att_s, att_t], dim=1)   #B,S+T,C
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds, token_ids), attention_mask=attention_mask, labels=labels) #, **kwargs
		return outputs

	def generate(self, sent_ids=None, token_ids=None, att_s=None, max_new_tokens=32): #primitive version without caching       
		generated = []
		for i in range(max_new_tokens):
			att_t = torch.tensor( [[True] * token_ids.size(1)], device=device)
			attention_mask = torch.cat([att_s, att_t], dim=1).to(device) #B,S+T,C
			outputs = self.forward(sent_ids=sent_ids, token_ids=token_ids, att_s=att_s, att_t=att_t)
			#past_key_values = outputs.past_key_values
			logits = outputs.logits[:, -1, :]
			next_token = torch.argmax(logits, dim=-1, keepdim=True)
			generated.append(next_token)
			if next_token.item() == tokenizer.eos_token_id: break #15164
			token_ids = torch.cat([token_ids, next_token], dim=1)
		
		return torch.cat(generated, dim=-1)



class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				gen_ids = self.model.generate(sent_ids=inputs['sent_ids'], token_ids=inputs['token_ids'], att_s=inputs['att_s'])
				generated = tokenizer.batch_decode(gen_ids)
				print(generated, "ans:", inputs['answers'], "\n\n")
				#compute_metrics( (preds, inputs['answers'], inputs['question']) )
		return {"accuracy": 1.0}



#==============================================================================================
if __name__ == "__main__":	
	device = torch.device("cuda")
	mode = 1 #1-train,2-test
	model_id = "Qwen/Qwen2-0.5B-Instruct" #Qwen/Qwen2-0.5B | Qwen/Qwen2-0.5B-Instruct | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | "meta-llama/Llama-3.2-1B-Instruct" | "meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	
	# Dataset
	if False:
		embedding_model = JinaAI()  #Sonar(1)
		nlp = NLP()
		dataset = load_dataset("deepmind/narrativeqa", split="train[20000:]") #"mwpt5/MAWPS" | "ChilleD/SVAMP"
		dataset = dataset.map(preprocess, batched=True, batch_size=12)
		print("preprocess finished")
		dataset = dataset.map(postprocess)
		dataset.save_to_disk("./temp/narrativeqa_train_20k_2"); exit()		
	else:
		#dataset = load_from_disk("./temp/narrativeqa_train_20k")
		dataset = concatenate_datasets([ load_from_disk("./temp/"+fname) for fname in ["narrativeqa_train_20k", "narrativeqa_train_20k_2"] ])
	
	dataset = dataset.train_test_split(test_size=0.005, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if mode==1: #first training
		model = MyModel.from_pretrained(model_id) #, torch_dtype=torch.float16
		#model.config.num_hidden_layers=12
		#model.model.layers = get_averaged_layers(model.model.layers, 12) #model.model.layers[:4]
		freeze_some_layers(model.model.layers, 2, 1)		
		#print(model, model.config); exit()
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-")
	
	# Start training
	data_collator = ConvDataCollator() #FT/PT

	print("starting", "Traininig" if mode==1 else "Testing")	
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=50,
		per_device_train_batch_size=2,
		gradient_accumulation_steps=2,
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=5000,
		save_total_limit=2,
		eval_strategy="steps",
		eval_steps=5000,
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.01,
		warmup_steps=1000,
		fp16=True,
		#disable_tqdm=True,
		report_to="none" #"tensorboard",
	)

	trainer = OwnTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		#compute_metrics=compute_metrics
	)
	
	if mode==1:
		trainer.train() #"./model_temp/checkpoint-18000"
	else:
		print(trainer.predict(test_dataset))

