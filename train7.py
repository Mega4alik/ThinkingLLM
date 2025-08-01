# T5 [ CNN(t1, t2 t3), t4, t5,t6,t7] -> [t5,t6,t7] 
# preprocessing is done in train4 
# venv: US1-asr3.12

import json, os, random, time
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import Sonar, NLP, JinaAI, get_magnitudes
from modeling import get_averaged_layers, freeze_some_layers, find_subsequence


def postprocess(row):
	row["len"] = len(row["sentences"])
	return row

class ConvDataCollator:
	def __init__(self):
		self.kernel_size, self.stride = 3, 3		

	def apply_template(self, question):
		return f"question: {question} context: "

	def __call__(self, batch):
		# conv part - sents
		out = tokenizer([" ".join(x['sentences']) for x in batch], return_tensors="pt", padding=True, return_attention_mask=True, add_special_tokens=False) #B,S
		sent_ids, att_s = out.input_ids, out.attention_mask
		att_s = att_s.unsqueeze(1).float()  # (B, 1, T)
		pooled = nn.functional.max_pool1d(att_s, self.kernel_size, stride=self.stride)
		att_s = pooled.squeeze(1).bool()  # (B, T')

		#tokens
		st = [self.apply_template(x['question']) for x in batch]
		out = tokenizer(st, return_tensors="pt", padding=True, return_attention_mask=True, add_special_tokens=False)
		token_ids, att_t = out.input_ids, out.attention_mask
		
		# labels
		out = tokenizer([random.choice(x['answers']) for x in batch], return_tensors="pt", padding=True, return_attention_mask=True, add_special_tokens=True)
		labels = out.input_ids
		
		out = {'sent_ids': sent_ids, 'token_ids': token_ids, 'att_s':att_s, 'att_t':att_t, 'labels':labels}
		if mode==2: #test
			out["question"] = [x["question"] for x in batch]
			out["answers"] = [x["answers"] for x in batch]
		return out


class MyModel(T5ForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.hidden_dim = 512 #t5-small
		self.kernel_size, self.stride = 3, 3
		self.embed_types = nn.Embedding(2, self.hidden_dim) #number of types, hidden_dim
		self.conv = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=0) #v1
		"""
		self.conv = nn.Sequential( #v1.2
			nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=0),
			nn.ReLU()		
		)
		"""
	
	def trans(self, sent_ids, token_ids):
		tokens_embeds = self.encoder.embed_tokens(token_ids)
		x1 = self.encoder.embed_tokens(sent_ids)
		x = x1.transpose(1, 2)
		x = self.conv(x)  # (B, C, T') where T' = (T - 3)//3 + 1
		inputs_embeds = x.transpose(1, 2)  # back to (B, T', C)
		x = torch.cat([tokens_embeds, inputs_embeds], dim=1)

		B, S, T = inputs_embeds.size(0), inputs_embeds.size(1), token_ids.size(1)
		type_ids = torch.cat([torch.zeros((B, T), dtype=torch.long), torch.ones((B, S), dtype=torch.long)], dim=1).to(device)
		type_embeds = self.embed_types(type_ids)
		x = x + type_embeds

		if torch.isnan(inputs_embeds).any().item():
			print(get_magnitudes([x1, inputs_embeds, tokens_embeds, type_embeds]), sent_ids.shape); exit()
		return x


	def forward(self, sent_ids=None, token_ids=None, att_s=None, att_t=None, labels=None, **kwargs):
		#print("\nsent_ids", sent_ids.shape, sent_ids, "\ninputs_embeds", inputs_embeds.shape, "att_s", att_s.shape, att_s, "\ntoken_ids", token_ids, "\natt_t", att_t); exit()
		attention_mask = torch.cat([att_t, att_s], dim=1)   #B,S+T,C
		outputs = super().forward(inputs_embeds=self.trans(sent_ids, token_ids), attention_mask=attention_mask, labels=labels) #, **kwargs
		return outputs


	def generate(self, sent_ids=None, token_ids=None, att_s=None, att_t=None, max_new_tokens=32):
		attention_mask = torch.cat([att_t, att_s], dim=1)   #B,S+T,C		
		encoder_outputs = self.encoder(inputs_embeds=self.trans(sent_ids, token_ids), attention_mask=attention_mask)
		decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
		generated = []
		for i in range(max_new_tokens):
			outputs = super().forward(
			    encoder_outputs=encoder_outputs,
			    decoder_input_ids=decoder_input_ids,
			    attention_mask=attention_mask,
			    #max_new_tokens=32
			)
			next_token = torch.argmax(outputs.logits[:,-1,:], dim=-1, keepdim=True)
			generated.append(next_token)
			if next_token.item() == tokenizer.eos_token_id: break #15164
			decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
		return torch.cat(generated, dim=-1)


class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				gen_ids = self.model.generate(sent_ids=inputs['sent_ids'], token_ids=inputs['token_ids'], att_s=inputs['att_s'], att_t=inputs['att_t'])
				generated = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
				print(generated, "ans:", inputs['answers'], "\n\n")
				#compute_metrics( (preds, inputs['answers'], inputs['question']) )
		return {"accuracy": 1.0}



#==============================================================================================
if __name__ == "__main__":	
	device = torch.device("cuda")
	mode = 1 #1-train,2-test
	model_id = "t5-small"
	tokenizer = T5Tokenizer.from_pretrained(model_id)
	
	# Dataset
	#dataset = load_from_disk("./temp/narrativeqa_train_20k")
	dataset = concatenate_datasets([ load_from_disk("./temp/"+fname) for fname in ["narrativeqa_train_20k", "narrativeqa_train_20k_2"] ])
	#dataset = dataset.map(postprocess) #set "len"

	dataset = dataset.train_test_split(test_size=0.005, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if mode==1: #training
		model = MyModel.from_pretrained(model_id)
		#model.model.embed_tokens.requires_grad_(False)
		import math; nn.init.kaiming_uniform_(model.conv.weight, a=math.sqrt(6))
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-50000")
		model.eval()
	
	# Start training
	data_collator = ConvDataCollator()

	print("starting", "Traininig" if mode==1 else "Testing")
	training_args = TrainingArguments(
		output_dir='./model_temp',
	  	#group_by_length=True, length_column_name="len",
		num_train_epochs=50,
		per_device_train_batch_size=16,
		gradient_accumulation_steps=1,
		learning_rate=1e-5,
		logging_steps=20,
		save_total_limit=2,
		save_steps=1000,
		eval_steps=1000,
		eval_strategy="steps",		
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.005,
		warmup_steps=100,
		#fp16=True,
		#disable_tqdm=True,
		report_to="none" #"tensorboard",
	)

	trainer = OwnTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
	)
	
	if mode==1:
		trainer.train("./model_temp/checkpoint-1000") # "./model_temp/checkpoint-200"
	else:
		print(trainer.predict(test_dataset))

