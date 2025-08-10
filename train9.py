# T5 [ qemb, sembs] -> label(t1,t2,t3,...)
# preprocessing is done in train4 
# venv: US1-asr3.12

import json, os, random, time
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration

#============================================================================
from utils import Sonar, NLP, JinaAI, get_magnitudes
from modeling import get_averaged_layers, freeze_some_layers, find_subsequence
#============================================================================

class EmbDataCollator:
	def __init__(self):
		self.pad_value = 0.0

	def __call__(self, batch):
		# emb(question, sentences)
		inputs_embeds = [torch.tensor([x['question_emb']] + x['sentences_emb']) for x in batch]
		att_s = [ torch.ones(x.size(0))  for x in inputs_embeds]
		att_s = pad_sequence(att_s, batch_first=True, padding_value=0)
		att_s = att_s.ne(0) #B,S
		inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=self.pad_value) #B,S,C

		# labels
		out = tokenizer([random.choice(x['answers']) for x in batch], return_tensors="pt", padding=True, return_attention_mask=True, add_special_tokens=True)
		labels = out.input_ids

		out = {'inputs_embeds': inputs_embeds, 'attention_mask':att_s, 'labels':labels}
		if mode==2: #test
			out["question"] = [x["question"] for x in batch]
			out["answers"] = [x["answers"] for x in batch]
		return out


class MyModel(T5ForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.embedding_dim = 1024 #jinaai:1024, sonar:1024
		self.hidden_dim = 768 #t5-base:768, small:512
		#self.ln1 = nn.LayerNorm(self.embedding_dim)
		#self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
		self.sent_proj = nn.Sequential(
			nn.Linear(self.embedding_dim, 4864, bias=False),
			nn.GELU(),
			nn.Linear(4864, self.hidden_dim, bias=False)
		)		
					
	def trans(self, inputs_embeds):
		x = self.sent_proj(inputs_embeds) #self.fc1(self.ln1(inputs_embeds))
		if torch.isnan(x).any().item() or torch.isnan(inputs_embeds).any().item():
			print(get_magnitudes([x, inputs_embeds]), inputs_embeds.shape); exit()
		return x

	def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds), attention_mask=attention_mask, labels=labels) #, **kwargs
		return outputs

	def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=32):
		encoder_outputs = self.encoder(inputs_embeds=self.trans(inputs_embeds), attention_mask=attention_mask)
		decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
		generated = []
		for i in range(max_new_tokens):
			outputs = super().forward(
			    encoder_outputs=encoder_outputs,
			    decoder_input_ids=decoder_input_ids,
			    attention_mask=attention_mask			    
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
				gen_ids = self.model.generate(inputs_embeds=inputs['inputs_embeds'])
				generated = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
				print(generated, "ans:", inputs['answers'], "\n\n")				
		return {"accuracy": 1.0}


#==============================================================================================
if __name__ == "__main__":
	device = torch.device("cuda")
	mode = 1 #1-train,2-test
	model_id = "t5-base"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	
	# Dataset
	dataset = concatenate_datasets([ load_from_disk("./temp/"+fname) for fname in ["narrativeqa_train_20k", "narrativeqa_train_20k_2", "hotpotqa_train_jinaai_1", "hotpotqa_train_jinaai_2"] ])
	#dataset = dataset.map(postprocess) #set "len"
	dataset = dataset.train_test_split(test_size=0.005, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if mode==1: #training
		model = MyModel.from_pretrained(model_id)		
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-")
		model.eval()
	
	# Start training
	data_collator = EmbDataCollator()

	print("starting", "Traininig" if mode==1 else "Testing")
	training_args = TrainingArguments(
		output_dir='./model_temp',
	  	#group_by_length=True, length_column_name="len",
		num_train_epochs=20,
		per_device_train_batch_size=8,
		gradient_accumulation_steps=2,
		learning_rate=1e-5,
		logging_steps=20,
		save_total_limit=2,
		save_steps=2000,
		eval_steps=2000,
		eval_strategy="steps",		
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.005,
		warmup_steps=100,
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
	)
	
	if mode==1:
		trainer.train("./model_temp/checkpoint-86000")
	else:
		print(trainer.predict(test_dataset))

