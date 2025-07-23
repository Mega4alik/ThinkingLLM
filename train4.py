# like lcm.py check for duplicates!
# [semb1, semb2, t1, t2, t3] -> [t2,t3,t4]
# venv: US1-asr3.12

import json, os, random, time
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, load_from_disk
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, TrainerCallback
from transformers import Qwen2Model,Qwen2ForCausalLM
from utils import Sonar, NLP
from modeling import get_averaged_layers

def preprocess(batch):
	sentences, answers, questions = [], [], []
	for i in range(len(batch["question"])):
		question = batch["question"][i]["text"]
		anss = [x["text"] for x in batch["answers"][i]]
		text = batch["document"][i]["summary"]["text"]
		sents = nlp.split_sentences(text)
		#print(text, sentences, len(text), len(sentences)); exit()
		questions.append(question)
		sentences.append(sents)
		answers.append(anss)
	return {"sentences":sentences, "answers":answers, "question":questions}


def postprocess(row):
	sentences, answers, question = row["sentences"], row["answers"], row["question"]
	row["question_emb"] = sonar.encode([question])[0].detach().cpu() #B
	row["sentences_emb"] = sonar.encode(sentences).detach().cpu() #[ [sent, sent], ... ] #B,T
	row["answers_emb"] = sonar.encode(answers).detach().cpu() #B, T
	del row["document"]
	return row


class myDataCollator:
	def __init__(self):
		self.pad_value = 0.0

	def apply_template(self, question, answer):
		return f"\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

	def __call__(self, batch):
		# sentences
		inputs_embeds = [torch.tensor(x['sentences_emb']) for x in batch] # + [x['question_emb']]
		att_s = [ torch.ones(x.size(0))  for x in inputs_embeds]
		att_s = pad_sequence(att_s, batch_first=True, padding_value=0)
		att_s = att_s.ne(0) #B,S
		inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=self.pad_value) #B,S,C
		
		# tokens
		tokens = [self.apply_template(x['question'], random.choice(x['answers'])) for x in batch]
		x = tokenizer(tokens, return_tensors="pt", padding=True, return_attention_mask=True)
		token_ids, att_t = x.input_ids, x.attention_mask
		
		# labels
		labels = []
		for t_ids in token_ids:
			#label = [-100] * S + t_ids[1:] + [-100] * (pad_len + 1)  # shift left
			label = [-100] * inputs_embeds.size(1) +   [-100 if v==0 else v for v in t_ids[1:].tolist()]   + [-100]
			labels.append(label)
		labels = torch.tensor(labels, dtype=torch.long)

		#print("inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids, "\ntype_ids", type_ids, "\nattention_mask", attention_mask, "\nlabels", labels); exit()
		out = {'inputs_embeds': inputs_embeds, 'token_ids': token_ids, 'att_s':att_s, 'att_t':att_t, 'labels':labels}
		if mode==2: #test
			out["question"] = [x["question"] for x in batch]
			out["answers"] = [x["answers"] for x in batch]
		return out


class MyModel(Qwen2ForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.embedding_dim = 1024 #sonar:1024
		self.hidden_dim = 896 #qwen:896
		self.embed_types = nn.Embedding(2, self.hidden_dim) #number of types, hidden_dim
		self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
	
	def trans(self, inputs_embeds, token_ids):
		tokens_embeds = self.model.embed_tokens(token_ids)
		if inputs_embeds is not None:
			x = self.fc1(self.ln1(inputs_embeds)) #normalizing and ignoring magnitude
			x = torch.cat([x, tokens_embeds], dim=1)
			B, S, T = inputs_embeds.size(0), inputs_embeds.size(1), token_ids.size(1)
			type_ids = torch.cat([
			    torch.zeros((B, S), dtype=torch.long),  # sentences
			    torch.ones((B, T), dtype=torch.long),   # tokens
			], dim=1).to(device)			
		else:
			x = tokens_embeds
			type_ids = torch.ones((tokens_embeds.size(0), tokens_embeds.size(1)), dtype=torch.long)

		x = x + self.embed_types(type_ids)
		return x

	def forward(self, inputs_embeds=None, token_ids=None, att_s=None, att_t=None, labels=None, **kwargs):		
		#print("\n\nforward. inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids, "\natt_s", att_s, "\natt_t", att_t)		
		attention_mask = torch.cat([att_s, att_t], dim=1)   #B,S+T,C
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds, token_ids), attention_mask=attention_mask, labels=labels) #, **kwargs
		return outputs

	def generate(self, inputs_embeds=None, token_ids=None, att_s=None, max_new_tokens=32):
		token_ids, att_t = token_ids[:, :3], torch.tensor( [[True, True, True]], device=device )
		attention_mask = torch.cat([att_s, att_t], dim=1).to(device)   #B,S+T,C		
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds, token_ids), attention_mask=attention_mask, use_cache=True)
		past_key_values = outputs.past_key_values
		logits = outputs.logits[:, -1, :]
		cur_token = torch.argmax(logits, dim=-1, keepdim=True)
		return (token_ids, cur_token)

		#attention_mask = att_t
		#B, T = token_ids.size(0), token_ids.size(1)
		#type_ids = torch.ones((B, T), dtype=torch.long, device=device)
		for i in range(max_new_tokens):
			outputs = self.forward(
				token_ids=cur_token,
				att_t=att_t,
				past_key_values=past_key_values,
				use_cache=True
			)
			logits = outputs.logits[:, -1, :]
			past_key_values = outputs.past_key_values
			next_token = torch.argmax(logits, dim=-1, keepdim=True)
			generated.append(next_token)
			cur_token = next_token
			if next_token.item() == tokenizer.eos_token_id: #151643
				break

		gen_ids = torch.cat(generated, dim=-1)
		return gen_ids



class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				token_ids, gen_ids = self.model.generate(inputs_embeds=inputs['inputs_embeds'], token_ids=inputs['token_ids'], att_s=inputs['att_s'])
				tokens = tokenizer.batch_decode(token_ids)
				generated = tokenizer.batch_decode(gen_ids)
				print(tokens, "-----", generated, "ans:", inputs['answers'])
				#compute_metrics( (preds, inputs['answers'], inputs['question']) )
		return {"accuracy": 1.0}


def compute_metrics(p):
	pred, answers, question = p
	pred = sonar.decode(pred)
	for p, ans, ques in zip(pred, answers, question):
		print(p, " -- q", ques, "ans:", ans, "\n#==================\n")
	return {"eval_accuracy": 1.0}


#==============================================================================================
if __name__ == "__main__":
	device = torch.device("cuda")
	mode = 1 #1-train,2-test
	model_id = "Qwen/Qwen2-0.5B-Instruct" #Qwen/Qwen2-0.5B | Qwen/Qwen2-0.5B-Instruct | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | "meta-llama/Llama-3.2-1B-Instruct" | "meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	
	# Dataset
	if False:
		sonar = Sonar(1)
		nlp = NLP()
		dataset = load_dataset("deepmind/narrativeqa", split="train[:20000]") #"mwpt5/MAWPS" | "ChilleD/SVAMP"
		dataset = dataset.map(preprocess, batched=True, batch_size=8, num_proc=1, load_from_cache_file=False, keep_in_memory=True)
		print("preprocess finished")
		dataset = dataset.map(postprocess)
		dataset.save_to_disk("./temp/narrativeqa_??")
		sonar.delete(); exit()
	else:
		dataset = load_from_disk("./temp/narrativeqa_train_20k")
	
	dataset = dataset.train_test_split(test_size=0.01, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if True: #first training
		model = MyModel.from_pretrained(model_id) #, torch_dtype=torch.float16
		#model.config.num_hidden_layers=12
		#model.model.layers = get_averaged_layers(model.model.layers, 12) #model.model.layers[:4]
		# Freeze all layers >= 4
		for i, layer in enumerate(model.model.layers):
		    if i >= 4:
		        for param in layer.parameters():
		            param.requires_grad = False		
		#print(model, model.config); exit()
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-")
	
	# Start training
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=100,
		per_device_train_batch_size=8,
		gradient_accumulation_steps=1,
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=5000,
		save_total_limit=5,
		eval_strategy="steps",
		eval_steps=5000,
		per_device_eval_batch_size=1,
		#metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.01, #instead of L2(pred).mean ?
		warmup_steps=10000,
		#fp16=True,
		#logging_dir="./logs/",
		#report_to="tensorboard",
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
		trainer.train() #"./model_temp/checkpoint-150"
	else:
		print(trainer.predict(test_dataset))

