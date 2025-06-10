# parts
#dataset = load_dataset("openai/gsm8k", "main") #  "Maxwell-Jia/AIME_2024"

#===
def preprocess_gsm(batch):	
	prompts, labels = [], []
	for i in range(len(batch["question"])): #Problem, Solution, Answer | question, answer
		messages = [
			{"role": "system", "content": "Given math problem, generate final answer."},
			{"role": "user", "content": batch["question"][i]}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		label = str(batch["answer"][i])
		labels.append(label)		
	return {"prompts":prompts, "labels":labels}



def preprocess_svamp(batch):	
	prompts, labels = [], []
	for i in range(len(batch["Question"])): #Problem, Solution, Answer | question, answer
		messages = [
			{"role": "system", "content": "Given math question, generate final answer"},
			{"role": "user", "content": batch["Body"][i] + "\nQuestion: " + batch["Question"][i]}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		label = str(batch["Answer"][i]).strip()
		labels.append(label)		
		#print(prompt, label); exit()
	return {"prompts":prompts, "labels":labels}



def preprocess_mawps(batch):	
	prompts, labels = [], []
	for i in range(len(batch["Question"])):
		messages = [
			{"role": "system", "content": "Given math problem, generate answer"},
			{"role": "user", "content": batch["Question"][i] + "\nNumbers: " + batch["Numbers"][i]}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		label = str(batch["Answer"][i]).strip()
		labels.append(label)				
	return {"prompts":prompts, "labels":labels}