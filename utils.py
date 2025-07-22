
class NLP:
	def __init__(self):
		import spacy
		self.nlp = spacy.load("en_core_web_sm")

	def split_sentences(self, text):
		doc = self.nlp(text)
		sentences = [sent.text.strip() for sent in doc.sents]
		return sentences


class Sonar:
	def __init__(self, mode):
		from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
		if mode==1 or mode==3:
			self.t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		if mode==2 or mode==3:
			self.vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		print("Sonar initialized")

	def delete(self):
		del self.t2vec_model #, self.vec2text_model
		torch.cuda.empty_cache()		

	def encode(self, sentences):
		#sentences = ["Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?", "N_02 / ( N_00 + N_01 + N_02 ) * 100.0"]
		embeddings = self.t2vec_model.predict(sentences, source_lang="eng_Latn")
		return embeddings

	def decode(self, embeddings):
		reconstructed = self.vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
		return reconstructed