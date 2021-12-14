from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, pipeline
import numpy as np
from scipy.special import softmax
from tqdm import tqdm 
import torch

class Roberta:
    def __init__(self, device):
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.device = device 
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device)

    def predict_scores(self, text):
        encoded_inputs = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_inputs)
        scores = output[0][0].detach().cpu().numpy()
        return softmax(scores)

    def predict_binary(self, text):
        scores = self.predict_scores(text)
        return int(scores[2] > scores[0])

    def predict_compound(self, text):
        scores = self.predict_scores(text)
        return scores[2] - scores[0]

    def predict_scores_batch(self, texts):
        with torch.no_grad():
          encoded_inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
          output = self.model(**encoded_inputs)
          scores = output[0].detach().cpu().numpy()
          del encoded_inputs
          del output
          return softmax(scores, axis=1)

    def predict_binary_batch(self, texts):
        scores = self.predict_scores_batch(texts)
        return 1 * (scores[:, 2] > scores[:, 0])

    def predict_compound_batch(self, texts):
        scores = self.predict_scores_batch(texts)
        return (scores[:, 2] - scores[:, 0])

    def batch_and_predict(self, texts, batch_size, type):
        predictions = np.zeros((len(texts, )))
        prediction_function = self.predict_compound_batch if type=="compound" else self.predict_binary_batch
        for batch_id in tqdm(range(len(texts) // batch_size)):
            start = batch_size * batch_id
            end = start + batch_size
            batch = texts[start:end]
            # print(f"Batch {batch_id} from {start} to {end}. Size: {len(batch)}")
            predictions[start:end] = prediction_function(batch)
        return predictions