"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (as in Lab 9).

The Hugging Face code lives only here. Anyone who loads the logged model through
MLflow gets a DataFrame back and never has to touch transformers directly.
"""
import mlflow.pyfunc
import pandas as pd


def normalise_label(raw_label, id2label=None):
    """Map whatever the model outputs to 'positive' / 'negative' / 'neutral'."""
    lab = str(raw_label).lower()
    if lab.startswith("label_"):
        # e.g. textattack models: LABEL_0 = negative, LABEL_1 = positive
        idx = int(lab.split("_")[1])
        if id2label and idx in id2label and not str(id2label[idx]).lower().startswith("label_"):
            lab = str(id2label[idx]).lower()
        else:
            lab = "positive" if idx == 1 else "negative"
    if lab.startswith("pos"):
        return "positive"
    if lab.startswith("neg"):
        return "negative"
    return "neutral"


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a 'text' column. Output: text, prediction, score."""

    def __init__(self, hf_model_id, batch_size=16, max_length=128):
        self.hf_model_id = hf_model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.pipe = None

    def load_context(self, context):
        from transformers import pipeline  # imported only inside the wrapper

        self.pipe = pipeline("sentiment-analysis", model=self.hf_model_id, device=-1)

    def predict(self, context, model_input, params=None):
        if self.pipe is None:
            self.load_context(context)
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        outputs = self.pipe(texts, batch_size=self.batch_size,
                            truncation=True, max_length=self.max_length)
        id2label = getattr(self.pipe.model.config, "id2label", None)
        return pd.DataFrame({
            "text": texts,
            "prediction": [normalise_label(o["label"], id2label) for o in outputs],
            "score": [round(float(o["score"]), 4) for o in outputs],
        })
