import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/electra-base-squad2"
# Define the pipeline
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Get predictions
def answer(question, text):
  qa_input= {
    'question' : question,
    'context' : text
  }

  answer = nlp(qa_input)['answer']
  return answer if  len(answer) != 0 else 'could not find an answer'