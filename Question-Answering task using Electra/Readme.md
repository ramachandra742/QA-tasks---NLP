### Question-Answering task using Electra
We use wikipedia text as context for this task.

#### Description     

This repository implements a pipeline to answer questions using wikipedia text.      
Bellow is the pipeline:      
* Using the input query, search on google filtering the wikipedia pages.
* Read the body content of the wikipedia, preprocess text and split the corpus in paragraphs.
* Use BM25 algorithm to rank the best candidate passages, using the top K paragraphs.
* Selected paragraphs are used as input to Electra model.
* Electra model try to find the answer given the candidate paragraphs.

#### How to run?
This QA  task was done on Google Colab using flask-ngrok for serving. For more details, check [here](https://github.com/ramachandra742/QA_tasks-NLP/blob/master/Question-Answering%20task%20using%20Electra/Question_Answering_task_using_ALBERT.ipynb).

### Inference from Electra QA model      
![](https://github.com/ramachandra742/QA_tasks-NLP/blob/master/Question-Answering%20task%20using%20Electra/fig1.png)    
![](https://github.com/ramachandra742/QA_tasks-NLP/blob/master/Question-Answering%20task%20using%20Electra/fig2.png)


References:   
https://github.com/renatoviolin/Question-Answering-Albert-Electra       
https://huggingface.co/deepset/electra-base-squad2  
