# AnoteML
## About
We are a team of 5 college students working on this project with Anote, a NYC-based AI startup. We have used various ML and NLP techniques to create a chatbot to answer domain-specific questions. Specifically, we have created a chatbot that will accurately answer questions on financial documents (such as 10-K documents).


## Installation and Usage
```
git clone https://github.com/eden-chung/AnoteML/
cd AnoteML/frontend
```
Usage:
```
streamlit run Chatbot.py
```
## Architecture

Our model uses a combination of different ML techniques. It first uses Retrieval Augmented Generation (RAG), to put all the data from the 10-K into a database, known as the Knowledge Hub
. We chose to use ChromaDB, as it is hosted locally. The model will use a similarity search to find the most relevant "chunk" of the 10-K document to answer the given question.

<img src="Images/RAG.png">

The model then uses fine-tuning with a training set of 40 data points.

<img src="Images/fine_tuning_diagram.png">


## Evaluation and Results
Our final model can take as a user input either a PDF of the 10-K or the company ticker, along with the prompt to ask. 

We evaluated the model upon 3 new 10-K documents: Dropbox, Google, and Netflix and 15 questions.

Results from the baseline model: GPT-3.5 Turbo
<img src="Images/gpt_eval.png">
Overall accuracy: 58%

Results from our fine-tuned model: Fine tuned version of GPT-3.5 Turbo
<img src="Images/finetuned_eval.png">
Overall accuracy: 64%

It is clear that our fine-tuned model is still not 100% accurate. From the evaluation, the fine tuning increased the accuracy by 6% to 64%. This accuracy could be further improved had we had more time, which will be discussed below.



## Next Steps
In a limited time frame, and as full time college students, there were many limitations to our model that we recognized but were not able to address.

1. Our model was trained using a training set that we manually created. As students, we obviously do not have the best understanding of the answers that industry analysts may be seeking, and since our model is fine-tuned on this training data the accuracy could be further improved by having better, and more, training data.
2. The model uses ChromaDB and upon asking a question, creates a new database, then deletes it. This process is time-consuming and can take between 20-60 seconds for a single prompt, which is not ideal. The model could be improved by optimizing this performance.
3. Currently, the model uses OpenAI's GPT-3.5 as a base model. Ideally, the entire model could be hosted locally to reduce privacy concerns. Instead of using OpenAI's GPT-3.5, we could use a locally hosted model such as Meta's LLaMA.
4. The model is specific to 10-K documents right now, but to scale up, we could train the model to answer questions on other financial documents too. 


