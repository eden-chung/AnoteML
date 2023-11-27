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

RAG -> Fine Tuning -> 

## Evaluation and Results

## Next Steps

1. Our model was trained using a training set that we manually created. As students, we obviously do not have the best understanding of the answers that industry analysts may be seeking, and since our model is fine-tuned on this training data the accuracy could be further improved by having better, and more, training data.
2. The model uses ChromaDB and upon asking a question, creates a new database, then deletes it. This process is time-consuming and can take between 20-60 seconds for a single prompt, which is not ideal. The model could be improved by optimizing this performance.
3. The model is specific to 10-K documents right now, but to scale up, we could train the model to answer questions on other financial documents too. 


