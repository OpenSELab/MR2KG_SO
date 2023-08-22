# MR2KG_SO
This repo provides the code for reproducing the experiments in MR2-KG: A multi-relation multi-rationale knowledge graph for modeling software engineering knowledge on Stack Overflow. Specially, we propose a novel classification model for edge types method (i.e.,  duplicate, concatenation, containment, pre-knowledge, post-knowledge, working example, and other), tool for best answer generator.

### Dataset

We build our manually labelled knowledge coupling and complete datase to construct and verify the effectiveness of our classification model. You can get the data through the following link：https://github.com/glnmzx888/MR2KG_SO/blob/main/data/Processed_edage_rationale.csv

### Dependency

- pip install torch
- pip install transformers
- pip install sklearn 
- pip install nltk

### Relevant Pretrained-Models

Our classification model mainly relies on the following pre-trained models as the backbone to obtain the embedding representation of NL-NL pairs and obtain feature vectors to complete edage type classification.
- NL-NL Encoder: [RoBERTa-large](https://huggingface.co/roberta-large)

### Start

First, you shold download the dataset from our [link](https://github.com/glnmzx888/MR2KG_SO/blob/main/data/Processed_edage_rationale.csv). 

You can reproduce the results of our classification model by running the [file](https://github.com/glnmzx888/MR2KG_SO/blob/main/code/model/run_sun.sh). 

You can reproduce the best answer generaotr by running the [file](https://github.com/glnmzx888/MR2KG_SO/blob/main/demo/BestAnswer.py). 

Meanwhile, we provide the demo for our best answer generator in the [file](https://github.com/glnmzx888/MR2KG_SO/blob/main/demo/SO_best_answer_generator.mp4)

# Reference
If you use this code or BTLink, please consider citing us.
<pre><code></code></pre>
