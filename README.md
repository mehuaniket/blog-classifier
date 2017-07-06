# Blog-Classifier

## Introduction 

- This is small tool to classify blog in categories using random-forest.
- Training set is in blogs.csv in following format
- To predict category for blog you've to use `use_forest_prediction.py`. May be you can change the way 
  data is provided to the function to integrate with backend. 
 
```
   |----------------------|
   |   blog  |  category  |
   |----------------------|

```

## Installation

- Installing dependency from requirements.txt using following command
  
```bash
   pip install -r requirements.txt
```

- Training from `blogs.csv` (save model in forest.pickle and vocab.pickle)
- run following command in folder.

```bash
   python bag-of-words.py
```
- comment out 15th line in `bag-of-words.py` after downloading stopwords from popup.

## Make Prediction

- In order to make prediction,Run `user_forest_prediction.py'

```python
   python user_forest_predition.py
```

- Basically,Collection blog is collected from following urls with scrapy framework.So Random blog will
  not provide good results.
  
  - tsa - https://www.dhs.gov/archive/news-releases/blog
  - air-traffic - http://www.atc-network.com/atc-news
  
- Reference: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words 