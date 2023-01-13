## WSDM Cup 2023 -- THUIR
This codebase contains source-code that we use to participate in the [WSDM Cup 2023](https://aistudio.baidu.com/aistudio/competition/detail/536/0/leaderboard).  

## Features
Final features that we use include:

| **Feature ID** |   **Feature Name** |  **Feature Description** |   
| :--: | :-- | :-- |    
| 1     |  cross_encoder | Fine-tune the pre-trained transformer model for 200 epochs with annotation data using BCE loss |  
| 2     |  bm25 | BM25 score of title+content using [Pyserini](https://github.com/castorini/pyserini) (k1=1.6, b=0.87, tuned on the fine-tune data) |    
| 3 |  query_length | Length of the query  |  
| 4   |  title_length |  Length of the title  |  
| 5  |  content_length |  Length of the content  |   
| 6  |  query_freq |  Frequency bucket of the query   |   
| 7  |  ql |  Query likelihood score of title+content |   
| 8  |  prox-1 |  Averaged proximity score of query terms in title+content |  
| 9  |  prox-2 |  Averaged position of query terms appearing in title+content |  
| 10  |  prox-3 |  Number of query term pairs appearing in title+content within a distance of 5 |  
| 11  |  prox-4 |  Number of query term pairs appearing in title+content within a distance of 10 |  
| 12  |  prox-1-nonstop | PROX-1 score of title+content after being filtered stopwords |  
| 13  |  prox-2-nonstop |  PROX-2 score of title+content after being filtered stopwords |  
| 14  |  prox-3-nonstop |  PROX-3 score of title+content after being filtered stopwords |  
| 15  |  prox-4-nonstop |  PROX-4 score of title+content after being filtered stopwords |  
| 16  |  tf-idf |  TF-IDF score of title+content w.r.t. the query |  
| 17  |  tf |  TF score of title+content w.r.t. the query |  
| 18  |  idf |  IDF score of title+content |  
| 19  |  bm25_title |  BM25 score of title using Pyserini (k1=1.6, b=0.87) |  
| 20  |  bm25_content |  BM25 score of content using Pyserini (k1=1.6, b=0.87) |  


## Results
For Task 2: Pretraining for Web Search, we used all the aforementioned features except 14 and achieved ```DCG=10.04097``` on the leaderboard.  
As for Task 1: Unbiased Learning to Rank, we used feature 2-13 & 15-19 and finally achieved ```DCG=9.91182``` on the leaderboard.  

## Download  
You can download the best checkpoint we have trained through the following entries:  
Best checkpoint with pre-training (ctr+mlm loss): [save_steps27000_6.31586.model](https://cloud.tsinghua.edu.cn/f/310db76c238f42edbdef/?dl=1).  
Best checkpoint with fine-tuning (human label bce loss): [save_steps143000_10.08166.model](https://cloud.tsinghua.edu.cn/f/004aca88e7ba4c62b539/?dl=1).

## More
More details of our experiments will come soon in our competition papers. Please stay tuned.
