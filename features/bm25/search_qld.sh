nohup python -m pyserini.search.lucene \
  --index /home/lht/wsdm_cup/utils/test_bm25_bigram/index \
  --topics /home/lht/wsdm_cup/utils/test_bm25_bigram/query.tsv \
  --output /home/lht/wsdm_cup/utils/test_bm25_bigram/output_qld.tsv \
  --qld \
  --hits 5000 \
  --threads 36 \
  --batch-size 256 > log/qld-2.log 2>&1 &
 