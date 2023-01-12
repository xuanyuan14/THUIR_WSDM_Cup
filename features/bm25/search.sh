nohup python -m pyserini.search.lucene \
  --index /home/lht/wsdm_cup/utils/test_bm25_nonstop/index \
  --topics /home/lht/wsdm_cup/utils/test_bm25_nonstop/query.tsv \
  --output /home/lht/wsdm_cup/utils/test_bm25_nonstop/output_bm25.tsv \
  --bm25 \
  --k1 1.6 \
  --b 0.86 \
  --hits 5000 \
  --threads 36 \
  --batch-size 256 > log/bm25.log 2>&1 &
 