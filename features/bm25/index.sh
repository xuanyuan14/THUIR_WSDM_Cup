nohup python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/lht/wsdm_cup/utils/test_bm25_nonstop \
  --index /home/lht/wsdm_cup/utils/test_bm25_nonstop/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 128 \
  --storePositions --storeDocvectors --storeRaw > log/index-2.log 2>&1 &