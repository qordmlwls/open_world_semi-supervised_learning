# openworld semi-supervised learning
- This project is for testing semi-supervised learning of text
- The research process can be seen here: https://joyous-snout-4cc.notion.site/Classification-Classify-Garbage-documents-using-Open-world-Semi-Supervised-Learning-4226a881f8ad46d195a9b36f0de0e0d9
## Structure
- [input]
  - you need dataset whick is multiclass case(data is not provided)
  
- [lightning_logs] 
  - save trained model

- [modules]
  - ossl_classification.py -> train moedel module
  - SNS_content_text_one_module_for_garbage_ossl.py -> predict model modult


- [test] : 
  - test_train.py -> execute train
  - test_prediction.py -> execute prediction

