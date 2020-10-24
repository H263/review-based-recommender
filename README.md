# review-based-recommender
The repository contains several Pytorch model implementations for review-based recommendation. The implemented models are as followed,
  - DeepCoNN: [Joint Deep Modeling of Users and Items Using Reviews for Recommendation](https://arxiv.org/abs/1701.04783)
  - D-ATT: [Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction](https://dl.acm.org/doi/10.1145/3109859.3109890)
  - NARRE: [Neural Attentional Rating Regression with Review-level Explanations](https://dl.acm.org/doi/10.1145/3178876.3186070)
  - AHN: [Asymmetrical Hierarchical Networks with Attentive Interactions for Interpretable Review-Based Recommendation](https://arxiv.org/abs/2001.04346)
  - SimpleSiamese [Unpublished but strong baseline]()
  
 # Data preparation
  - To run models NARRE, SimpleSiamese:
  
    ```python preprocess/divide_and_create_example_word.py```
  - To run models AHN:
  
    ```python preprocess/divide_and_create_example_sent.py```
  - To run models D-ATT, DeepCoNN:
  
    ```python preprocess/divide_and_create_example_doc.py```
 
 # Run Model
  ```python/trainer/train_model.py```
  where model = "ahn, deepconn_pp, dual_att, narre, simple_siamese"
