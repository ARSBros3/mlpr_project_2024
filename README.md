# mlpr_project_2024
Our MLPR project on detecting the degree of factuality of news. Focusing on text.

## Our data
We got our data from Kaggle, at https://www.kaggle.com/datasets/anmolkumar/fake-news-content-detection/

The data's format is as follows-

### train.csv
There are 10240 instances, each with features-
1. **Labels-** The target variable we are trying to predict. This is a variable to indicate the degree of factuality of the text. The possible values are-
     - 0: Barely True
     - 1: False
     - 2: Half-True
     - 3: Mostly True
     - 4: Not Known
     - 5: True
2. **Text-** The raw text.
3. **Text_tag-** Various tags given to the text, based on the type of content in the raw text.

### test.csv
There are 1267 instances here, with 2 features- the raw text, and the tags for the text.
