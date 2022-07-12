
# Capstone: Abuse & Hate Speech Detection in Online Content

## Problem Statement
With the advent of the internet and social media, online abuse is becoming an increasingly pressing issue. Online abuse can cause users of a platform to leave the platform, presenting an issue to internet companies. At the same time, online abuse in the form of personal attacks and cyberbullying can have serious consequences on the mental health of platform users, especially minorities who may become exposed to hate speech. Hate speech has also been shown to contribute to discriminatory attitudes and an increase in violence towards minorities.

However, moderating online abuse can be an extremely manpower-intensive exercise when performed by human moderators due to the large volume of online content being constantly generated. This also exposes moderators to traumatic content that may affect their mental health, especially minority moderators who may become exposed to online hate speech. As such, machine learning has become an attractive method of flagging abusive content online. The downside is that supervised machine learning methods require large amounts of labelled data which may not always be available. However, several large annotated datasets have been made publicly available for the purpose of training abuse detection models.

Archive Of Our Own (Ao3) is a popular non-profit open-source platform used to host self-published fanfiction. It hosts over 7 million works created by over 4 million users, and sees traffic of around 40 million visits a day. In 2019, it won a Hugo Award for Best Related Work and has generally received praise for its content tagging system. However, it has begun to receive criticism in recent years for slow response to abuse sent to users through its commenting system, particularly racist harassment of minority users. As a volunteer-driven organization, it is difficult for Ao3 to dedicate human moderators to the task, making machine learning an attractive method for moderation. However, there is a lack of datasets containing Ao3 comments annotated for abuse.

The goal of this project is to build text classification model trained on a large dataset of Wikipedia discussion comments, labelled for personal attacks under the Wikipedia detox project. The model will then be tested on a small test dataset of 330 Ao3 comments labelled for abusiveness. If accuracy on the Ao3 test set is high, this model could serve as proof of concept for automated comment moderation on Ao3, which could be used to:
1. Automatically filter abusive comments into a separate inbox
2. Flag trolls and hateful accounts to the Abuse Team

## Contents
- [Problem Statement](#Problem-Statement)
- [Data Description](#Data-Description)
- [Modelling](#Modelling)
- [Conclusions](#Conclusions-and-Recommendations)

## Data Description
My training data was drawn from a dataset of 100,000 Wikipedia discussion comments, rated for toxicity, aggression, and whether it contained a personal attack. 

My test data comprised 320 Ao3 comments which I manually compiled and labelled. The abusive comments were obtained by searching the names of four trolls known to comment on fanfiction of a popular Chinese drama. By searching their names, I found two datasets of abusive comments for one troll, several documents containing screenshots from other known trolls, and also screenshots of abusive comments posted directly to Twitter.

#### Wikipedia comments
|**Variable**|**Type**|**Definition**|
|---|---|---|
|**rev_id**|integer|Unique identifier given to each comment|
|**attack**|binary|Indicates if comment was rated by majority of workers as a personal attack|
|**recipient_attack**|float|Percentage of workers who identified the comment as an attack targeted at a recipient (e.g. "you suck")|
|**third_party_attack**|float|Percentage of workers who identified the comment as an attack targeted at a third party (e.g. "Bob sucks")|
|**quoting_attack**|float|Percentage of workers who identified the comment as an attack that was reported or quoted (e.g. "Bob said Henri sucks")|
|**other_attack**|float|Percentage of workers who identified the comment as some other kind of attack or harassment|
|**toxicity_score**|float|Average score given by workers as to how toxic a comment was on a scale of 1- 5|
|**aggression_score**|float|Average score given by workers as to how aggression a comment was on a scale of 1 - 7|
|**comment**|string|Contains the text of the Wikipedia discussion comment|
|**year**|integer|Indicates the year in which the comment was made|
|**logged_in**|binary|Indicates if the user who posted the comment was logged in or anonymous|
|**comment_len**|integer|Indicates the length of the comment in number of characters|

#### Ao3 comments
|**Variable**|**Type**|**Definition**|
|---|---|---|
|**abusive**|binary|Indicates whether the comment was abusive towards the author|
|**comment**|string|Contains the text of the Ao3 comment|
|**comment_len**|integer|Indicates the length of the comment in number of characters|
|**aggressor**|string|Dataset contains abusive comments made by 4 known trolls, indicated as troll1, troll2, troll3, and troll4, with abusive comments made by unknown commenters being labeled as "unknown"|
|**logged_in**|binary|Indicates whether aggressor was logged in|

## Modelling
I trained and tested a broad range of models. My first set of models were all bag of word models, vectorized using TF-IDF, and run through PyCaret, which compared the performance of 13 different models, including models like logistic regression, Naïve Bayes, and SVM. From this set of models, the SVM model performed best with a precision score of 89.38% precision, a recall score of 80.47%, and an accuracy of 85.47% on the Wikipedia comment set after hyperparameter tuning. The tuned model achieved 58.75% accuracy on the Ao3 comment set, with a precision of 59.33% and a recall of 55.62%. This model was set as a baseline model to compare subsequent deep learning models against.

I subsequently ran a series of neural networks trained on word embeddings. I used 100-dimensional and 300-dimensional GloVE embeddings for this purpose. The first set of models were three LSTM models. Model 1A was a unidirectional LSTM model trained on 100-dimensional GloVE embeddings. In the hopes of boosting model performance, a second model, Model 1B was trained on 300-dimensional GloVE embeddings using a unidirectional LSTM layer, but the plotted learning curves showed signs of overfitting. Model 1C reverted to 100-dimensional GloVE embeddings, but utilized a bidirectional LSTM layer to boost model performance, and an L2 penalty to curb overfitting. The second set of models were two 1D CNN models. Model 2A was a 1D CNN trained on 100-dimensional GloVE embeddings, and Model 2B was a 1D CNN trained on 300-dimensional GloVE embeddings. Both models underperformed the baseline bag of words SVM model. Finally, the third and final model was a BERT model trained on 100-dimensional GloVE embeddings.

The BERT model (Model 3) outperformed all other models on the Wikipedia comments, achieving a 98.7% train accuracy and a 87.4% test accuracy. However, Model 1A performed the best on the Ao3 comments, achieving a 65.1% accuracy with a precision of 64.1% and a recall of 60.6%.

||1A|1B|1C|2A|2B|3|
|-|-|-|-|-|-|-|
|Precision (Train)|86.3%|89.7%|83.7%|79.2%|80.7%|**98.7%**|
|Recall (Train)|83.8%|84.3%|88.3%|72.7%|83.2%|**99.1%**|
|Accuracy (Train)|85.2%|87.3%|85.6%|76.8%|81.6%|**98.9%**|
|-|-|-|-|-|-|-|
|Precision (Test)|87.1%|88.5%|83.5%|79.7%|81.9%|**87.4%**|
|Recall (Test)|80.5%|80.7%|84.8%|70.3%|81.4%|**89.3%**|
|Accuracy (Test)|84.3%|85.1%|84.0%|76.2%|81.7%|**88.2%**|
|-|-|-|-|-|-|-|
|Precision (Ao3)|**65.1%**|63.3%|58.6%|51.9%|60.7%|60.4%|
|Recall (Ao3)|**60.6%**|50.6%|59.4%|52.5%|63.7%|54.4%|
|Accuracy (Ao3)|**64.1%**|60.5%|58.8%|51.9%|61.3%|59.4%|


## Conclusions and Recommendations
- We achieved a baseline accuracy of 58.8% with SVM and tf-idf vectorization
- The LSTM model was able to outperform the baseline model with an accuracy of 64.1%. However, the model is subject to a number of key weaknesses.
    - Firstly, it still doesn’t do well on positive comments containing profanity.
    - Secondly, it doesn’t do well with vocabulary it wasn’t trained on, so it can’t handle fanfiction-specific terminologies and attacks.
    - Finally, the model is quite sensitive to typos.

According to the literature, this last weakness can likely be helped by using character-level rather than word-level embeddings. However, the other two weaknesses will likely require tailored training data to improve. There are two options available. Either we can attempt to choose a better dataset to train on, which would be difficult because fanfiction-terms can be very specific, or we can collect and label Ao3 comments to create a train dataset. I have thus organized recommendations into three main types:

**1. More advanced models**
- The first involve utilization of more advanced models. The models I’ve implemented are relatively shallow models. So maybe deeper layers, or hybrid models (for example, I’ve read about model which use both LSTM and convolutional layers on character-level embeddings) might work better. Tailored pre-trained models like HateBERT may also be worth a shot.

**2. Choose better training datasets**
- The second set of recommendations involved choosing better train sets, that better matches the writing style and vocabulary used in Ao3 comments. However, this may be potentially difficult.

**3. Collect and label Ao3 comments**
- The best way would probably be to collect and label Ao3 comments for training. One of the better ways to do this would be to integrate a data collection system into the platform, for example by allowing users to flag abusive comments with some kind of button. Users will thus do the labelling for you, however this requires support and action from platform managers.
- If data is manually collected and labelled by an individual, it will be manpower-intensive, but will likely lead to large improvements in model performance.
