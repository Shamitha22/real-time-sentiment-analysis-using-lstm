# real-time-sentiment-analysis-using-lstmOverview
This project focuses on building a sentiment analysis system using deep learning, specifically LSTM (Long Short-Term Memory) networks, to classify tweets as positive or negative.
The goal was to work with real-world, noisy text data (Twitter) and build an end-to-end NLP pipeline—from raw data cleaning to model training and evaluation—using TensorFlow and Keras.
This project helped me understand how sequential models like LSTMs capture context in text better than traditional machine learning approaches.
What This Project Does
Takes raw Twitter data and cleans it step by step
Converts text into numerical sequences that a neural network can understand
Trains an LSTM-based deep learning model
Predicts whether a tweet expresses positive or negative sentiment
Evaluates the model using standard classification metrics
Dataset Used
Sentiment140 Twitter Dataset
Originally contains 1.6 million tweets
For this project, a balanced subset of 30,000 tweets was used:
15,000 positive tweets
15,000 negative tweets
Sentiment Labels
0 → Negative
4 → Positive
Only the sentiment label and tweet text were used. All unnecessary metadata (IDs, dates, users, etc.) was removed.
Data Cleaning & Preprocessing
Twitter data is messy, so a lot of effort went into cleaning it properly. The following steps were applied:
Removed Twitter handles (@username)
Removed URLs and links
Removed punctuation, numbers, and special characters
Converted text to lowercase
Removed stopwords using NLTK
Tokenized text into words
Applied stemming using Porter Stemmer
Removed very short and meaningless words
Reconstructed clean text for modeling
These steps significantly reduced noise and improved model performance.
Model Architecture
The sentiment classifier is built using TensorFlow / Keras with the following structure:
Embedding layer (vocabulary size: 10,000, embedding dimension: 256)
Dropout layer to prevent overfitting
Two stacked LSTM layers (256 units each)
Fully connected output layer with softmax activation
Two output classes: Positive and Negative
Optimizer: Adam
Loss Function: Categorical Cross-Entropy
This architecture allows the model to learn both short-term and long-term dependencies in tweet text.
Training Details
Train–test split: 80% training, 20% testing
Batch size: 32
Epochs: 8
Framework: TensorFlow 2.17.1
Environment: Google Colab
Model Performance
After training, the model achieved an overall accuracy of approximately 69% on the test data.
Classification Results
Precision: ~0.69
Recall: ~0.69
F1-score: ~0.69
The results are reasonable considering the informal language, abbreviations, and ambiguity commonly found in tweets.
Visualizations
To better understand the data and model behavior, the project includes:
Sentiment distribution plots
Tweet length distribution
Word clouds for positive tweets
Word clouds for negative tweets
Visual representation of the LSTM model architecture
Key Takeaways
LSTMs are effective for sentiment analysis because they capture word order and context
Proper text preprocessing has a major impact on model performance
Deep learning models handle social media text better than traditional bag-of-words approaches
Balanced datasets help avoid biased predictions
Possible Improvements
Add real-time Twitter API streaming
Extend to multi-class sentiment (positive, negative, neutral)
Experiment with Bidirectional LSTM or GRU
Compare results with transformer-based models like BERT
Deploy the model using Flask or FastAPI
Technologies Used
Python
TensorFlow / Keras
NLTK
Scikit-learn
Pandas, NumPy
Matplotlib
Google Colab
Author
Shamitha Reddy Cheedu
University of North Texas
Department of Information Science
