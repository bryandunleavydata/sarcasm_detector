# sarcasm_detector

Title: Sarcasm Detection with Bi-directional LSTM Model Bryan Dunleavy & Tahora Husaini Goal: To research train and implement Deep Learning Model concerned with the classification of individual sarcastic comments. Process: Data Collection: Reddit dataset

Introduction: The LSTM model is a powerful deep learning technique used for sentiment analysis, which involves predicting the sentiment (positive or negative) of textual data, such as comments or reviews. Model Architecture:

Our LSTM model is designed to process text data efficiently and make accurate sentiment predictions. The model consists of multiple layers: Input Layer: Receives sequences of word indices representing preprocessed text data. Embedding Layer: Maps the word indices to dense word vectors using pre-trained word embeddings. Bidirectional LSTM Layer: Utilizes LSTM units in both forward and backward directions to capture context from both past and future words effectively.Global MaxPooling1D Layer: Extracts the most important features from the LSTM output sequence. Dense Layers with Dropout: Fully connected layers with ReLU activation and dropout regularization to prevent overfitting. Output Layer: A single neuron with sigmoid activation for binary sentiment classification.

Preprocessing: The data is preprocessed to remove noise and standardize the text data for better model performance. Contractions and misspellings are expanded and corrected, and special characters are removed. The text is tokenized, and the tokenized sequences are padded to a fixed length to ensure uniform input shape.

Embedding: We use pre-trained word embeddings from a large corpus to represent words as dense vectors. The embedding matrix is constructed with word vectors that align with the tokenized words in the dataset. By using pre-trained embeddings, the model benefits from transfer learning and generalizes better to unseen data.

Training: The model is trained using the Adam optimizer with binary cross-entropy loss, a common choice for binary classification problems. We fix the random seed for reproducibility and split the data into training and validation sets for model evaluation. The training process involves iterating over epochs, with each epoch updating the model's weights to minimize the loss on the training data. The validation set is used to monitor the model's performance and avoid overfitting.

Results: The LSTM model achieves promising results in sentiment analysis, with high accuracy on both the training (74,66%) and validation (71,35%). The loss and accuracy curves during training indicate that the model is learning effectively and not overfitting the data. By leveraging pre-trained embeddings and the LSTM's sequential memory capabilities, the model captures contextual information and patterns in the text, leading to accurate sentiment predictions.

Conclusion: The LSTM model demonstrates its effectiveness in sentiment analysis, making it a valuable tool for understanding and interpreting textual data. With the ability to handle varying lengths of text and learn sequential dependencies, the LSTM model is widely applicable to various NLP tasks.

Tools: Python and Google colab

Dataset: https://www.kaggle.com/datasets/danofer/sarcasm?select=train-balanced-sarcasm.csv Raddit Sarcasm dataset contains 1.3 million rows and 10 columns from which we use 2 columns (label and comments)

https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb

