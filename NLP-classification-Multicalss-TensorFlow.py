import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

 
df = pd.read_csv("data.csv")  

 
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

 
max_words = 10000 # the dictionary
max_len = 100 # number of tokens in each input. if long text, will be truncated

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

#  multiple labels  if charactor, label encoder first: eg: ['positive', 'neutral', 'negative'] 
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)

# model construct
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.5),
    Dense(3, activation='softmax')  #softmax(multiclass)/sigmoid(binary)
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #crossentropy if binary

# train
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# eval
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"Test accuracy: {acc:.4f}")

# pred
y_pred_proba = model.predict(X_test_pad)  # shape: (num_samples, 3) 
y_pred = y_pred_proba.argmax(axis=1)
