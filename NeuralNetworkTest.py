import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_test_data(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)
    return x_train, x_test, y_train, y_test

def standardize_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    return x_train_std, x_test_std

def to_onehot(arr):
    onehot = []

    for label in arr:
        if label == 0:
            _1hot = [1, 0]
        elif label == 1:
            _1hot = [0, 1]
        
        onehot.append(_1hot)

    # converting from python list to numpy array
    return np.array(onehot)

def create_model(n_features):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(units = 50, input_shape = (n_features,), activation = 'relu'))

    model.add(tf.keras.layers.Dense(units = 20, input_shape=(n_features,), activation='relu'))

    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, x_train, y_train_1hot):
  return model.fit(x_train, y_train_1hot, epochs=50)

def plot(hist):
  _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

  ax[0].plot(hist.history['loss'])
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Loss')
  ax[0].set_title('Loss Curve')

  ax[1].plot(hist.history['accuracy'])
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Accuracy')
  ax[1].set_title('Accuracy Curve')

  plt.show()

def auto_eval(model, x_test, y_test_1hot): 
  loss, accuracy = model.evaluate(x=x_test, y=y_test_1hot)

  print('loss = ', loss)
  print('accuracy = ', accuracy)    

def main():
    df = pd.read_csv('framingham.csv')
    df_cleaned = df.drop(columns=['education'])
    df_cleaned_filled = df_cleaned.fillna(df_cleaned.mean())
    print(df_cleaned)

    # ".values" to change from panda's dataframe 
    # data structure to numpy's array
    data = df_cleaned_filled.values

    # These two columns are not our features:
    #   - our label (at last column)
    #   - running number (at first column)
    features = data[:,1:-1]
    
    # The last column, that specifies Genuine or Counterfeit,
    # contains our labels 
    labels = data[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size = 0.1, stratify=labels)

    # standardize our data
    x_train_std, x_test_std = standardize_data(x_train, x_test)

    # perform one-hot encoding
    y_train_1hot = to_onehot(y_train)
    y_test_1hot = to_onehot(y_test)

    # create and train our model       
    model = create_model(
        x_train_std.shape[1]    # no. of features used for training 
    ) 
    hist = train_model(model, x_train_std, y_train_1hot)

    # loss and accuracy plots
    plot(hist)

    # tensorflow will do auto-evaluation of model against test set
    auto_eval(model, x_test_std, y_test_1hot)
    


if __name__ == "__main__":
  main()