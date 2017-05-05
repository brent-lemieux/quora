Loss = .4222
```
def build_cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=question_len * 2))
    model.add(Conv1D(filters=32, kernel_size=12, padding='same', activation='tanh'))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='tanh'))
    model.add(MaxPooling1D(pool_size=4))
    # model.add(Conv1D(filters=32, kernel_size=12, padding='same', activation='tanh'))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```
