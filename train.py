def trainModels(self):

  ##### Model Structure ####
  model = Sequential()
  model.add(CuDNNLSTM(64, input_shape = (self.train_x.shape[1:]), return_sequences=True))
  model.add(Dropout(0.1))
  model.add(BatchNormalization())

  model.add(CuDNNLSTM(128, input_shape = (self.train_x.shape[1:]), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(CuDNNLSTM(128, input_shape = (self.train_x.shape[1:]), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(CuDNNLSTM(64, input_shape = (self.train_x.shape[1:])))
  model.add(Dropout(0.1))
  model.add(BatchNormalization())

  model.add(Dense(32, activation="relu"))
  model.add(Dropout(0.2))

  model.add(Dense(2, activation="softmax"))

  opt = tf.keras.optimizers.Adam(lr=self.hyperparams['LEARNING_RATE'], decay=1e-6)
  ###########################
  model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

  ####### filepath #########
  filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
  checkpoint = ModelCheckpoint("training/models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
  ##########################

  history = model.fit(
    self.train_x, self.train_y,
    batch_size=self.hyperparams['BATCH_SIZE'],
    epochs=self.hyperparams['EPOCHS'],
    validation_data=(self.validation_x, self.validation_y),
    callbacks=[checkpoint])
