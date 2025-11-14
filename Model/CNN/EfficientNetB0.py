class EfficientNetModel:
  
    def __init__(self, num_classes, lr=1e-4):
        base = EfficientNetB0(weights="imagenet", include_top=False)

        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)  
        output = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=base.input, outputs=output)
        self.model.compile(
            optimizer=Adam(lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, train_loader, valid_loader, epochs=10):
        return self.model.fit(train_loader, validation_data=valid_loader, epochs=epochs)

    def evaluate(self, test_loader):
        return self.model.evaluate(test_loader)

    def predict(self, test_loader):
        return self.model.predict(test_loader)

    def summary(self):
        self.model.summary()
