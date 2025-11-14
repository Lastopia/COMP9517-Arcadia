class InceptionV3Model:
    def __init__(self, num_classes, lr=1e-4):
        # InceptionV3的基础模型
        base = InceptionV3(weights="imagenet", include_top=False)

        x = base.output # 把特征图缩成向量
        x = GlobalAveragePooling2D()(x) # 对每个通道做平均，变成2048维向量
        x = Dropout(0.3)(x) #防止全连接层过拟合，随机丢弃30%神经元
        output = Dense(num_classes, activation="softmax")(x) #最终分类层Dense Softmaz层

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
