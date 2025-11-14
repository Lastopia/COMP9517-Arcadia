import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#  Inception搭建
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
                         
    # Branch1: 1x1 卷积
    branch1 = Conv2D(filters_1x1, (1,1), padding='same', activation='relu')(x)

    # Branch2: 1x1 -> 3x3 
    branch2 = Conv2D(filters_3x3_reduce, (1,1), padding='same', activation='relu')(x)
    branch2 = Conv2D(filters_3x3, (3,3), padding='same', activation='relu')(branch2)

    # Branch3: 1x1 -> 5x5
    branch3 = Conv2D(filters_5x5_reduce, (1,1), padding='same', activation='relu')(x)
    branch3 = Conv2D(filters_5x5, (5,5), padding='same', activation='relu')(branch3)

    # Branch4: pooling -> 1x1卷积
    branch4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    branch4 = Conv2D(filters_pool_proj, (1,1), padding='same', activation='relu')(branch4)

    # 合并所有分支
    output = Concatenate(axis=-1, name=name)([branch1, branch2, branch3, branch4])
    return output

#   InceptionV3封装

class CustomInceptionV3:
    
    def __init__(self, input_shape=(224,224,3), num_classes=12, lr=1e-4, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        # 输入层
        inputs = Input(shape=self.input_shape)

        # 卷积+池化层
        x = Conv2D(32, (3,3), strides=(2,2), padding='valid', activation='relu')(inputs)
        x = Conv2D(32, (3,3), padding='valid', activation='relu')(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)
        x = Conv2D(80, (1,1), padding='same', activation='relu')(x)
        x = Conv2D(192, (3,3), padding='valid', activation='relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)

        # inception模块堆叠
        x = inception_module(x, 64, 48, 64, 16, 32, 32, name='inception_3a')
        x = inception_module(x, 64, 48, 64, 16, 32, 32, name='inception_3b')
        x = inception_module(x, 64, 48, 64, 16, 32, 32, name='inception_3c')

        # 全局平均池化 + Dropout + Dense分类层
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # 构建模型
        model = Model(inputs, outputs, name='CustomInceptionV3')
        model.compile(optimizer=Adam(self.lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # 训练接口
    def train(self, train_loader, valid_loader, epochs=10):
        return self.model.fit(train_loader,
                              validation_data=valid_loader,
                              epochs=epochs)

    # 测试集评估接口
    def evaluate(self, test_loader):
        return self.model.evaluate(test_loader)

    # 预测接口
    def predict(self, test_loader):
        return self.model.predict(test_loader)

    # 模型结构展示
    def summary(self):
        self.model.summary()

#   数据加载封装
class InsectDatasetLoader:
    def __init__(self, base_path, img_size=(224,224), batch_size=32):
        self.base_path = base_path
        self.img_size = img_size
        self.batch_size = batch_size

    def load(self):
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=20,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_loader = train_datagen.flow_from_directory(
            os.path.join(self.base_path, "train"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        valid_loader = valid_datagen.flow_from_directory(
            os.path.join(self.base_path, "valid"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical"
        )

        test_loader = test_datagen.flow_from_directory(
            os.path.join(self.base_path, "test"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )

        self.class_names = list(train_loader.class_indices.keys())
        print("类别列表:", self.class_names)
        return train_loader, valid_loader, test_loader
