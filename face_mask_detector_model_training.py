from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-d", "--dataset", required=True)
argument_parser.add_argument("-p", "--metrics", type=str, default="metrics.png")
argument_parser.add_argument("-m", "--maskModel", type=str,
                             default="FaceMaskModel.model")
params = vars(argument_parser.parse_args())
INITIAL_LEARNING_RATE = 1e-4
EPOCHS = 2
BATCH_SIZE = 32
dataset_path = list(paths.list_images(params["dataset"]))
data = []
labels = []


for image_data in dataset_path:
    label = image_data.split(os.path.sep)[-2]
    image = load_img(image_data, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

data_augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

rootModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

masterModel = rootModel.output
masterModel = AveragePooling2D(pool_size=(7, 7))(masterModel)
masterModel = Flatten(name="flatten")(masterModel)
masterModel = Dense(128, activation="relu")(masterModel)
masterModel = Dropout(0.5)(masterModel)
masterModel = Dense(2, activation="softmax")(masterModel)
model = Model(inputs=rootModel.input, outputs=masterModel)

for modelLayer in rootModel.layers:
    modelLayer.trainable = False

opt = Adam(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
modelController = model.fit(
    data_augmentation.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS)

predictionIndex = model.predict(testX, batch_size=BATCH_SIZE)
predictionIndex = np.argmax(predictionIndex, axis=1)

print(classification_report(testY.argmax(axis=1), predictionIndex,
                            target_names=lb.classes_))

model.save(params["maskModel"], save_format="h5")
print("Face mask detector model saved successfully")

plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, EPOCHS), modelController.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), modelController.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, EPOCHS), modelController.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), modelController.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Number of Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center")
plt.savefig(params["metrics"])

