import matplotlib
matplotlib.use("Agg")
# подключаем необходимые пакеты
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# создаём парсер аргументов и передаём их
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# инициализируем число эпох, скорость обучения,
# размер пакета и размерность изображения
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# берём пути к изображениям и рандомно перемешиваем
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# инициализируем данные и метки
data = []
labels = []

# цикл по изображениям
for imagePath in imagePaths:
	# загружаем изображение, обрабатываем и добавляем в список
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# извлекаем метку класса из пути к изображению и обновляем
	# список меток
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# бинаризуем метки с помощью многозначного 
# бинаризатора scikit-learn
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# цикл по всем меткам
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# разбиваем данные на обучающую и тестовую выборки, используя 80%
# данных для обучения и оставшиеся 20% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# создаём генератор для добавления изображений
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# инициализируем модель с активацией sigmoid
# для многозначной классификации
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# инициализируем оптимизатор
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# компилируем модель, используя двоичную кросс-энтропию
# вместо категориальной. Это может показаться нелогичным
# для многозначной классификации, но имейте в виду, что цель --
# обрабатывать каждую выходную метку как независимое
# распределение Бернулли
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# обучаем нейросеть
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# сохраняем модель на диск
print("[INFO] serializing network...")
model.save(args["model"])

# сохраняем бинаризатор меток на диск
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# строим график потерь и точности
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])