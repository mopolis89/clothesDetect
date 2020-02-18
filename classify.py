# Использование
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# импортируем необходимые пакеты
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# создаём парсер аргументов и передаём их
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# загружаем изображение 
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
 
# обрабатываем изображение для классификации
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# загружаем обученную нейросеть и бинаризатор 
# с несколькими метками
print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# классифицируем входное изображение и находим
# индексы наиболее вероятных классов
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# цикл по индексам меток классов
for (i, j) in enumerate(idxs):
	# рисуем метку на изображении
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# показываем вероятность для каждой метки
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))

# показываем выходное изображение
cv2.imshow("Output", output)
cv2.waitKey(0)