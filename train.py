import matplotlib
matplotlib.use("Agg")

from helper.clr_callback import CyclicLR
from helper.learningratefinder import LearningRateFinder
from helper.minigooglenet import MiniGoogLeNet
from helper import config
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import sys
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

((trainX ,trainY) ,(testX ,testY)) = cifar10.load_data()
trainX = np.array([cv2.resize(x ,(32 ,32)) for x in trainX])
testX = np.array([cv2.resize(x ,(32 ,32)) for x in testX])

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

trainX = trainX.reshape((trainX.shape[0] ,32 ,32 ,1))
testX = test.reshape((testX.shape[0] ,32 ,32 ,1))

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1 ,height_shift_range=0.1 ,horizontal_flip=True ,fill_mode="nearest")

model = MiniGoogLeNet.build(32 ,32 ,1 ,classes = len(config.CLASSES))
opt = SGD(lr= config.MIN_LR ,momentum = 0.9)
model.compile(optimizer=opt ,loss = "categorical_crossentropy" ,metrics="accuracy")

if args["lr_find"]:
	lrf = LearningRateFinder(model = model)
	lrf.find(aug.flow(trainX ,trainY ,batch_size = config.BATCH_SIZE) ,
						startLR = 1e-1 ,endLR=1e+1 ,stepsPerEpoch = len(trainX)//config.BATCH_SIZE ,
						batchSize=config.BATCH_SIZE)
	lrf.plot_loss(title = "LRs_vs_Loss")
	plt.savefig(config.LR_FIND_PLOT_PATH)

	print("[INFO] learning rate finder complete")
	print("[INFO] examine plot and adjust learning rates before training")
	sys.exit(0)

stepSize = config.STEP_SIZE * (len(trainX) // config.BATCH_SIZE)
clr = CyclicLR(config.MIN_LR ,config.MAX_LR ,stepSize ,mode = config.CLR_METHOD)
H = model.fit(x = aug.flow(trainX ,trainY ,batch_size =config.BATCH_SIZE),validation_data = testX ,testY ,
						steps_per_epoch = len(traiX)//config.BATCH_SIZE ,callbacks = [clr] ,verbose = True)

predictions = model.predict(testX ,batch_size=config.BATCH_SIZE)
print(classification_report(trainY.argmax(axis = 1) ,predictions.argmax(axis = 1) ,target_names=config.CLASSES))

N = np.arange(0 ,config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N ,H.history["loss"] , label = "Train_loss")
plt.plot(N ,H.history["val_loss"] ,label = "Validation_loss")
plt.plot(N ,H.historty["accuracy"] ,label = "Train_accuracy")
plt.plot(N ,H.history["val_accuracy"] ,label = "validation_accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.title("Training Loss and Accuracy")
plt.legend(loc = "lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

N = np.arange(0 ,len(clr.history["lr"]))
plt.figure()
plt.plot(N ,clr.history["lr"])
plt.xlabel("Training Iterations #")
plt.ylabel("Learning Rate")
plt.title("cyclic learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
