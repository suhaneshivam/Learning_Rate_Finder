from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tempfile

class LearningRateFinder:
    def __init__(self ,model ,stopFactor= 4 ,beta = 0.98):
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta


        self.lrs = []
        self.losses = []

        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self ,data):
        iterClasses = ["NumpyArrayIterator" ,"DirectoryIterator" ,"DataFrameIterator" ,"Iterator" ,"Sequence"]

        return (data.__class__.__name__ in iterClasses)

    def on_batch_end(self ,batch ,logs):
        lr = K.get_value(K.model.optimizer.learning_rate)
        self.lrs.append(lr)

        l = logs["loss"]
        self.batchNum +=1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - self.beta ** self.batchNum)
        self.losses.append(smooth)

        stopLoss = self. stopFactor * self.bestLoss

        if self.batchNum > 1 and smooth > stopLoss:
            self.model.stop_Training = True
            return

        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        lr *= self.lrMult
        k.set_value(self.model.optimizer.learning_rate)

    def find(self ,trainData ,startLR ,endLR ,epoch = None ,stepPerEpoch = None ,batchSize = 32 ,sampleSize = 2048 ,verbose = 1):
        self.reset()

        # determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)

        if useGen and stepPerEpoch is None:
            msg = "Using generator without supplying the stepPerEpoch"
            raise Exception(msg)

        elif not useGen:

            numSample = len(trainData[0])
            stepsPerEpoch = np.ceil(numSample / float(batchSize))

        if epochs is None:
            #samplesize essentially represent the no of batches we want to train our network for.
            epochs = int(np.ceil(sampleSize / float(stepPerEpoch)))

        numBatchUpdates = epochs * stepsPerEpoch

        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)


        #mkstemp() returns a tuple containing an OS-level handle to an open file
        #(as would be returned by os.open()) and the absolute pathname of that file, in that order.
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
		# later), and then set the *starting* learning rate
        originalLr = K.get_value(self.model.optimizer.learning_rate)
        K.set_value(self.model.optimizer.learning_rate ,startLR)

        callback = LambdaCallback(on_epoch_end=lambda batch ,logs :self.on_batch_end(batch ,logs))

        if useGen:
            self.model.fit(x = trainData ,steps_per_epoch = stepPerEpoch ,epochs = epochs ,verbose = verbose ,callbacks = [callback])

        else:
            self.model.fit(x = trainData[0] ,y = trainData[1] ,batch_size = batchSize ,epochs = epochs ,callbacks = [callback] ,verbose = verbose)

        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.learning_rate ,originalLr)

    def plot_loss(self ,skipBegin = 10 ,skipEnd = 1 ,title = ""):
        lrs = self.lrs[skipBegin : -skipEnd]
        losses = self.losses[skipBegin : -skipEnd]

        plt.plot(lrs ,losses)
        plt.xlabel("Learning Rates")
        plt.ylabel("Loss")

        if title != "":
            plt.title(title)
