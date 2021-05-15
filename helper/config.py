import os

CLASSES = ["top" ,"trouser" ,"pullover" ,"dress" ,"coat" ,"sandel" ,"shirt" ,"sneaker" ,"bag" ,"ankel boot"]

MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48

LR_FIND_PLOT_PATH = os.path.sep.join(["output" ,"lrfinf_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output" ,"train_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output" ,"clr_plot.png"])
