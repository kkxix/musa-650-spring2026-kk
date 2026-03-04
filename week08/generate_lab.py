"""
generate_lab.py
Generates two step-by-step lab notebooks from DL_Basics1_SimpleMLP.ipynb
and DLBasics_SimpleCNN.ipynb:
  - week06_lab_student.ipynb   (questions / TODOs)
  - week06_lab_solutions.ipynb (complete solutions)
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT_STUDENT   = os.path.join(FOLDER, "week06_lab_student.ipynb")
OUT_SOLUTIONS = os.path.join(FOLDER, "week06_lab_solutions.ipynb")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def md(src):
    return new_markdown_cell(src)

def code(src):
    return new_code_cell(src)

# ---------------------------------------------------------------------------
# PART 1 – MLP
# ---------------------------------------------------------------------------

PART1_TITLE = """\
---
# Part 1 — Simple Multi-Layer Perceptron (MLP) on MNIST

In this part you will build, train, and evaluate a fully-connected neural
network to classify handwritten digits from the MNIST dataset.

**Learning goals**
- Understand how to prepare image data for a dense neural network
- Build a Sequential Keras model with `Dense` layers
- Train and evaluate a model on a real dataset
"""

# ── Step 1 ──────────────────────────────────────────────────────────────────

STEP1_MD = """\
## Step 1 — Imports

Import all libraries needed for the MLP experiment.
"""

STEP1_STUDENT = """\
from __future__ import print_function

# TODO 1a: Import keras and the Sequential model class
# TODO 1b: Import Dense and Dropout from keras.layers
# TODO 1c: Import RMSprop from tensorflow.keras.optimizers
# TODO 1d: Import numpy as np
# TODO 1e: Import np_utils from tensorflow.python.keras.utils
# TODO 1f: Import matplotlib.pyplot as plt
"""

STEP1_SOLUTION = """\
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt
"""

# ── Step 2 ──────────────────────────────────────────────────────────────────

STEP2_MD = """\
## Step 2 — Set Parameters and Load MNIST Data

MNIST contains 70 000 greyscale images of handwritten digits (28×28 pixels).
"""

STEP2_STUDENT = """\
# TODO 2a: Set batch_size = 20000
# TODO 2b: Set num_classes = 10  (digits 0–9)
# TODO 2c: Set epochs = 5

batch_size  = ___
num_classes = ___
epochs      = ___

# TODO 2d: Load the MNIST dataset into (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = ___

# TODO 2e: Print the shape of x_train
print("x_train shape:", ___)

# TODO 2f: How many training samples belong to digit '8'?
print("Number of 8s in training set:", ___)
"""

STEP2_SOLUTION = """\
batch_size  = 20000
num_classes = 10
epochs      = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train shape:", x_train.shape)          # (60000, 28, 28)
print("Number of 8s in training set:", np.sum(y_train == 8))
"""

# ── Step 3 ──────────────────────────────────────────────────────────────────

STEP3_MD = """\
## Step 3 — Visualise a Sample

Before preprocessing, inspect a raw image and its label.
"""

STEP3_STUDENT = """\
# TODO 3a: Display the image at index 3890 using plt.imshow
#          Use cmap=plt.cm.binary so the digit appears dark on white
# TODO 3b: Print the label (y_train value) at that index

plt.imshow(___, cmap=plt.cm.binary)
plt.title("Label: " + str(___))
plt.show()
"""

STEP3_SOLUTION = """\
plt.imshow(x_train[3890], cmap=plt.cm.binary)
plt.title("Label: " + str(y_train[3890]))
plt.show()
"""

# ── Step 4 ──────────────────────────────────────────────────────────────────

STEP4_MD = """\
## Step 4 — Preprocess the Data

Dense layers expect a 1-D feature vector, not a 2-D image, so we must
**flatten** each 28×28 image into a vector of length 784.
We also **normalise** pixel values from [0, 255] to [0, 1].
"""

STEP4_STUDENT = """\
# TODO 4a: Reshape x_train from (60000, 28, 28) → (60000, 784)
x_train = x_train.reshape(___, ___)

# TODO 4b: Reshape x_test  from (10000, 28, 28) → (10000, 784)
x_test  = x_test.reshape(___, ___)

# TODO 4c: Cast both arrays to float32
x_train = x_train.astype(___)
x_test  = x_test.astype(___)

# TODO 4d: Normalise to [0, 1]
x_train /= ___
x_test  /= ___

print(x_train.shape[0], "train samples")
print(x_test.shape[0],  "test samples")
"""

STEP4_SOLUTION = """\
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255

print(x_train.shape[0], "train samples")   # 60000
print(x_test.shape[0],  "test samples")    # 10000
"""

# ── Step 5 ──────────────────────────────────────────────────────────────────

STEP5_MD = """\
## Step 5 — One-Hot Encode the Labels

The output layer will have 10 neurons (one per class).
We convert integer labels (e.g. `5`) into binary vectors
(e.g. `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`).
"""

STEP5_STUDENT = """\
# TODO 5a: Print y_train[0:5] to see the integer labels before encoding
print("Before encoding:", ___)

# TODO 5b: Convert y_train to one-hot using np_utils.to_categorical
y_train = np_utils.to_categorical(___, ___)

# TODO 5c: Do the same for y_test
y_test  = np_utils.to_categorical(___, ___)

# TODO 5d: Print y_train.shape – should now be (60000, 10)
print("After encoding, y_train shape:", ___)
"""

STEP5_SOLUTION = """\
# Save originals for display (already converted above in solution flow)
# Note: in the actual solution flow, load y_train before this cell.
print("Before encoding:", y_train[:5])           # integer labels

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test,  num_classes)

print("After encoding, y_train shape:", y_train.shape)  # (60000, 10)
"""

# ── Step 6 ──────────────────────────────────────────────────────────────────

STEP6_MD = """\
## Step 6 — Build the MLP Model

We use Keras' `Sequential` API to stack layers.
Our simple MLP has:
- **One hidden layer**: 512 neurons, ReLU activation
- **One output layer**: 10 neurons (one per class), Softmax activation

> **Tip**: `model.summary()` prints a table showing every layer,
> its output shape, and the number of trainable parameters.
"""

STEP6_STUDENT = """\
model = Sequential()

# TODO 6a: Add a Dense hidden layer with 512 neurons, relu activation
#          and input_shape=(784,)
model.add(Dense(___, activation=___, input_shape=___))

# TODO 6b: Add the output Dense layer (num_classes neurons, softmax)
model.add(Dense(___, activation=___))

# TODO 6c: Print the model summary
model.summary()
"""

STEP6_SOLUTION = """\
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
"""

# ── Step 7 ──────────────────────────────────────────────────────────────────

STEP7_MD = """\
## Step 7 — Visualise the Model Architecture

`plot_model` draws a diagram of the layer graph.
"""

STEP7_STUDENT = """\
from tensorflow.keras.utils import plot_model

# TODO 7: Call plot_model with show_shapes=True and show_layer_names=True
plot_model(___, show_shapes=___, show_layer_names=___)
"""

STEP7_SOLUTION = """\
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)
"""

# ── Step 8 ──────────────────────────────────────────────────────────────────

STEP8_MD = """\
## Step 8 — Compile the Model

Before training we must choose:
- **Loss function** – `categorical_crossentropy` for multi-class classification
- **Optimizer** – `RMSprop` adapts the learning rate during training
- **Metric** – `accuracy` is easy to interpret
"""

STEP8_STUDENT = """\
# TODO 8: Compile the model
model.compile(
    loss=___,          # hint: 'categorical_crossentropy'
    optimizer=___,     # hint: RMSprop()
    metrics=___        # hint: ['accuracy']
)
"""

STEP8_SOLUTION = """\
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)
"""

# ── Step 9 ──────────────────────────────────────────────────────────────────

STEP9_MD = """\
## Step 9 — Train the Model

`model.fit` runs the training loop.
The `history` object it returns lets us plot learning curves later.
"""

STEP9_STUDENT = """\
# TODO 9: Train the model for 10 epochs
history = model.fit(
    x_train, y_train,
    batch_size=___,
    epochs=___,           # try 10
    verbose=1,
    validation_data=___   # hint: (x_test, y_test)
)
"""

STEP9_SOLUTION = """\
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)
"""

# ── Step 10 ─────────────────────────────────────────────────────────────────

STEP10_MD = """\
## Step 10 — Evaluate the Model

Run the model on the held-out test set and report the final metrics.
"""

STEP10_STUDENT = """\
# TODO 10: Evaluate the model on the test set (verbose=0)
score = model.evaluate(___, ___, verbose=0)
print('Test loss:    ', ___)
print('Test accuracy:', ___)
"""

STEP10_SOLUTION = """\
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:    ', score[0])
print('Test accuracy:', score[1])
"""

# ── Step 11 – Reflection ─────────────────────────────────────────────────────

STEP11_MD = """\
## Step 11 — Reflection

Answer the following questions in a new markdown cell:

1. How many **trainable parameters** does the MLP have?
   Where does most of the computation happen?
2. What accuracy did you achieve after 10 epochs?
   Do you think the model is overfitting or underfitting?
3. The commented-out lines in the original notebook add a second hidden
   layer and `Dropout`. Try uncommenting them. How does accuracy change?
4. Why do we need `softmax` in the output layer but `relu` in hidden layers?
"""

# ---------------------------------------------------------------------------
# PART 2 – CNN
# ---------------------------------------------------------------------------

PART2_TITLE = """\
---
# Part 2 — Simple Convolutional Neural Network (CNN) on MNIST

CNNs learn **spatial features** directly from 2-D images using convolutional
filters instead of flattening the image up-front.

**Learning goals**
- Understand how CNNs differ from MLPs in terms of input shape
- Build a CNN with `Conv2D`, `MaxPooling2D`, and `Dropout` layers
- Compare CNN performance with the MLP from Part 1
"""

# ── Step 1 ──────────────────────────────────────────────────────────────────

P2_STEP1_MD = """\
## Step 1 — Imports (CNN)

In addition to the MLP imports, we need CNN-specific layers.
"""

P2_STEP1_STUDENT = """\
from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# TODO 1: Also import Conv2D and MaxPooling2D from keras.layers
from keras.layers import ___, ___

from keras import backend as K
from tensorflow.python.keras.utils import np_utils
"""

P2_STEP1_SOLUTION = """\
from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.python.keras.utils import np_utils
"""

# ── Step 2 ──────────────────────────────────────────────────────────────────

P2_STEP2_MD = """\
## Step 2 — Set Parameters and Load Data

CNNs keep the spatial dimensions.
The input shape for MNIST is **(28, 28, 1)** (height × width × channels).

> **channels_first vs channels_last**: Some GPU backends expect the
> channel dimension first `(1, 28, 28)`. We use `K.image_data_format()`
> to handle both cases automatically.
"""

P2_STEP2_STUDENT = """\
# TODO 2a: Set batch_size = 128, num_classes = 10, epochs = 12
batch_size  = ___
num_classes = ___
epochs      = ___

# TODO 2b: Set img_rows = 28, img_cols = 28
img_rows, img_cols = ___, ___

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# TODO 2c: Reshape the data depending on K.image_data_format()
#   If 'channels_first': shape → (n_samples, 1, img_rows, img_cols)
#   If 'channels_last':  shape → (n_samples, img_rows, img_cols, 1)
if K.image_data_format() == 'channels_first':
    x_train     = x_train.reshape(x_train.shape[0], ___, img_rows, img_cols)
    x_test      = x_test.reshape(x_test.shape[0],   ___, img_rows, img_cols)
    input_shape = (___, img_rows, img_cols)
else:
    x_train     = x_train.reshape(x_train.shape[0], img_rows, img_cols, ___)
    x_test      = x_test.reshape(x_test.shape[0],   img_rows, img_cols, ___)
    input_shape = (img_rows, img_cols, ___)

# TODO 2d: Cast to float32 and normalise to [0, 1]
x_train = x_train.astype(___) / ___
x_test  = x_test.astype(___) / ___

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0],  'test samples')

# TODO 2e: One-hot encode labels
y_train = np_utils.to_categorical(___, ___)
y_test  = np_utils.to_categorical(___, ___)
"""

P2_STEP2_SOLUTION = """\
batch_size  = 128
num_classes = 10
epochs      = 12
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train     = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test      = x_test.reshape(x_test.shape[0],   1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train     = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test      = x_test.reshape(x_test.shape[0],   img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')  / 255

print('x_train shape:', x_train.shape)   # (60000, 28, 28, 1)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0],  'test samples')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test  = np_utils.to_categorical(y_test,  num_classes)
"""

# ── Step 3 ──────────────────────────────────────────────────────────────────

P2_STEP3_MD = """\
## Step 3 — Build the CNN Model

The architecture:

| Layer | Output shape | Notes |
|---|---|---|
| Conv2D(32, 3×3, relu) | (26, 26, 32) | Learns 32 local filters |
| Conv2D(64, 3×3, relu) | (24, 24, 64) | Deeper feature maps |
| MaxPooling2D(2×2) | (12, 12, 64) | Down-samples by 2 |
| Dropout(0.25) | (12, 12, 64) | Regularisation |
| Flatten | (9216,) | Converts 3-D → 1-D |
| Dense(128, relu) | (128,) | Fully-connected |
| Dropout(0.5) | (128,) | More regularisation |
| Dense(10, softmax) | (10,) | One probability per class |
"""

P2_STEP3_STUDENT = """\
model = Sequential()

# TODO 3a: First Conv2D – 32 filters, 3×3 kernel, relu, correct input_shape
model.add(Conv2D(___, kernel_size=(3, 3), activation=___, input_shape=___))

# TODO 3b: Second Conv2D – 64 filters, 3×3 kernel, relu
model.add(Conv2D(___, (___, ___), activation=___))

# TODO 3c: MaxPooling with pool_size (2, 2)
model.add(MaxPooling2D(pool_size=___))

# TODO 3d: Dropout with rate 0.25
model.add(Dropout(___))

# TODO 3e: Flatten
model.add(Flatten())

# TODO 3f: Dense hidden layer – 128 neurons, relu
model.add(Dense(___, activation=___))

# TODO 3g: Dropout with rate 0.5
model.add(Dropout(___))

# TODO 3h: Output layer – num_classes neurons, softmax
model.add(Dense(___, activation=___))

model.summary()
"""

P2_STEP3_SOLUTION = """\
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
"""

# ── Step 4 ──────────────────────────────────────────────────────────────────

P2_STEP4_MD = """\
## Step 4 — Visualise and Compile the CNN
"""

P2_STEP4_STUDENT = """\
from tensorflow.keras.utils import plot_model

# TODO 4a: Visualise the CNN model
plot_model(___, show_shapes=True, show_layer_names=True)

# TODO 4b: Compile with categorical crossentropy loss, Adadelta optimiser,
#          and accuracy metric
model.compile(
    loss=___,
    optimizer=___,    # hint: tf.keras.optimizers.Adadelta()
    metrics=___
)
"""

P2_STEP4_SOLUTION = """\
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=['accuracy']
)
"""

# ── Step 5 ──────────────────────────────────────────────────────────────────

P2_STEP5_MD = """\
## Step 5 — Train the CNN

> **Note**: Training a CNN takes longer than an MLP.
> We use only **3 epochs** here for speed — try more when you have time!
"""

P2_STEP5_STUDENT = """\
# TODO 5: Train for 3 epochs and store the history
epochs = 3

history = model.fit(
    x_train, y_train,
    batch_size=___,
    epochs=___,
    verbose=1,
    validation_data=___
)
"""

P2_STEP5_SOLUTION = """\
epochs = 3

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)
"""

# ── Step 6 ──────────────────────────────────────────────────────────────────

P2_STEP6_MD = """\
## Step 6 — Evaluate the CNN
"""

P2_STEP6_STUDENT = """\
# TODO 6: Evaluate the CNN model on the test set
score = model.evaluate(___, ___, verbose=0)
print('Test loss:    ', ___)
print('Test accuracy:', ___)
"""

P2_STEP6_SOLUTION = """\
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:    ', score[0])
print('Test accuracy:', score[1])
"""

# ── Step 7 – Comparison ──────────────────────────────────────────────────────

P2_STEP7_MD = """\
## Step 7 — Compare MLP vs CNN

Answer these questions in a new markdown cell:

1. How many **trainable parameters** does the CNN have compared to the MLP?
   Which model is more parameter-efficient?
2. After only 3 epochs the CNN accuracy may be lower than the MLP after 10
   epochs. Why might this be, and what would happen with more epochs?
3. What is the role of `MaxPooling2D` and `Dropout` in the CNN?
4. When would you prefer a CNN over an MLP for image classification?
   When might an MLP be sufficient?
"""

# ---------------------------------------------------------------------------
# Build notebooks
# ---------------------------------------------------------------------------

def build_student():
    nb = new_notebook()
    cells = [
        md("# Week 06 Lab — Deep Learning Basics: MLP and CNN (Student)\n\n"
           "**Instructions**: fill in every `___` blank and implement every "
           "`# TODO` comment.\n"
           "Do *not* look at the solutions notebook until you have tried each "
           "step yourself!\n\n"
           "Run cells in order — each step depends on variables set in the "
           "previous one."),

        # Part 1
        md(PART1_TITLE),
        md(STEP1_MD),  code(STEP1_STUDENT),
        md(STEP2_MD),  code(STEP2_STUDENT),
        md(STEP3_MD),  code(STEP3_STUDENT),
        md(STEP4_MD),  code(STEP4_STUDENT),
        md(STEP5_MD),  code(STEP5_STUDENT),
        md(STEP6_MD),  code(STEP6_STUDENT),
        md(STEP7_MD),  code(STEP7_STUDENT),
        md(STEP8_MD),  code(STEP8_STUDENT),
        md(STEP9_MD),  code(STEP9_STUDENT),
        md(STEP10_MD), code(STEP10_STUDENT),
        md(STEP11_MD),

        # Part 2
        md(PART2_TITLE),
        md(P2_STEP1_MD), code(P2_STEP1_STUDENT),
        md(P2_STEP2_MD), code(P2_STEP2_STUDENT),
        md(P2_STEP3_MD), code(P2_STEP3_STUDENT),
        md(P2_STEP4_MD), code(P2_STEP4_STUDENT),
        md(P2_STEP5_MD), code(P2_STEP5_STUDENT),
        md(P2_STEP6_MD), code(P2_STEP6_STUDENT),
        md(P2_STEP7_MD),
    ]
    nb['cells'] = cells
    return nb


def build_solutions():
    nb = new_notebook()
    cells = [
        md("# Week 06 Lab — Deep Learning Basics: MLP and CNN (Solutions)\n\n"
           "Complete reference solutions."),

        # Part 1
        md(PART1_TITLE),
        md(STEP1_MD),  code(STEP1_SOLUTION),
        md(STEP2_MD),  code(STEP2_SOLUTION),
        md(STEP3_MD),  code(STEP3_SOLUTION),
        md(STEP4_MD),  code(STEP4_SOLUTION),
        md(STEP5_MD),  code(STEP5_SOLUTION),
        md(STEP6_MD),  code(STEP6_SOLUTION),
        md(STEP7_MD),  code(STEP7_SOLUTION),
        md(STEP8_MD),  code(STEP8_SOLUTION),
        md(STEP9_MD),  code(STEP9_SOLUTION),
        md(STEP10_MD), code(STEP10_SOLUTION),
        md(STEP11_MD),

        # Part 2
        md(PART2_TITLE),
        md(P2_STEP1_MD), code(P2_STEP1_SOLUTION),
        md(P2_STEP2_MD), code(P2_STEP2_SOLUTION),
        md(P2_STEP3_MD), code(P2_STEP3_SOLUTION),
        md(P2_STEP4_MD), code(P2_STEP4_SOLUTION),
        md(P2_STEP5_MD), code(P2_STEP5_SOLUTION),
        md(P2_STEP6_MD), code(P2_STEP6_SOLUTION),
        md(P2_STEP7_MD),
    ]
    nb['cells'] = cells
    return nb


if __name__ == "__main__":
    student   = build_student()
    solutions = build_solutions()

    nbformat.write(student,   OUT_STUDENT)
    nbformat.write(solutions, OUT_SOLUTIONS)

    print("Wrote:", OUT_STUDENT)
    print("Wrote:", OUT_SOLUTIONS)
