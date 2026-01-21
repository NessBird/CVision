# Program 3: Cats and Dogs
# Ness Blackbird for Computer Vision 510.

import cv2
import numpy as np
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras import layers
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import models
from keras import Model
import os
import matplotlib.pyplot as plt
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf
import sklearn as sk


def visualize_filters(model):
    # Use the first Conv2D layer. It's the second layer.
    first_conv = model.layers[1]

    # Get the filters.
    filters = first_conv.get_weights()[0]
    print(f"Filter shape: {filters.shape}")

    # Normalize values to 0-1, it's easier to see what's going on.
    f_min, f_max = filters.min(), filters.max()
    filters_norm = (filters - f_min) / (f_max - f_min)

    # Graph all the filters. There are 32.
    n_filters = filters.shape[-1]
    # We'll make them little, in rows of 8. Each will be a little 12x12.
    fig, axes = plt.subplots(4, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters_norm[:, :, :, i])  # 7x7x3 RGB image
        ax.axis('off')
        ax.set_title(str(i), fontsize=14)
    plt.suptitle('First LayerFilters')
    plt.tight_layout()
    plt.show()

def pad_image(image, label):
    # Used by dataset.map().
    image = tf.image.resize_with_pad(image, 150, 150)
    return image, label

def preprocess_pair(img, label):
    # This needs to be a function instead of a callback because preprocess_input only takes one parameter, and
    # it'll return an error when it gets the image/label combination that .map() sends.
    return preprocess_input(img), label

def run_once(layer_name, epochs = 10, use_pooling = False):
    # Load the InceptionResNetV2 model, deciding where to truncate it based on the layer parameter,
    # then add a categorization header to distinguish cats from dogs.
    # Then train it, and run the test data through to see how it's working.
    # If layer is "no-train", use the whole model, but don't train it, just evaluate it.
    # Start with a pretrained CNN model from InceptionResNetV2. It's a big fancy thing with like 200 layers.
    # It already knows how to distinguish many things including different types of dogs and cats.
    print('Experiment: ', layer_name)
    if use_pooling:
        print("Using pooling")
    pre_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    pre_model.trainable = False

    if layer_name == 'no-train' or layer_name == 'base':
        # Start with a Sequential base model, and the regular pretrained model.
        model = models.Sequential()
        model.add(pre_model)
        # Flatten its output into a long vector.
        model.add(layers.Flatten())
        # Add a dense layer with ReLU activation.
        model.add(layers.Dense(units=256, activation='relu'))
        # Final output with a single neuron: A binary dog/cat selector.
        model.add(layers.Dense(units=1, activation='sigmoid'))
    else:
        # Truncate it at the given layer. We have to build a new version of the pretrained model, with the same input
        # layer, but a new output layer. So get the layer we want to truncate at, supplied in the parameter. Using ".output"
        # will make the return value an output tensor (the object, without any data).

        # This is not a Sequential model, it's a Functional one. This gives us the flexibility we need to mess around with layers.
        x = pre_model.get_layer(layer_name).output
        if use_pooling:
            # On mixed_5b, I want to try reducing the size of the pretrained model output -- it's 20M if I don't do anything.
            x = GlobalAveragePooling2D(None, True)(x)

        # Add a Flatten layer to turn it into a vector. This records the connection between layer_name and
        # the new Flatten layer. Again, it returns a tensor stub.
        # The cool part about this system is that using this tensor object in this way actually tells the model,
        # "Your next step, after where x came from (either layer_name or pooling) is to do the Flatten layer."
        x = Flatten()(x)

        # And a Dense layer that can learn the dog/cat difference.
        x = Dense(256, activation='relu')(x)
        # Categorization layer. This builds our little output tensor.
        output = Dense(1, activation='sigmoid')(x)
        # Now string it together from input (of the pre_model) to the categorization layer. Model() actually traces
        # a route back from outputs to the input of pre_model. This route actually includes branching in pre_model,
        # but it comes back together to a single input.
        model = Model(inputs=pre_model.input, outputs=output)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Don't train it at all for one experiment.
    if layer_name != 'no-train':
        model.fit(
            training_ds,
            epochs=epochs,
            validation_data=test_ds,
            verbose=2
        )

    t_dataset = test_ds

    predictions = model.predict(t_dataset, verbose=1)
    # Pull out the labels. They have to, first, be turned into numpy arrays, and second, flattened out, because
    # they're in format [[1], [1], [1]...], that is, shape (32, 1).
    labels = np.concatenate([lb.numpy().flatten() for im, lb in t_dataset])

    # predicted_labels is a similar story. It's (2000, 1), and it's full of floating point values.
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # Print metrics.
    print('Accuracy: ', sk.metrics.accuracy_score(labels, predicted_labels))
    print(sk.metrics.confusion_matrix(labels, predicted_labels))

d = '/mnt/c/Users/Ness/Documents/PSU/Comp Vision/Prog3/dataset/'
training_ds = tf.keras.utils.image_dataset_from_directory(
    d + 'training_set',
    image_size=(150, 150),
    batch_size=32,
)
training_ds = training_ds.map(preprocess_pair)

test_ds = tf.keras.utils.image_dataset_from_directory(
    d + 'test_set',
    image_size=(150, 150),
    batch_size=32,
    # Turn off shuffle for the test data -- otherwise, it will keep reordering itself, which makes it complicated
    # to check labels vs predictions.
    shuffle=False,
)
test_ds = test_ds.map(preprocess_pair)

# Graph all the filters.
pre_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
visualize_filters(pre_model)

# Now run some experiments.
run_once('no-train')
run_once('base', 6)
run_once('mixed_5b', 12)
# mixed_5b has a really wide output. Try using global pooling to reduce it.
run_once('mixed_5b', 18,True)
run_once('mixed_6a', 9)
run_once('mixed_7a', 5)
