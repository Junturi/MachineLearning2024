# Code provided by our teacher starts here
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras import Model


def import_model(filepath: str) -> Model:
    # Load the Keras model from the file system
    model: Model = models.load_model(filepath)
    return model


def get_layer_weights(layer: str, model: Model) -> list[np.ndarray]:
    # Get the weights from the Keras model layer
    return model.get_layer(layer).get_weights()


# Code provided by our teacher ends here.
# My own code starts here.

def make_spectrogram(waveform):
    # Turn the audio wave into a spectrogram
    waveform = waveform.numpy()
    frame_length = 255
    frame_step = 128

    frames = list()

    # We will need to do a short-time Fourier transformation to turn the waveform for a spectrogram
    # For STFT we need to take frames out of the wave, and we will need to use some kind of window function

    # Calculate how many frames we need to take
    num_frames = (waveform.shape[0] - frame_length) // frame_step + 1

    # Iterate through the number of frames and get the correct piece of data for the frame
    for i in range(num_frames):
        frame = i * frame_step
        frames.append(waveform[frame:frame + frame_length])

    frames = np.array(frames)

    # Get the window function, in this case a Hann window
    window = 0.5 - (0.5 * np.cos(2.0 * np.pi * np.arange(frame_length + 1) / (frame_length - 1)))
    window = window[:-1]

    # Multiply the frames with the window and do a real fast Fourier transformation
    frames = frames * window
    spectrogram = np.fft.rfft(frames)

    # Take the absolute values of the result
    spectrogram = np.abs(spectrogram)

    return spectrogram


def resize_image(data):
    # Resize the data to smaller size by taking every 4th pixel from the spectrogram
    # Some padding is needed in order to get a 32 x 32 image expected by rest of the functions
    padded_image = np.pad(data, pad_width=((2, 1), (0, 0)), mode='constant', constant_values=0)
    image = padded_image[::4, 1::4]

    return image


def normalize(data):
    # Normalize the data with the weights extracted from the corresponding Keras model layer
    weights = get_layer_weights("normalization", model)
    mean = weights[0]
    variance = weights[1]
    image_rows = data.shape[0]
    image_columns = data.shape[1]

    final_image = np.zeros((image_rows, image_columns))

    # Iterate through every pixel in the data and normalize
    for row in range(image_rows):
        for column in range(image_columns):
            pixel = data[row][column]
            pixel = ((pixel - mean) / np.sqrt(variance)).item()
            final_image[row][column] = pixel

    # Add one axis to the array, as 3D array is expected in the next step
    final_image = final_image[..., np.newaxis]
    return final_image


def relu(data):
    # ReLu function used to turn negative numbers into 0
    for rows in range(len(data)):
        for columns in range(len(data[rows])):
            if data[rows][columns] < 0:
                data[rows][columns] = 0

    return data


def convolute(dataset, filters, bias):
    # Filter the images with filters extracted from the convolution layer
    final_images = list()

    image_rows, image_columns, image_number = dataset.shape
    filter_rows = filters.shape[0]
    filter_columns = filters.shape[1]
    output_height = image_rows - filter_rows + 1
    output_width = image_columns - filter_columns + 1

    for filter in range(filters.shape[3]):
        final_image = np.zeros((output_height, output_width))
        for image in range(image_number):
            filtered_image = np.zeros((output_height, output_width))
            kernel = filters[:, :, image, filter]
            for row in range(output_height):
                for column in range(output_width):
                    sub_matrix = dataset[row: row + filter_rows, column: column + filter_columns, image]
                    filtered_image[row, column] = np.sum(sub_matrix * kernel)
            # All images going through one filter are added together to make one image from the filtering results
            final_image = final_image + filtered_image
        # Bias is added once all images have gone through one filter
        final_image = final_image + bias[filter]
        final_image = relu(final_image)
        final_images.append(final_image)
    final_images = np.array(final_images)
    # Array layers are reordered to get the correct format
    final_images = np.transpose(final_images, (1, 2, 0))

    return final_images


def convolution1(dataset):
    # Extract filters and biases from the corresponding Keras model layer
    weights = get_layer_weights("conv2d", model)
    filters, bias = weights

    # Call the convolute function to filter the images
    images = convolute(dataset, filters, bias)

    return images


def convolution2(dataset):
    # Extract filters and biases from the corresponding Keras model layer
    weights = get_layer_weights("conv2d_1", model)
    filters, bias = weights

    # Call the convolute function to filter the images
    images = convolute(dataset, filters, bias)

    return images


def maxpooling(dataset):
    # Find the highest value in a 2x2 area of an image and create new image
    rows, columns, images = dataset.shape
    maximages = list()

    for image in range(images):
        temp = list()
        for row in range(0, rows, 2):
            row_values = list()
            for column in range(0, columns, 2):
                window = [dataset[row, column, image], dataset[row + 1, column, image], dataset[row, column + 1, image],
                          dataset[row + 1, column + 1, image]]
                max_value = max(window)
                row_values.append(max_value)
            temp.append(row_values)
        maximages.append(temp)

    maximages = np.array(maximages)
    # Array layers are reordered to get the correct format
    maximages = np.transpose(maximages, (1, 2, 0))

    return maximages


def flattening(images):
    # Reshape the data from multiple images into one row of an array
    return images.reshape(1, -1)


def matrix_multiplication(data, weights, biases):
    # Perform matrix multiplication
    result = np.zeros((len(data), len(weights[0])))

    for value_in_a1 in range(len(data)):
        for rows_in_w2 in range(len(weights[0])):
            for weight_in_w2 in range(len(weights)):
                result[value_in_a1][rows_in_w2] += data[value_in_a1][weight_in_w2] * weights[weight_in_w2][rows_in_w2]

    for index in range(len(result)):
        result[index] + biases[index]

    return result


def dense1(data):
    # Get the weights and biases from corresponding Keras model layer
    weights_from_model = get_layer_weights("dense", model)
    weights = weights_from_model[0]
    biases = weights_from_model[1]

    # Call the matrix multiplication function to perform the calculation
    result = matrix_multiplication(data, weights, biases)

    # On this layer, the result needs to go through the ReLu function
    result = relu(result)

    return result


def dense2(data):
    # Get the weights and biases from corresponding Keras model layer
    weights_from_model = get_layer_weights("dense_1", model)
    weights = weights_from_model[0]
    biases = weights_from_model[1]

    result = matrix_multiplication(data, weights, biases)

    return result


if __name__ == "__main__":
    """
    # Import all the speech commands and unzip
    DATASET_PATH = 'data/mini_speech_commands'

    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data')"""


    # Read the audio from file and turn it into a spectrogram
    audio = 'fc94edb0_nohash_0.wav' # Insert here the audio file path you want to use
    data = tf.io.read_file(str(audio))
    data, sample_rate = tf.audio.decode_wav(data, desired_channels=1, desired_samples=16000)
    data = tf.squeeze(data, axis=-1)
    data = make_spectrogram(data)

    # Import the Keras model and get the model's result
    # The model expects data to be in a 4D array, so adding two more axis is needed
    model = import_model('model_export.keras')
    model_data = data[tf.newaxis, ..., tf.newaxis]
    model_prediction = model(model_data)

    # Call our own functions representing the model layers to make our own prediction
    res_data = resize_image(data)
    norm_image = normalize(res_data)
    con1_image = convolution1(norm_image)
    con2_image = convolution2(con1_image)
    max_image = maxpooling(con2_image)
    flat_image = flattening(max_image)
    dense1_image = dense1(flat_image)
    my_prediction = dense2(dense1_image)

    # Plot both predictions for comparison
    x_labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.bar(x_labels, tf.nn.softmax(model_prediction[0, :]))
    plt.title("Model prediction")
    plt.subplot(1, 2, 2)
    plt.bar(x_labels, tf.nn.softmax(my_prediction[0, :]))
    plt.title("My prediction")
    plt.show()
