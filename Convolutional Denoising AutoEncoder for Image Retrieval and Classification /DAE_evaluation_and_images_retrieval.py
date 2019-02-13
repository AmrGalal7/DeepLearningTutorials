"""
    Functions:
    ---------
    - evaluation of Denoising AutoEncoder
    - plotting the denoised image of a noisy one
    - testing the encoder model with 'label_ranking_average_precision_score' ( uncomment 'test_model' function )
    - retrieval of the 'n_closest_images' to a query image
"""


import tensorflow as tf
import numpy as np
from keras.models import Model, load_model
from keras.datasets import mnist
import cv2
from sklearn.metrics import label_ranking_average_precision_score


def evaluate_and_plot_denoised_images(model, ae_input, ae_target, image_number = 0):
    """
        - evaluates the Denoising AutoEncoder (DAE) performance ( between 'x_test_noisy' and 'x_test' )
        - outputs the DAE's denoised images
        - plots an arbitrary noisy image before and after applying DAE
    """
    score = model.evaluate(ae_input, ae_target, verbose = 0)
    print('Test loss:', score)
    # output the denoised images of the noisy input images
    denoised_images = model.predict(ae_input)
    print('denoisied images shape: ', denoised_images.shape)
    # pick one of the noisy images, and show it
    test_img = ae_input[image_number]
    resized_test_img = cv2.resize(test_img, (280, 280))
    cv2.imshow('input', resized_test_img)
    cv2.waitKey(0)
    # select the corresponding denoised image, and show it
    output = denoised_images[image_number]
    resized_output = cv2.resize(output, (280, 280))
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)

def retrieve_images(model, x_train, learned_codes, query_image, n_closest_images = 5):
    """
        retrieves the 'n_closest_images' to the 'query_image'
    """

    # Compute Features for the query image
    test_code = model.predict(query_image.reshape(1, query_image.shape[0], query_image.shape[1], 1))
    test_code = np.reshape(test_code, (1, -1))

    # For each query image feature we compute the closest images from training dataset
    print('computing distances for the test code...')
    distances = []
    # Compute the euclidian distance for each feature from training dataset
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)

    # Store the computed distances and corresponding labels from training dataset
    distances = np.array(distances)
    # find the 'n_closest_images' training examples according to their distances from the query images ( ascendingly )
    indices = distances.argsort()
    retrieved_images_array = x_train[indices[:n_closest_images], :, :, :]

    # stack the retrieved images horizontally for display
    retrieved_images_stacked_horiz = cv2.resize(retrieved_images_array[0, :, :, 0], (280, 280))

    if(retrieved_images_array.shape[0] > 1):
        for i in range(1, retrieved_images_array.shape[0]):
            retrieved_images_stacked_horiz = np.concatenate( (retrieved_images_stacked_horiz, \
                                                              cv2.resize(retrieved_images_array[i, :, :, 0], (280, 280))), axis = -1)

    resized_query_image = cv2.resize(query_image, (280, 280))
    cv2.imshow('query image', resized_query_image)
    cv2.imshow('retrieved images', retrieved_images_stacked_horiz)
    cv2.waitKey(0)



def test_model(model, x_test, y_test, n_test_samples, learned_codes, n_closest_images):
    """
        uses func: 'compute_average_precision_score' to evaluate the model's performance on test data
    """


    # Compute Representation/Features for the test dataset
    test_codes = model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    test_codes = np.reshape(test_codes, (test_codes.shape[0], -1))

    # We keep only n_test_samples query images from test dataset
    indices =  np.arange(len(y_test))
    np.random.shuffle(indices)
    indices = indices[:n_test_samples]

    # Compute score
    score = compute_average_precision_score(test_codes = test_codes[indices], test_labels = y_test[indices], \
                                            learned_codes = learned_codes, y_train =  y_train, n_samples = n_closest_images)

    print('model score: ', score)



def compute_average_precision_score(test_codes, test_labels, learned_codes, y_train, n_closest_images):
    """
        calculates the distances between each test code and all learned codes, then \
        uses func: 'label_ranking_average_precision_score' to compute the model score, for the 'n_closest_images'
    """

    print('computing score...')

    # For each n_closest_images (number of retrieved images to assess) we store the corresponding labels and distances
    out_labels = []
    out_distances = []

    # For each query image feature we compute the closest images from training dataset
    for i in range(len(test_codes)):
        print('computing distances for test code: ', i)
        distances = []
        # Compute the euclidian distance for each feature from training dataset
        for code in learned_codes:
            distance = np.linalg.norm(code - test_codes[i])
            distances.append(distance)

        # Store the computed distances and corresponding labels from training dataset
        distances = np.array(distances)
        # Scoring function needs to replace similar labels by 1 and different ones by 0
        labels = np.copy(y_train).astype('float32')
        labels[labels != test_labels[i]] = -1
        labels[labels == test_labels[i]] = 1
        labels[labels == -1] = 0
        distance_with_labels = np.stack( (distances, labels), axis = -1 )
        sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]
        # The distances are between 0 and 28. The lesser the distance the bigger the relevance score should be
        sorted_distances = max(distances) - sorted_distance_with_labels[:, 0]
        sorted_labels = sorted_distance_with_labels[:, 1]
        # We keep only 'n_closest_images' closest elements from the images retrieved
        out_distances.append(sorted_distances[:n_closest_images])
        out_labels.append(sorted_labels[:n_closest_images])


    out_labels = np.array(out_labels)
    out_distances = np.array(out_distances)

    # Score the model based on n_samples first images retrieved
    score = label_ranking_average_precision_score(out_labels, out_distances)

    return score


def main():

    # Load mnist dataset
    print('loading mnist dataset...')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # Load previously trained autoencoder
    print('Loading model...')
    autoencoder = load_model('autoencoder.h5')

    # Evaluate the DAE model on 'x_test_noisy' with 'x_test', and show the denoised 'image_number' image from 'x_test'
    # evaluate_and_plot_denoised_images(model = autoencoder2, ae_input = x_test_noisy, ae_target = x_test, image_number = 0)

    # Build the encoder model used to find the images Representation/Features
    encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('encoded').output)

    # In order to save time on computations we keep only 1000 query images from test dataset
    n_test_samples = 1000

    # number of closest images to retrieve
    n_closest_images = 10

    # Compute Representation/Features for the training images
    learned_codes = encoder.predict(x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    learned_codes = np.reshape(learned_codes, (learned_codes.shape[0], -1))

    # Test the encoder model on 'n_test_samples' from 'x_test', and output the 'label_ranking_average_precision_score' score
    # test_model(model  = encoder, x_test = x_test, y_test = y_test, learned_codes = learned_codes, n_closest_images = n_closest_images)

    # Retrieve the 'n_closest_images'
    retrieve_images(model = encoder, x_train = x_train, learned_codes = learned_codes, query_image = x_test[0], n_closest_images = 5)



if __name__ == '__main__':
    main()
