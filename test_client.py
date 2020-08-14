from __future__ import print_function

import fire
import requests
import json
import tensorflow as tf

import numpy as np


# The server URL specifies the endpoint of your server running the
# model and using the predict interface.

def image_preproc_for_model_input(path_to_your_image):
    '''
    TODO
    define your preprocessing here to suit your model input
    :param path_to_your_image:
    :return: input to model
    '''
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU

    img = tf.keras.preprocessing.image.load_img(path_to_your_image, grayscale=True, color_mode='rgb',
                                                target_size=(320, 240))
    output = tf.keras.preprocessing.image.img_to_array(img)
    output = tf.expand_dims(output, 0).numpy()
    print(output.shape)
    # print(f'>>>> Input to model: {output}')
    output = output.tolist()

    return output


def main(path_to_your_image, server_url):
    # Get single image for testing
    input = image_preproc_for_model_input(path_to_your_image)

    # Compose a JSON Predict request
    predict_request = json.dumps({
        'instances': input
    })
    headers = {"content-type": "application/json"}

    # Send a few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(url=server_url, data=predict_request, headers=headers)
        # response.raise_for_status()

    # Send a few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(server_url, data=predict_request, headers=headers)
        # response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        # print(response.content)
        prediction = response.json()['predictions'][0]  # Single input test image
        print(prediction)

    print('Prediction class: {}, avg latency: {} ms'.format(
        np.argmax(prediction), (total_time * 1000) / num_requests))


if __name__ == '__main__':
    fire.Fire(main)
