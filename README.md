# tensorflow_serving_pipeline
Pipeline to convert model, load into TensorFlow Serving Docker image, warmup and test model

Tested on Python 3.7

## Converting of Model
E.g. of usage: `python convert_model.py convert_xception_h5_model ./sample_models/best_xception_based_mnist/best_xception_based_mnist.h5 ./your_output_saved_model/1`

`1` refers to the model version number.

## Warming up of Model (Optional)
The TensorFlow runtime has components that are lazily initialized, which can cause high latency for the first request/s sent to a model after it is loaded. This latency can be several orders of magnitude higher than that of a single inference request.

To reduce the impact of lazy initialization on request latency, it's possible to trigger the initialization of the sub-systems and components at model load time by providing a sample set of inference requests along with the SavedModel. This process is known as "warming up" the model.

Refer to https://www.tensorflow.org/tfx/serving/saved_model_warmup

## Loading of Model into your TensorFlow Serving Docker container/image
Either mount your model folder into your container at runtime or copy your model into your Docker image by copying your model into `your_model_here`. Refer to `Dockerfile` for examples of commands.

## Quick Test
Copy contents of desired sub-folder from `sample_models` into `your_model_here` for quick testing.

