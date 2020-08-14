# tensorflow_serving_pipeline
Pipeline to convert model, load into TensorFlow Serving Docker image, warmup and test model

Tested on Python 3.7

## Converting of Model
E.g. of usage: `python convert_model.py convert_xception_h5_model ./your_input_model/model.h5 ./your_output_saved_model/1`

`1` refers to the model version number.

Example used: https://medium.com/analytics-vidhya/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206

## Warming up of Model (Optional)
The TensorFlow runtime has components that are lazily initialized, which can cause high latency for the first request/s sent to a model after it is loaded. This latency can be several orders of magnitude higher than that of a single inference request.

To reduce the impact of lazy initialization on request latency, it's possible to trigger the initialization of the sub-systems and components at model load time by providing a sample set of inference requests along with the SavedModel. This process is known as "warming up" the model.

Refer to https://www.tensorflow.org/tfx/serving/saved_model_warmup

## Loading of Model into your TensorFlow Serving Docker image/container
Refer to `Dockerfile` for more details. Adapt as necessary for use-case.

Serving multiple models: https://www.tensorflow.org/tfx/serving/serving_config#model_server_configuration
### Copying of Model into Image
Copy your model into your Docker image by copying your model into `your_output_saved_model`.

`docker build -t tensorflow_serving_server .`
```
docker run --runtime=nvidia -p 8501:8501 \
-e MODEL_NAME=model -t tensorflow_serving_server &
```

### Mounting of Model into Container
```
docker run --runtime=nvidia -p 8501:8501 \
--mount type=bind,\
source=/path/to/your/saved_model,\
target=/models/model \
 -e MODEL_NAME=model -t tensorflow_serving_server &
```

## Quick Test using `saved_model_half_plus_two_gpu model`
Use contents from `sample_models` into `your_model_here` for quick testing. Refer to "Copying Of Model into Image" above.

Test Server (for included saved_model_half_plus_two_gpu model)
```
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
 -X POST http://localhost:8501/v1/models/model:predict
```

Server is only ready when the following message appears, may take some time:
```
2018-07-27 00:07:20.773693: I tensorflow_serving/model_servers/main.cc:333]
Exporting HTTP/REST API at:localhost:8501 ...
```

## Testing Your Model
Change `image_preproc_for_model_input` in `test_client.py` to preprocess your input image to match your model's input.

`python test_client.py ./test.jpg http://localhost:8501/v1/models/model:predict`

## Errors
`E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR`

Should be caused by TensorFlow using up all of the available GPU memory, use the following arguments to run your container if you are testing on your development environment and not running a dedicated server:

`docker run --runtime=nvidia -p 8501:8501 -e MODEL_NAME=model -t tensorflow_serving_server --per_process_gpu_memory_fraction=0.7 --enable_batching=true --tensorflow_session_parallelism=2`

https://github.com/tensorflow/serving/issues/1440#issuecomment-541578798

