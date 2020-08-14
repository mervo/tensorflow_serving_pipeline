#docker build -t tensorflow_serving_server .

## If model was copied into image
#docker run --runtime=nvidia -p 8501:8501 \
#  -e MODEL_NAME=model -t tensorflow_serving_server &

## If mounting in your model
#docker run --runtime=nvidia -p 8501:8501 \
#--mount type=bind,\
#source=/path/to/your/saved_model,\
#target=/models/model \
#  -e MODEL_NAME=model -t tensorflow_serving_server &

## Server is only ready when the following message appears, may take some time:
#2018-07-27 00:07:20.773693: I tensorflow_serving/model_servers/main.cc:333]
#Exporting HTTP/REST API at:localhost:8501 ...

## Test Server (for included saved_model_half_plus_two_gpu model)
#curl -d '{"instances": [1.0, 2.0, 5.0]}' \
#  -X POST http://localhost:8501/v1/models/model:predict

## Output
# {
#     "predictions": [2.5, 3.0, 4.5
#     ]
# }

## Test Server (for image classifiers, use classifier_client.py)

FROM tensorflow/serving:latest-gpu

COPY ./your_output_saved_model /models/model
WORKDIR /models/model

# ENV PATH /usr/local/cuda/bin/:$PATH
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES all
# LABEL com.nvidia.volumes.needed="nvidia_driver"
