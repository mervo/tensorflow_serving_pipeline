import tensorflow as tf

from tensorflow.keras.models import load_model
import efficientnet.tfkeras

import os
import h5py


def get_model(weight_path):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    model = load_model(weight_path)
    f = h5py.File(weight_path, mode='r')
    if 'class_names' in f.attrs:
        _class_name = f.attrs.get('class_names').list()
        class_name = [x.decode() for x in _class_name]

    else:
        class_name = [str(x) for x in range(model.layers[-1].output_shape[1])]
    f.close()
    return model, class_name


test, test1 = get_model('/home/user/Downloads/grayscale_shipspotting_efficientnetb7_teyr.h5')
print(test.input, test.output)

# model_input = build_tensor_info(test.input)
# model_output = build_tensor_info(test.output)

# Create signature definition for tfserving
# signature_definition = signature_def_utils.build_signature_def(
#     inputs=(placeholder_name: model_input),
#     outputs=(operation_name: model_output),
#     method_name=signature_constants.CLASSIFY_METHOD_NAME
# )

tf.saved_model.save(test, './test')
