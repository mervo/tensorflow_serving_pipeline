import tensorflow as tf

from tensorflow.keras.models import load_model

import os
import h5py
import fire


class ModelConverter():
    """ Class to help convert models to SavedModel used by TensorFlow Serving """

    def convert_xception_h5_model(self, path_to_h5_weights, path_to_output):
        import efficientnet.tfkeras
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        model = load_model(path_to_h5_weights)
        f = h5py.File(path_to_h5_weights, mode='r')
        if 'class_names' in f.attrs:
            _class_name = f.attrs.get('class_names').list()
            class_name = [x.decode() for x in _class_name]

        else:
            class_name = [str(x) for x in range(model.layers[-1].output_shape[1])]
        f.close()

        print(f'>>> Model\'s Input: {model.input}, Model\'s Output: {model.output}, Classes: {class_name} <<<')

        tf.saved_model.save(model, path_to_output)


if __name__ == '__main__':
    fire.Fire(ModelConverter)
