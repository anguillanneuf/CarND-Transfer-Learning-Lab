import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Reshape
from keras.utils import np_utils
from keras.regularizers import l2
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg-100/vgg_cifar10_100_bottleneck_features_train.p', 
"Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'vgg-100/vgg_cifar10_bottleneck_features_validation.p', 
"Bottleneck features validation file (.p)")
flags.DEFINE_string('n_epochs', 10, "Define number of epochs, default is 10")
flags.DEFINE_string('batch_size', 16, "Define batch size, default is 16")

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, 
                                                          FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    n_class = np.unique(y_train).shape[0]
    batch_size = int(FLAGS.batch_size)
    n_epoch = int(FLAGS.n_epochs)
    
    y_train = np_utils.to_categorical(y_train, n_class)
    y_val = np_utils.to_categorical(y_val, n_class)
    input_shape = X_train.shape[1:]
    
    model = Sequential()
    model.add(Reshape((np.prod(input_shape),),input_shape=input_shape))
    model.add(Dense(128, activation="relu", W_regularizer = l2(1e-2),
                    input_shape=(np.prod(X_train.shape[1:]),)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_class, activation="softmax"))
    # model.summary()
    model.compile(loss ='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

    
    # TODO: train your model here
    model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=n_epoch,
                    verbose=1, validation_data=(X_val, y_val))
    
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

"""
--n_epoch 25
--batch_size 512

Cifar10
vgg validation: 73.77%
resnet validation: 74.20%
inception validation: 64.95%

Traffic Sign
vgg validation: 84.05%
resnet validation: 76.84%
inception validation: 69.88%
    
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  vgg-100/vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg-100/vgg_cifar10_bottleneck_features_validation.p
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  vgg-100/vgg_traffic_100_bottleneck_features_train.p --validation_file vgg-100/vgg_traffic_bottleneck_features_validation.p
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  resnet-100/resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet-100/resnet_cifar10_bottleneck_features_validation.p
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  resnet-100/resnet_traffic_100_bottleneck_features_train.p --validation_file resnet-100/resnet_traffic_bottleneck_features_validation.p
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  inception-100/inception_cifar10_100_bottleneck_features_train.p --validation_file inception-100/inception_cifar10_bottleneck_features_validation.p
python feature_extraction.py --n_epochs 25 --batch_size 512 --training_file  inception-100/inception_traffic_100_bottleneck_features_train.p --validation_file inception-100/inception_traffic_bottleneck_features_validation.p


"""