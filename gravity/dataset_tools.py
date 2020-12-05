"""This module contain functions related to benchmark datasets"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_last')


def print_dataset_info(name, x_train, y_train, x_test, y_test):
    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]
    print(f'{name} dataset information:')
    print(f'  * Total Train Samples: {train_samples}')
    print(f'  * Total Test Samples:  {test_samples}')
    print(f'  * Input Shape:  {x_train.shape[1:]}')
    print(f'  * Output Shape: {y_train.shape[1:]}')
    print(f'Shapes Detail:')
    print(f'  * x_train:  {x_train.shape}')
    print(f'  * y_train:  {y_train.shape}')
    print(f'  * x_test:   {x_test.shape}')
    print(f'  * y_test:   {y_test.shape}')


def show_random_image(name, x_train, y_train, x_test, y_test):
    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]
    if x_train.ndim == 4 and x_train.shape[-1] not in (3, 4):
        print(f'Invalid image shape {x_train.shape[1:]} for image data')
        x_train = x_train.reshape(x_train.shape[:-1])
        x_test = x_test.reshape(x_test.shape[:-1])
        print(f'Images reshaped to: {x_train.shape[1:]}')
    print('Random Train Example: ')
    index = np.random.randint(0, train_samples)
    plt.imshow(x_train[index, ])
    plt.show()
    print(f'Label: {y_train[index,]}')
    print('Random Test Example: ')
    index = np.random.randint(0, test_samples)
    plt.imshow(x_test[index, ])
    plt.show()
    print(f'Label: {y_test[index,]}')


def print_cifar10_classes():
    num_classes = 10
    classes = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    print('CIFAR10 Classes:')
    for i in range(num_classes):
        print(f'  {i}- {classes[i]}')


def get_dataset_mnist(verbose=True, show_random_image=True):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist = x_train_mnist.reshape((x_train_mnist.shape[0], 28, 28, 1)).astype('float32')
    x_test_mnist = x_test_mnist.reshape((x_test_mnist.shape[0], 28, 28, 1)).astype('float32')
    x_train_mnist, x_test_mnist = x_train_mnist / 255.0, x_test_mnist / 255.0
    input_shape_mnist = x_train_mnist.shape[1:]
    classes_mnist = 10
    if verbose:
        print_dataset_info('MNIST', x_train_mnist, y_train_mnist, x_test_mnist,
                        y_test_mnist)
    if show_random_image:
        show_random_image('MNIST', x_train_mnist, y_train_mnist, x_test_mnist,
                        y_test_mnist)

    result_dict = {
        'train_data': (x_train_mnist, y_train_mnist),
        'test_data': (x_test_mnist, y_test_mnist),
        'classes': classes_mnist
    }
    return result_dict


def get_dataset_cifar10(verbose=True, show_random_image=True):
    (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = tf.keras.datasets.cifar10.load_data()
    x_train_cifar10 = x_train_cifar10.astype('float32')
    x_test_cifar10 = x_test_cifar10.astype('float32')
    x_train_cifar10, x_test_cifar10= x_train_cifar10 / 255, x_test_cifar10 / 255
    input_shape_cifar10 = x_train_cifar10.shape[1:]
    classes_cifar10 = 10
    if verbose:
        print_dataset_info('CIFAR10', x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10)
        print_cifar10_classes()
    if show_random_image:
        show_random_image('CIFAR10', x_train_cifar10, y_train_cifar10, x_test_cifar10, y_test_cifar10)

    result_dict = {
        'train_data': (x_train_cifar10, y_train_cifar10),
        'test_data': (x_test_cifar10, y_test_cifar10),
        'classes': classes_cifar10
    }
    return result_dict
