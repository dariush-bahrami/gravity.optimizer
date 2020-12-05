"""This module contain functions related to benchmark datasets"""
import numpy as np
import matplotlib.pyplot as plt


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
