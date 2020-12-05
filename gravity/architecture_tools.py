import tensorflow as tf


def get_vgg16(classes=10,
              input_shape=(32, 23, 3),
              drop_out_rate=0,
              print_summary=False):
    base_model = tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=classes,
        classifier_activation=None,
    )
    fc1 = base_model.layers[-3]
    fc2 = base_model.layers[-2]
    predictions = base_model.layers[-1]

    # Create the dropout layers
    dropout1 = tf.keras.layers.Dropout(drop_out_rate)
    dropout2 = tf.keras.layers.Dropout(drop_out_rate)

    # Reconnect the layers
    x = dropout1(fc1.output)
    x = fc2(x)
    x = dropout2(x)
    predictors = predictions(x)

    # Create a new model
    model_vgg16 = tf.keras.models.Model(base_model.input,
                                        predictors,
                                        name='vgg16')

    if print_summary:
        model_vgg16.summary()
    return model_vgg16


def resize_model_input_size(base_model,
                            target_size=(32, 32),
                            data_shape=(28, 28, 1),
                            print_summary=False):
    resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(
        target_size[0], target_size[1])
    inputs = tf.keras.Input(shape=data_shape)
    x = resize_layer(inputs)
    outputs = base_model(x)
    resized_model = tf.keras.Model(inputs, outputs, name=base_model.name)
    if print_summary:
        resized_model.summary()
    return resized_model