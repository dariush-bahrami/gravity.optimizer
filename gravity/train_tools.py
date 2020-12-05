"""This module contain functions needed in saving and visalizing training info"""
import time
from zipfile import ZipFile
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def save_run_info(model, history, optimizer, dataset: str, comment=None):
    # Load Data
    model_name = model.name
    optimizer_name = optimizer.get_config()['name']
    model_config = model.to_json()
    history_dict = {}
    epochs = len(history.history['loss'])
    history_dict['epochs'] = list(range(1, epochs+1))
    history_dict.update(history.history)
    optimizer_config = optimizer.get_config()
    for i in optimizer_config:
        if type(optimizer_config[i]) not in (str, float) :
            optimizer_config[i] = round(float(optimizer_config[i]), 4)
    # Serializing
    max_acc = round(max(history_dict['accuracy']), 2)
    max_val_acc = round(max(history_dict['val_accuracy']), 2)
    acc_str = f'{max_val_acc:.2f}{max_acc:.2f}'.replace('.', '')
    time_str = time.strftime("%Y%m%d_%H%M%S")
    zipfile_name = f'{dataset}_{model_name}_{acc_str}_{optimizer_name}_{time_str}.zip'
    history_df = pd.DataFrame(history_dict)
    with ZipFile(zipfile_name, mode='w') as zip_file:
        zip_file.writestr(f'history.csv', history_df.to_csv(index=False))
        zip_file.writestr(f'dataset.txt', dataset)
        zip_file.writestr(f'model_config.json',
                          json.dumps(model_config, indent=4))
        zip_file.writestr(f'optimizer_config.json',
                          json.dumps(optimizer_config, indent=4))
        if comment:
            zip_file.writestr('comment.txt', comment)
    print('save_run_info completed')
