from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from models import model_builder
from utils import read_config
import argparse
import pandas as pd



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir',
                        default='/covidxray/',
                        help='Path to image')

    parser.add_argument('--config',
                        default='config.ini',
                        help='Path to the config file')
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = args_parser()

    configs = read_config(args.config)

    train = pd.read_csv(configs['path']['train'])

    train_gen = ImageDataGenerator(**configs['preprocess']['train'])

    val_gen = ImageDataGenerator(**configs['preprocess']['test'])

    test_gen = ImageDataGenerator(**configs['preprocess']['test'])

    training_data = train_gen.flow_from_dataframe(directory=args.dir,
                                                shuffle=True,
                                                x_col='filename',
                                                y_col='label',
                                                batch_size=configs['train_cfg']['batch_size'],
                                                target_size=configs['train_cfg']['target_shape'])

    validation_data = val_gen.flow_from_dataframe(directory=args.dir,
                                                shuffle=True,
                                                x_col='filename',
                                                y_col='label',
                                                batch_size=configs['train_cfg']['batch_size'],
                                                target_size=configs['train_cfg']['target_shape'])

    test_data = test_gen.flow_from_dataframe(directory=args.dir,
                                            x_col='filename',
                                            y_col='label',
                                            batch_size=configs['train_cfg']['batch_size'],
                                            target_size=configs['train_cfg']['target_shape'])


    history_logger = tf.keras.callbacks.CSVLogger(configs['path']['logs'], separator=",", append=True)

    tb_callback = tf.keras.callbacks.TensorBoard(configs['path']['tfboard'], update_freq=1)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=configs['path']['checkpoint'],
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)

    model = model_builder(configs['train_cfg']['backbone'])

    if configs['opt_cfg']['opt'] == 'adam':
        opt = tf.keras.optimizer.Adam(learning_rate=configs['opt_cfg']['lr'])
    elif configs['opt_cfg']['opt'] == 'sgd':
        opt = tf.keras.optimizer.SGD(learning_rate=configs['opt_cfg']['lr'])
    else:
        raise ValueError('Not support other type of optimizer') 

    loss=tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=opt, loss=loss, metrics='acc')

    history = model.fit(training_data,
                        epochs=configs['train_cfg']['epochs'],
                        validation_data=validation_data,
                        callbacks=[history_logger, tb_callback, model_checkpoint])
    
    results = model.evaluate(test_data)
    print("test loss, test acc:", results)
    
if __name__== '__main__':
    main()