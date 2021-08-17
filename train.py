from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from models import model_builder, update_params
from utils import read_config, scheduler
import argparse
import pandas as pd
import ast


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
    val = pd.read_csv(configs['path']['val'])
    test = pd.read_csv(configs['path']['test'])

    train['label'] = train['label'].astype('str')
    val['label'] = val['label'].astype('str')
    test['label'] = test['label'].astype('str')

    train_gen = ImageDataGenerator(**ast.literal_eval(configs['preprocess']['train']))

    val_gen = ImageDataGenerator(**ast.literal_eval(configs['preprocess']['test']))

    test_gen = ImageDataGenerator(**ast.literal_eval(configs['preprocess']['test']))

    training_data = train_gen.flow_from_dataframe(dataframe=train,
                                                directory=args.dir,
                                                shuffle=True,
                                                x_col='filename',
                                                y_col='label',
                                                class_mode="categorical",
                                                batch_size=int(configs['train_cfg']['batch_size']),
                                                target_size=ast.literal_eval(configs['train_cfg']['target_size']))

    validation_data = val_gen.flow_from_dataframe(dataframe=val,
                                                directory=args.dir,
                                                shuffle=True,
                                                x_col='filename',
                                                y_col='label',
                                                class_mode="categorical",
                                                batch_size=int(configs['train_cfg']['batch_size']),
                                                target_size=ast.literal_eval(configs['train_cfg']['target_size']))

    test_data = test_gen.flow_from_dataframe(dataframe=test,
                                            directory=args.dir,
                                            x_col='filename',
                                            y_col='label',
                                            class_mode="categorical",
                                            batch_size=int(configs['train_cfg']['batch_size']),
                                            target_size=ast.literal_eval(configs['train_cfg']['target_size']))

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    if configs['opt_cfg']['opt'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=float(configs['opt_cfg']['lr']))
    elif configs['opt_cfg']['opt'] == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=float(configs['opt_cfg']['lr']))
    else:
        raise ValueError('Not support other type of optimizer') 

    loss=tf.keras.losses.CategoricalCrossentropy()
    
      
    
    models_dict = {}    
    for i in range(1,4):
        if configs['opt_cfg']['opt'] == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=float(configs['opt_cfg']['lr']))
        elif configs['opt_cfg']['opt'] == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=float(configs['opt_cfg']['lr']))
        else:
            raise ValueError('Not support other type of optimizer')

        models_dict['model_{}'.format(i)] = model_builder(configs['train_cfg']['backbone'])
        models_dict['model_{}'.format(i)].compile(optimizer=opt, loss=loss, metrics='acc')

        save_path = configs['path']['directory'] + configs['train_cfg']['backbone'] + '/model_{}/'.format(i)

        history_logger = tf.keras.callbacks.CSVLogger(save_path+configs['path']['logs'], separator=",", append=True)

        tb_callback = tf.keras.callbacks.TensorBoard(save_path+configs['path']['tfboard'], update_freq=1)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=save_path+configs['path']['checkpoint'],
                                                    save_weights_only=True,
                                                    monitor='val_acc',
                                                    mode='max')

        history = models_dict['model_{}'.format(i)].fit(training_data,
                                                        epochs=int(configs['train_cfg']['epochs']),
                                                        validation_data=validation_data,
                                                        callbacks=[history_logger, tb_callback, model_checkpoint, lr_schedule])
    
    fusion_model = tf.keras.models.clone_model(models_dict['model_1'])
    fusion_model.compile(optimizer=opt, loss=loss, metrics='acc')
    fusion_model.set_weights(models_dict['model_1'].get_weights())
    fusion_model = update_params(fusion_model, models_dict['model_1'], models_dict['model_1'], models_dict['model_1'], mode='equal')
        
    results = fusion_model.evaluate(test_data)
    print("test loss, test acc:", results)
        
if __name__== '__main__':
    main()