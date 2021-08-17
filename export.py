from models import model_builder, update_params
from utils import read_config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import argparse
import os
import ast

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir',
                        default='/content/drive/MyDrive/src/xray-covid/resnet',
                        help='Path to directory contain 3 backbone model')
    
    parser.add_argument('--config',
                        default='/content/fusion-params/data/config.ini',
                        help='Path to the config file')
    
    parser.add_argument('--image',
                        default='/content/xray_zip/',
                        help='Directory contains images')

    parser.add_argument('--mode',
                        default='equal',
                        help='Mode of the fusion equal|linear|exp')

    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    configs = read_config(args.config)

    num_of_models = len(os.listdir(args.dir))

    result = []
    test = pd.read_csv(configs['path']['test'])
    test_gen = ImageDataGenerator(**ast.literal_eval(configs['preprocess']['test']))
    test_data = test_gen.flow_from_dataframe(dataframe=test,
                                            directory=args.image,
                                            x_col='filename',
                                            y_col='label',
                                            class_mode="categorical",
                                            batch_size=int(configs['train_cfg']['batch_size']),
                                            target_size=ast.literal_eval(configs['train_cfg']['target_size']))
    models_dict = {'model_{}'.format(j) : {'checkpoint':os.path.join(args.dir, 'model_{}'.format(j), 'checkpoint'),'weights': model_builder(configs['train_cfg']['backbone'])} for j in range(1,num_of_models+1)}

    for i in range(1,int(configs['train_cfg']['epochs'])+1):
        
        for j in range(1,num_of_models+1):
            epoch_checkpoint = os.path.join(models_dict['model_{}'.format(j)]['checkpoint'], 'cp-{:04d}.ckpt'.format(j))
            print(epoch_checkpoint)
            models_dict['model_{}'.format(j)]['weights'].load_weights(epoch_checkpoint)
        

        fusion_model = tf.keras.models.clone_model(models_dict['model_1']['weights'])
        fusion_model.compile(optimizer=configs['opt_cfg']['opt'], loss='categorical_crossentropy', metrics='acc')
        fusion_model.set_weights(models_dict['model_1']['weights'].get_weights())
        fusion_model = update_params(fusion_model, models_dict['model_1']['weights'], models_dict['model_2']['weights'], models_dict['model_3']['weights'], mode=args.mode)


        result.append(fusion_model.evaluate(test_data))

        save_path = os.path.join(args.dir, 'fusion', 'checkpoint', 'fusion-{:04d}.ckpt'.format(i))
        fusion_model.save_weights(save_path)
    
    pd.DataFrame(result, columns=['loss', 'acc']).to_csv(os.path.join(args.dir, 'fusion', 'evaluation.csv'), index=False)



if __name__ == "__main__":
    main()