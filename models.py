from tensorflow.keras.applications import VGG16, ResNet50V2, InceptionV3
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from math import exp
import numpy as np

def model_builder(name, input_shape=(224, 224, 3), include_top=False, weights='imagenet'):
    if name == 'vgg16':
        base_model = VGG16(input_shape=input_shape, include_top=include_top, weights=weights)
    elif name == 'resnet':
        base_model = ResNet50V2(input_shape=input_shape, include_top=include_top, weights=weights)
    elif name == 'inception':
        base_model = InceptionV3(input_shape=input_shape, include_top=include_top, weights=weights)
    else:
        raise ValueError('Incorrect name (name must be vgg16|resnet|inception')
    
    x = base_model.output
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(256, activation='relu')(x)
    predictions = tfl.Dense(3, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions) 

# model = model_builder('inception')
# model.summary()
# for i in range(len(model.layers)):
#     if model.layers[i].name.endswith('conv'):
#         for j in model.layers[i].get_weights():
#             print(j.shape)

def update_params(fusion_model, model_1, model_2, model_3, mode='equal', decay_rate=None):
    if mode == 'equal':
        W = lambda i: 1/3
    elif mode == 'linear':
        W = lambda i: i/3
    elif mode == 'exp':
        decay_rate = 1
        W = lambda i: exp(-i/decay_rate)
    else:
        raise ValueError("Only support equal|linear|exp mode")
    
    num_layer = len(fusion_model.layers)
    for i in range(num_layer):
        if 'conv2d' in fusion_model.layers[i].name or '_conv' in fusion_model.layers[i].name or 'dense' in fusion_model.layers[i].name:
            params_1 = model_1.layers[i].get_weights() 
            params_2 = model_2.layers[i].get_weights() 
            params_3 = model_3.layers[i].get_weights()
            
            fusion_params = []
            for j in range(len(params_1)):
                fusion = np.multiply(W(1), params_1[j]) + np.multiply(W(2), params_2[j]) + np.multiply(W(3), params_3[j])
                fusion_params.append(fusion)

            fusion_model.layers[i].set_weights(fusion_params)
    
    return fusion_model


