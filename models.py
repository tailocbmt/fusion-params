from tensorflow.keras.applications import VGG16, ResNet50V2, InceptionV3
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model


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

        
