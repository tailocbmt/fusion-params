import tensorflow as tf
from models import model_builder
import pandas as pd
import os

df = pd.read_excel('.xlsx')

model = model_builder('resnet')
model.load_weights('cp')

for row in df.iterrows():
    image_path = os.path.join('/content', row[0])
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    resize = tf.image.resize(image, [224, 224])
    normalize = resize / 255.

    print(model.predict(normalize))