from configparser import ConfigParser


def write_config(names, *args):
    config_obj = ConfigParser()

    for i in range(len(names)):
        config_obj[names[i]] = args[i]
    
    with open('config.ini', 'w') as conf:
        config_obj.write(conf)

def read_config(config_path):
    config_obj = ConfigParser()
    config_obj.read(config_path)
    return config_obj

# preprocess
# preprocess_cfgs = dict(
#     train = dict(rotation_range=15,
#             shear_range=0.1,
#             zoom_range=0.1,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             horizontal_flip=True,
#             fill_mode='nearest',
#             rescale=1/255.),

#             test = dict(rescale=1/255.)
# )

# train_cfg = dict(backbone = 'resnet',
#                 target_size=(224, 224),
#                 batch_size=8,
#                 epochs=50)

# opt_cfg = dict(opt = 'adam',
#             lr=1e-3)

# path = dict(checkpoint = '/checkpoint',
#             logs = 'log.csv',
#             tfboard = './logs')


# names = ['preprocess', 'train_cfg', 'opt_cfg', 'path']
# write_config(names, preprocess_cfgs, train_cfg, opt_cfg, path)