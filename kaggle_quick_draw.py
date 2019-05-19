# Seongho Jin
# kaggle quick draw competition

# drawing data를 이미지로 바꿈
# 이미지 전처리
# 좌우대칭, 가우시안 블러 적용 (그림 주변으로 다양한 화소값을 가지게 하기 위해)
# 입력 이미지는 0~1 사이의 값으로 적용

# MobileNetV2 사용하여 학습 (선정이유: weight는 적지만, 높은 성능을 보이는 모델로 학습하기 적합함)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import glob
from matplotlib import pyplot as plt
import cv2
import keras
import random

class_names = sorted([name[:-4] for name in os.listdir('../input/quickdraw-doodle-recognition/train_simplified/')]) # 340 classes
class_dic = {}
for i in range(len(class_names)):
    class_dic[class_names[i]] = i
class_paths = sorted(glob.glob('../input/quickdraw-doodle-recognition/train_simplified/' + "*"))

from tensorflow.keras.utils import to_categorical
from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0.0, 3.0))
])

BATCH_SIZE = 680
N_CLASS = 340
W, H = 224, 224

LINE_WIDTH = 3
train_path = '../input/mydata1/train3401000.csv'
val_path= '../input/mydata1/val3401000.csv'


def load_random_samples(file, sample_size):
    num_lines = sum(1 for l in open(file))

    skip_idx = random.sample(range(1, num_lines), num_lines - (sample_size + 1))
    samples = pd.read_csv(file, usecols=['drawing',  'word'], 
                       skiprows=skip_idx)
    return samples

def drawing_to_img(drawing,  line_width):
    if type(drawing) == str:
        drawing = eval(drawing)
        
    img = np.zeros((W, H), np.uint8)
    for step in drawing :
        x, y = step
        for i in range(len(x)-1):
            cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), 255, line_width) # line으로 연결
    return img


def img_preprocess(img_before):
    img = np.reshape(img_before, (1, W, H))
    img = seq.augment_images(images=img)
    img = np.array(img).astype('float32') / 255
    
    dummy = np.zeros((W, H, 3))
    for i in range(3):
        dummy[:,:,i] = img[0,:,:]
    return dummy

def generator_random(train_path, batch_size=680):
    
    while True:
        df_batch = load_random_samples(train_path, batch_size)

        batch_img = []
        batch_label = []
        for i in range(batch_size):
            img = drawing_to_img(df_batch.drawing[i], LINE_WIDTH)
            img = img_preprocess(img)

            label = class_dic[df_batch.word[i]]
            label = to_categorical(label, N_CLASS)
            
            batch_img.append(img)
            batch_label.append(label)
            
        yield [np.array(batch_img), np.array(batch_label)]

train_gen = generator_random(train_path, BATCH_SIZE)
val_gen = generator_random(val_path, BATCH_SIZE)
x, y = next(train_gen)
print(x.shape, y.shape)



from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import GlobalAvgPool2D
from keras.layers import Dense

# mobile_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False, classes=N_CLASS)
mobile_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, classes=N_CLASS)
model = Sequential()
model.add(mobile_model)
model.add(GlobalAvgPool2D())
model.add(Dense(N_CLASS, activation='softmax', use_bias=True))
model.summary()



from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint
from math import ceil

model.compile(optimizer=Adam(lr=1e-4), loss=categorical_crossentropy, metrics=[categorical_accuracy])

weight_save = 'mobileNet_01.hdf5'
callback = ModelCheckpoint(weight_save,
                           monitor='val_acc',
                           mode='max',
                           save_best_only=True,
                           save_weights_only=True,
                           verbose=1)

n_train = sum(1 for l in open(train_path)) - 1
n_val = sum(1 for l in open(val_path)) - 1
N_EPOCH = 100

history = model.fit_generator(train_gen, 
                                steps_per_epoch=ceil(n_train/BATCH_SIZE),
                                epochs=N_EPOCH,
                                validation_data=val_gen,
                                validation_steps=ceil(n_val/BATCH_SIZE))
                                
                         
