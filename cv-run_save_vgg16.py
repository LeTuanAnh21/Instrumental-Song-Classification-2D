import argparse

parser = argparse.ArgumentParser(description="run",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--image_len", default=32, type=int, help="IMAGE_LEN")
parser.add_argument("-b", "--batch_size", default=32, type=int,  help="BATCH_SIZE")
parser.add_argument("-t", "--folder_train",default='train',  help="folder_train")
parser.add_argument("-r", "--folder_test", default='test',  help="folder test")
parser.add_argument("-e", "--num_epoch", default=3, type=int, help="epoch")
parser.add_argument("-p", "--patience_epoch", default=3, type=int, help="epoch patience")
parser.add_argument("-c", "--num_class", default=2, type=int, help="number of classes")
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float, help="number of classes")
parser.add_argument("-k", "--nkfold", default=0, type=int, help="kfold ")
parser.add_argument("-f", "--folder", default='save_models', help="folder to save")
parser.add_argument("--seed", default=1,type=int, help="folder to save")
parser.add_argument("--mod", default='vgg', help="model")
parser.add_argument("--nfilter", default=64, type=int, help="number of filter (for cnn3)")

args = parser.parse_args()
config = vars(args)
print(config)



# setup cac bien
# IMAGE_LEN = 32
# BATCH_SIZE = 32
# folder_train = 'data-train'
# folder_test = 'data-test'
# num_epoch = 3

# IMAGE_LEN = config.image_len
# BATCH_SIZE = config.batch_size
# folder_train = config.folder_train
# folder_test = config.folder_test
# num_epoch = config.num_epoch

IMAGE_LEN = args.image_len
BATCH_SIZE = args.batch_size
folder_train = args.folder_train
folder_test = args.folder_test
path_2 = folder_train + folder_test
path_named = path_2.replace("/", ".")
num_epoch = args.num_epoch
num_class = args.num_class
learning_rate=args.learning_rate
patience_epoch=args.patience_epoch
nkfold=args.nkfold
folder_save = args.folder
seed_v = args.seed
mod = args.mod

# IMAGE_LEN = args['image_len']
# BATCH_SIZE = args['batch_size']
# folder_train = args['folder_train']
# folder_test = args['folder_test']
# num_epoch = args['num_epoch']


import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
#khai báo frame work tensorflow
import tensorflow as tf
import random
#import keras từ frame work tensorflow
from tensorflow import keras
import tensorflow.keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.layers import MaxPooling2D


# gan cac seed de co kq thuc nghiem giong

tf.random.set_seed(seed_v)
random.seed(seed_v)
np.random.seed(seed_v)

import time
start = time.time()


## tim tap tin de tranh bi trung
import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result



#512x512 size của ảnh

IMAGE_SIZE = (IMAGE_LEN, IMAGE_LEN)
#chia dữ liệu huấn luyện/ kiểm thử thành từng batch

#tiền sử lý ảnh
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_train,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)
#tiền sử lý ảnh
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_test,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)


from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras.utils import to_categorical
from sklearn import preprocessing
#from exploit_pred import *



def model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE):


    img_input = Input(shape=(image_size,image_size,3))

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


if nkfold in [0,1]:

    if mod == 'vgg':
        name_model = ''
    else:
        name_model = mod + '_'
    name_saved= name_model + path_named + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold) + 'nfilter'+str(args.nfilter) 
    print ('name_saved='+name_saved)
    n_files = find(name_saved + '*.json',folder_save )
    if len(n_files)>0:
        #print('name_saved'+name_saved) #dung thuc nghiem neu lam roi
        print('thuc nghiem '+name_saved+' da lam roi10!')
        exit()

    if mod == 'vgg':
        model = Sequential()
        # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        #opt = Adam(lr=0.0001)
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('vggvggvggvgg')
        model.summary()
   
    elif mod == 'fc':        
        # Build the model
        model = Sequential()       
        model.add(InputLayer(input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))  
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))

        # Compile the model
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        
        from tensorflow.keras.callbacks import EarlyStopping
        # Define Early Stopping callback
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        
        
        print('fc')
        model.summary()
        
   
    
    elif mod =='fc_aug':
        from tensorflow.keras.layers import InputLayer, Dense, Flatten
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        import keras

        # Định nghĩa các hằng số
        IMAGE_LEN = 32
        num_class = 100
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        
         # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,          
            # tỷ lệ lại giá trị các pixel từ [0, 255] về [0, 1], để giảm thiểu độ lớn của các giá trị đầu vào và tăng tốc quá trình huấn luyện
            rotation_range=15,
            # rotation_range=10: xoay ảnh một góc ngẫu nhiên trong khoảng từ -15 đến 15 độ
            width_shift_range=0.1,
            # width_shift_range=0.1: dịch chuyển ảnh ngang theo chiều ngang một khoảng từ -10% đến 10% chiều rộng ảnh
            height_shift_range=0.1,
            # height_shift_range=0.1: dịch chuyển ảnh dọc theo chiều dọc một khoảng từ -10% đến 10% chiều cao ảnh
            shear_range=0.1,
            # shear_range=0.1: cắt ảnh một góc ngẫu nhiên trong khoảng từ -10 đến 10 độ
            zoom_range=0.1,
            # zoom_range=0.1: phóng to hoặc thu nhỏ ảnh với tỷ lệ ngẫu nhiên trong khoảng từ 0.9 đến 1.1 lần kích thước ảnh gốc
            horizontal_flip=True,
            # horizontal_flip=True: lật ngang ảnh
            vertical_flip=False,
            # vertical_flip=False: không lật dọc ảnh
            fill_mode='nearest'
            #  điền các pixel bị thiếu bằng pixel gần nhất
            )

        datagen_test = ImageDataGenerator(
            rescale=1./255
            )

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        learning_rates = [0.001, 0.0001, 0.00001]

        for lr in learning_rates:
            print(f"Training with learning rate: {lr}")
        model = Sequential()
        model.add(InputLayer(input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        
        
         # Compile the model
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Huấn luyện model với dữ liệu tăng cường
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )
            # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        print('fc_aug')
        model.summary()

    elif  mod == 'cnn1':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        #opt = Adam(lr=0.0001)
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn1')
        model.summary()
        
    elif  mod == 'cnn5':
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 32
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 100
        NUM_CLASSES = 100
        
        datagen_train = ImageDataGenerator(
            rotation_range=10, # xoay ảnh trong khoảng 10 độ
            width_shift_range=0.1, # tịnh tiến ảnh theo chiều ngang
            height_shift_range=0.1, # tịnh tiến ảnh theo chiều dọc
            shear_range=0.1, # cắt ảnh
            zoom_range=0.1, # phóng to hoặc thu nhỏ ảnh
            horizontal_flip=True, # lật ngang ảnh
            brightness_range=[0.7, 1.3], # tăng hoặc giảm độ sáng của ảnh
            fill_mode='nearest'
        )
    
        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        datagen_test = ImageDataGenerator(
            rotation_range=10, # xoay ảnh trong khoảng 10 độ
            width_shift_range=0.1, # tịnh tiến ảnh theo chiều ngang
            height_shift_range=0.1, # tịnh tiến ảnh theo chiều dọc
            shear_range=0.1, # cắt ảnh
            zoom_range=0.1, # phóng to hoặc thu nhỏ ảnh
            horizontal_flip=True, # lật ngang ảnh
            brightness_range=[0.7, 1.3], # tăng hoặc giảm độ sáng của ảnh
            fill_mode='nearest'
        )
        
        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN, IMAGE_LEN, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(units=num_class, activation="softmax"))
        
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn5')
        model.summary()
    elif mod =='cnn110':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn110')
        model.summary()
    elif mod =='cnn001_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn001_k')
        model.summary()
        
        
    elif mod =='cnn001_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
        
        # Print model summary
        print('cnn001_m')
        model.summary()
        
        
        
        
    elif mod =='cnn001':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        # Print model summary
        print('cnn001')
        model.summary()
        
        
        
    elif mod =='cnn002':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model 
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))
        

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn002')
        model.summary()
        
    elif mod =='cnn002_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        # Print model summary
        print('cnn002_k')
        model.summary()
    
    
    elif mod =='cnn002_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        # Print model summary
        print('cnn002_m')
        model.summary()
        
    
    
    elif mod =='cnn003':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile the model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn003')
        model.summary()
        
    
    
    elif mod =='cnn003_k':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))            
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn003_k')
        model.summary()
        
        
    elif mod =='cnn003_m':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Load and preprocess data
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            
            # Build the model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))    
            model.add(MaxPool2D(pool_size=(2, 2)))        
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile the model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Train the model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                validation_data=test_generator,
                callbacks=[early_stop, reduce_lr]
            )
            
            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
       

        # Print model summary
        print('cnn003_m')
        model.summary()
        
        
        
        
        
    elif mod =='cnn111':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100
        
        # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,          
            # tỷ lệ lại giá trị các pixel từ [0, 255] về [0, 1], để giảm thiểu độ lớn của các giá trị đầu vào và tăng tốc quá trình huấn luyện
            rotation_range=15,
            # rotation_range=10: xoay ảnh một góc ngẫu nhiên trong khoảng từ -15 đến 15 độ
            width_shift_range=0.1,
            # width_shift_range=0.1: dịch chuyển ảnh ngang theo chiều ngang một khoảng từ -10% đến 10% chiều rộng ảnh
            height_shift_range=0.1,
            # height_shift_range=0.1: dịch chuyển ảnh dọc theo chiều dọc một khoảng từ -10% đến 10% chiều cao ảnh
            shear_range=0.1,
            # shear_range=0.1: cắt ảnh một góc ngẫu nhiên trong khoảng từ -10 đến 10 độ
            zoom_range=0.1,
            # zoom_range=0.1: phóng to hoặc thu nhỏ ảnh với tỷ lệ ngẫu nhiên trong khoảng từ 0.9 đến 1.1 lần kích thước ảnh gốc
            horizontal_flip=True,
            # horizontal_flip=True: lật ngang ảnh
            vertical_flip=False,
            # vertical_flip=False: không lật dọc ảnh
            fill_mode='nearest'
            #  điền các pixel bị thiếu bằng pixel gần nhất
            )

        datagen_test = ImageDataGenerator(
            rescale=1./255
            )

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Build the model
        learning_rates = [0.001, 0.0001, 0.00001]

        for lr in learning_rates:
            print(f"Training with learning rate: {lr}")
        model = Sequential()
        
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))
        
        # Compile the model
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            validation_data=test_generator,
            callbacks=[early_stop, reduce_lr]
        )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        # Print model summary
        print('cnn111')
        model.summary()
        
    elif mod == 'cnn4':
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32        # IMAGE_LEN: kích thước ảnh đầu vào được chuẩn hóa thành kích thước này
        BATCH_SIZE = 64        # BATCH_SIZE: số lượng mẫu ảnh được sử dụng cho mỗi lần huấn luyện.
        LEARNING_RATE = 0.0001        # LEARNING_RATE: tỷ lệ học (learning rate) cho quá trình tối ưu hóa mô hình.
        NUM_EPOCHS = 500        # NUM_EPOCHS: số lần huấn luyện (epoch) cho mô hình.
        NUM_CLASSES = 100        # NUM_CLASSES: số lượng lớp (classes) cần phân loại.
        # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,          
            # tỷ lệ lại giá trị các pixel từ [0, 255] về [0, 1], để giảm thiểu độ lớn của các giá trị đầu vào và tăng tốc quá trình huấn luyện
            rotation_range=15,
            # rotation_range=10: xoay ảnh một góc ngẫu nhiên trong khoảng từ -15 đến 15 độ
            width_shift_range=0.1,
            # width_shift_range=0.1: dịch chuyển ảnh ngang theo chiều ngang một khoảng từ -10% đến 10% chiều rộng ảnh
            height_shift_range=0.1,
            # height_shift_range=0.1: dịch chuyển ảnh dọc theo chiều dọc một khoảng từ -10% đến 10% chiều cao ảnh
            shear_range=0.1,
            # shear_range=0.1: cắt ảnh một góc ngẫu nhiên trong khoảng từ -10 đến 10 độ
            zoom_range=0.1,
            # zoom_range=0.1: phóng to hoặc thu nhỏ ảnh với tỷ lệ ngẫu nhiên trong khoảng từ 0.9 đến 1.1 lần kích thước ảnh gốc
            horizontal_flip=True,
            # horizontal_flip=True: lật ngang ảnh
            vertical_flip=False,
            # vertical_flip=False: không lật dọc ảnh
            fill_mode='nearest'
            #  điền các pixel bị thiếu bằng pixel gần nhất
            )

        datagen_test = ImageDataGenerator(
            rescale=1./255
            )

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            model = Sequential()
                
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3), kernel_regularizer=regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3), kernel_regularizer=regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
                
            model.add(Flatten())
            model.add(Dense(units=256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile model
            opt = Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
             # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
            
            # Fit model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                steps_per_epoch=len(train_indexes) // BATCH_SIZE,
                validation_data=test_generator,
                validation_steps=len(val_indexes) // BATCH_SIZE,
                callbacks=[early_stop, reduce_lr]
            )

            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    
            print('cnn4')
            model.summary()
            
    elif mod =='cnn10':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        import numpy as np
        import random

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100

        # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        )

        datagen_test = ImageDataGenerator(
            rescale=1./255
        )

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        from tensorflow.keras import regularizers

        # Define model
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3), kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3), kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile model
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        # Fit model
        history = model.fit(
            train_generator,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_generator),
            validation_data=test_generator,
            validation_steps=len(test_generator),
            callbacks=[early_stop, reduce_lr]
        )
        print('cnn10')
        model.summary()
    
    elif mod =='cnn6':
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.001
        NUM_EPOCHS = 500
        NUM_CLASSES = 100

        # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest')

        datagen_test = ImageDataGenerator(rescale=1./255)

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Define model
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(units=NUM_CLASSES, activation='softmax'))

        # Compile model
        opt = Adam(lr=LEARNING_RATE)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print('cnn6')
        model.summary()
        
    elif  mod == 'cnn2':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn2')
        model.summary()
    elif mod == 'cnn7':
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Define constants
        IMAGE_LEN = 32
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 100
        NUM_CLASSES = 100

        # Define data generators
        datagen_train = ImageDataGenerator(
            rescale=1./255,       # Tỉ lệ giá trị pixel được chia để đưa về khoảng giá trị từ 0 đến 1.
            rotation_range=15,    #   Góc xoay ngẫu nhiên trong khoảng từ -15 đến 15 độ.
            width_shift_range=0.1,    # Tịnh tiến ngang ngẫu nhiên trong khoảng từ -10% đến 10% của chiều rộng của ảnh.
            height_shift_range=0.1,    # Tịnh tiến dọc ngẫu nhiên trong khoảng từ -10% đến 10% của chiều cao của ảnh.
            shear_range=0.1,    # Biến dạng ngẫu nhiên bằng cách cắt ảnh theo một góc ngẫu nhiên trong khoảng từ -10 đến 10 độ.
            zoom_range=0.1,    # Phóng to hoặc thu nhỏ ngẫu nhiên trong khoảng từ 0.9 đến 1.1 lần kích thước ban đầu của ảnh.
            horizontal_flip=True,    # Lật ngang ngẫu nhiên ảnh.
            vertical_flip=False,    # Không lật dọc ảnh.
            fill_mode='nearest'    #  Điền giá trị cho các pixel bị thiếu sau khi thực hiện các phép biến đổi. Ở đây, giá trị được điền là giá trị của pixel gần nhất.
)

        datagen_test = ImageDataGenerator(rescale=1./255)

        train_dir = 'train'
        train_generator = datagen_train.flow_from_directory(
            train_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        test_dir = 'test'
        test_generator = datagen_test.flow_from_directory(
            test_dir,
            target_size=(IMAGE_LEN, IMAGE_LEN),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        from tensorflow.keras import regularizers
        # Define model
        from sklearn.model_selection import StratifiedKFold
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_indexes, val_indexes) in enumerate(kfold.split(train_generator.filenames, train_generator.classes)):
    
            # Define model
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_LEN, IMAGE_LEN, 3)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))
            
            model.add(Flatten())
            model.add(Dense(units=256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            model.add(Dense(units=NUM_CLASSES, activation='softmax'))

            # Compile model
            opt = Adam(lr=LEARNING_RATE)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
            # Define callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

            # Fit model
            history = model.fit(
                train_generator,
                epochs=NUM_EPOCHS,
                steps_per_epoch=len(train_indexes) // BATCH_SIZE,
                validation_data=test_generator,
                validation_steps=len(val_indexes) // BATCH_SIZE,
                callbacks=[early_stop, reduce_lr]
            )

            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

            # Print results
            print(f"Fold {fold + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    
        print('cnn7')
        model.summary()
        
    elif  mod == 'cnn3':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=args.nfilter,kernel_size=(3,3),padding="same", activation="relu"))
     
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn3')
        model.summary()
    elif  mod == 'efficient':

        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        from tensorflow.keras.applications import EfficientNetB0

        import tensorflow as tf

        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            print("Device:", tpu.master())
            strategy = tf.distribute.TPUStrategy(tpu)
        except ValueError:
            print("Not connected to a TPU runtime. Using CPU/GPU strategy")
            strategy = tf.distribute.MirroredStrategy()


        from tensorflow.keras.models import Sequential
        from tensorflow.keras import layers

        # img_augmentation = Sequential(
        #     [
        #         layers.RandomCrop(IMAGE_LEN,IMAGE_LEN)
                
        #     ],
        #     name="img_augmentation",
        # )

        with strategy.scope():
            inputs = layers.Input(shape=(IMAGE_LEN, IMAGE_LEN, 3))
            #x = img_augmentation(inputs)
            outputs = EfficientNetB0(include_top=True, weights=None, classes=num_class)(inputs)

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

        model.summary()

        #epochs = 40  # @param {type: "slider", min:10, max:100}
        #hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   
    history = model.fit(train_ds, validation_data=test_ds, epochs=num_epoch, verbose=1,callbacks=[callback])
    #print('targets[test].shape')
    #print(targets[test].shape)
    #print(targets)
    #history = model.fit(data_full[train], targets[train], validation_data=(data_full[test], targets[test]), epochs=num_epoch, verbose=1,callbacks=[callback])
    from datetime import datetime
    now=datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    #print("date and time:",date_time)

    # name with hyperparameters used
    name_saved=  name_saved +'_' + str(date_time) #+'_k'+str(i)

    #i=i+1
    #model.save('VGG16_new.h5')
    print('Model Saved!')

    model.save(folder_save+'/'+ name_saved +'.h5')
    model_json = model.to_json()
    with open( folder_save + '/' + name_saved +".json", "w") as json_file:
        json_file.write(model_json)

        # luu lai log
    end = time.time()
    print("time run: ", end - start)


    # from sklearn.metrics import matthews_corrcoef
    # test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    # y_pred = model.predict(test_generator).argmax(axis=1)
    # y_true = test_generator.classes
    # mcc = matthews_corrcoef(y_true, y_pred)
    # print(f"Test_loss={test_loss:.4f}, Test_acc={test_acc:.4f}, MCC: {mcc:.4f}")


    ep_arr=range(1, len(history.history['accuracy'])+1, 1)
    idx = len(history.history['accuracy'])-1 #index of mang
    train_acc = history.history['accuracy']
    val_acc= history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']



    # Thêm MCC vào danh sách thông tin
    # mcc_list = [mcc] * len(train_acc)




    title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])      
    # res=(ep_arr,train_acc,val_acc, train_loss,val_loss)
    # res=np.transpose(res)
    # combined_res=np.array(np.vstack((title_cols,res)))
    res = (ep_arr, train_acc, val_acc, train_loss, val_loss)
    res = np.transpose(res)
    combined_res = np.array(np.vstack((title_cols, res)))
    
    log_name1 = name_saved +'s1'
    np.savetxt(folder_save + '/'+log_name1 +".txt", combined_res, fmt="%s",delimiter="\t") 

    #print('val_acc[len(history.history[accuracy])]' ) 
    #print(val_acc[len(history.history['accuracy'])-1])
    
    
    #log 2 luu lai tham so va cac thong tin ve sample
    log_name2 = name_saved +'s2_'  + 'time'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3))
    #np.savetxt('save_models/'+log_name+"log2.txt", args, fmt="%s",delimiter="\t")
    with open(folder_save + '/'+log_name2+ ".txt", 'w') as f:
        f.write(str(args))
    title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  
    
    
    train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))
    test_labels = np.concatenate(list(test_ds.map(lambda x, y:y)))
    
    res=(len(train_labels),len(test_labels),train_acc[idx],train_loss[idx],val_acc[idx],val_loss[idx])
    res=np.transpose(res)
    combined_res=np.array(np.vstack((title_cols,res)))

    with open(folder_save+ '/'+log_name2+ ".txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f, combined_res, fmt="%s",delimiter="\t")     

else:
    # lay label and data
    train_images = np.concatenate(list(train_ds.map(lambda x, y:x)))
    train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))

    # merge 2 data: train/test lai
    test_images = np.concatenate(list(test_ds.map(lambda x, y:x)))
    test_labels = np.concatenate(list(test_ds.map(lambda x, y:y)))

    data_full = np.concatenate((train_images, test_images), axis=0)
    targets = np.concatenate((train_labels, test_labels), axis=0)
    print('datadatadatadatadatadata')
    print(data_full.shape)
    print(targets.shape)
    #targets=targets.flatten()
    #print(targets)
    from sklearn.model_selection import StratifiedKFold, KFold
    #skf = StratifiedKFold(n_splits=nkfold)
    skf = KFold(n_splits=nkfold)
    skf.get_n_splits(data_full, test_labels)
    # xoa cac bien
    del train_ds
    del test_ds
    del test_images
    del test_labels

    i=1

    for train, test in skf.split(data_full, targets):

        if mod == 'vgg':
            name_model = ''
        else:
            name_model = mod + '_'

        name_saved=  name_model + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold)+ '_'+ str(i)
        i=i+1
        n_files = find(name_saved + '*.json',folder_save )
        if len(n_files)>0:
            print('thuc nghiem '+name_saved+' da lam roi!')
            continue #di den k tiep theo
        
        

        # model = Sequential()
        # # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Flatten())
        # model.add(Dense(units=4096,activation="relu"))
        # model.add(Dense(units=4096,activation="relu"))
        # model.add(Dense(units=num_class, activation="softmax"))
        # from tensorflow.keras.optimizers import Adam
        # #opt = Adam(lr=0.0001)
        # opt = Adam(lr=learning_rate)
        # model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # model.summary()


        if mod == 'vgg':
            model = Sequential()
            # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Flatten())
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=num_class, activation="softmax"))
            from tensorflow.keras.optimizers import Adam
            #opt = Adam(lr=0.0001)
            opt = Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            print('vggvggvggvgg')
            model.summary()
    
        elif mod == 'ef':
            print('efefefefefef')
            model = model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE)
            model.summary()
            
            
        elif mod == 'vgg5':
            model = Sequential()
            # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Flatten())
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=num_class, activation="softmax"))
            from tensorflow.keras.optimizers import Adam
            #opt = Adam(lr=0.0001)
            opt = Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            print('vggvggvggvgg')
            model.summary()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   

        history = model.fit(data_full[train], targets[train], validation_data=(data_full[test], targets[test]), epochs=num_epoch, verbose=1,callbacks=[callback])

        from datetime import datetime
        now=datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        print('Model Saved!')

        model.save(folder_save+'/'+ name_saved+ '_'+ str(date_time) + '.h5')
        model_json = model.to_json()
        with open( folder_save + '/' + name_saved + '_'+ str(date_time) + ".json", "w") as json_file:
            json_file.write(model_json)


            # luu lai log
        end = time.time()
        print("time run: ", end - start)

        ep_arr=range(1, len(history.history['accuracy'])+1, 1)
        idx = len(history.history['accuracy'])-1 #index of mang
        train_acc = history.history['accuracy']
        val_acc= history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])  
        res=(ep_arr,train_acc,val_acc, train_loss,val_loss)
        res=np.transpose(res)
        combined_res=np.array(np.vstack((title_cols,res)))

        
       
        log_name1 = name_saved +'s1' + '_'+ str(date_time)
        np.savetxt(folder_save + '/'+log_name1+ ".txt", combined_res, fmt="%s",delimiter="\t") 
        
        #log 2 luu lai tham so va cac thong tin ve sample
        log_name2 = name_saved +'s2' + '_'+ str(date_time) + 't'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3)) 
        #np.savetxt('save_models/'+log_name+"log2.txt", args, fmt="%s",delimiter="\t")
        with open(folder_save + '/'+log_name2+".txt", 'w') as f:
            f.write(str(args))
        title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  
        res=(len(targets[train]),len(targets[test]),train_acc[idx],train_loss[idx],val_acc[idx],val_loss[idx])
        res=np.transpose(res)
        combined_res=np.array(np.vstack((title_cols,res)))

        with open(folder_save+ '/'+log_name2 + ".txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, combined_res, fmt="%s",delimiter="\t")     