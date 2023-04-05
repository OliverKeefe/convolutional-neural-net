import os
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import keras.utils as image

class Initialization:
    def __init__(self, url, file_name, directory):
        self.url = url
        self.file_name = file_name
        self.directory = directory

class desired_accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nDesired model accuracy has been achieved. Cancelling training...")
            self.model.stop_training = True

def init(url, file_name, directory):
    print(f"Requesting url: {url}...")
    training_data = urllib.request.urlretrieve(url, file_name)
    print(f"Locating file: {file_name}...")
    zip_ref = zipfile.ZipFile(file_name, 'r')
    print(f"Extracting file to{directory}...")
    zip_ref.extractall(directory)
    print("[+] Done.")
    zip_ref.close()
    return 0
        
def generator(datagen, dir):
    generator = datagen.flow_from_directory(
        dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    return generator

def main():
    callbacks = desired_accuracy()
    augmentation = False
    enable_augmentation = ''
    while enable_augmentation == '':
        enable_augmentation = input('Enable image augmentation Y/N? Default [N] $: ')
        if enable_augmentation == 'Y' or enable_augmentation == 'y':
            augmentation = True
        elif augmentation == 'N' or augmentation == 'n':
            augmentation = False
        else:
            print("[!] Error, invalid input.")
            enable_augmentation == ''

    training_url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
    training_file_name = "horse-or-human.zip"
    training_dir = 'horse-or-human/training/'
    validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
    validation_file_name = "validation-horse-or-human.zip"
    validation_dir = 'horse-or-human/validation/'
    training_initialization = Initialization(
        training_url,
        training_file_name,
        training_dir
    )
    validation_initialization = Initialization(
        validation_url,
        validation_file_name,
        validation_dir
        )

    init(training_initialization.url, training_initialization.file_name, training_initialization.directory)
    init(validation_initialization.url, validation_initialization.file_name, validation_initialization.directory)
    if augmentation == True:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        # Rescale all images to 1./255
        train_datagen = ImageDataGenerator(rescale=1/255)
        
    # Rescale all images to 1./255
    validation_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = generator(train_datagen, training_initialization.directory)
    validation_generator = generator(train_datagen, validation_initialization.directory)

    model = tf.keras.models.Sequential([
        # Input shape is 300x300 with 3 bytes colour
        # First convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convolution 2
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # 3 ...
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # 4 ...
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # 5 ...
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten into DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Output neuron, 0 for ('horses') class and 1 for ('humans') class 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        validation_data=validation_generator)

    # Test the model on images of horses and humans.
    directory = input('Specify path to test images. $: ')
    for filename in os.scandir(directory):
        #fp = os.path.join(directory, filename)
        if filename.is_file():
            image_path = filename.path
            img = image.load_img(image_path, target_size=(300,300))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            image_tensor = np.vstack([x])
            classes = model.predict(image_tensor)
            if classes[0]>0.5:
                print(f"The image file {image_path} is an image of a human.")
            else:
                print(f"The image file {image_path} is an image of a horse.")

if __name__ == "__main__":
    main()