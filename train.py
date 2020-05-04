from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_data_gen(directory):
    train_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

    train_data_gen = train_image_generator.flow_from_directory(directory = directory,
                                                            target_size = (256,256),
                                                            batch_size=64,
                                                            shuffle=True,
                                                            class_mode='sparse',
                                                            subset='training')

    val_data_gen = train_image_generator.flow_from_directory(directory = directory,
                                                            target_size = (256,256),
                                                            batch_size=64,
                                                            shuffle=True,
                                                            class_mode='sparse',
                                                            subset='validation')
    return train_data_gen, val_data_gen

def train_and_save_model(train_data_gen, val_data_gen, fine_tune=True):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(256,256,3),
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(6)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    history = model.fit(
        train_data_gen,
        steps_per_epoch=100,
        epochs=1,
        validation_data=val_data_gen,
        validation_steps=10
    )

    if fine_tune:
        fine_tune_at = 100
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False
        
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
        
        history = model.fit(
            train_data_gen,
            steps_per_epoch=100,
            epochs=5,
            validation_data=val_data_gen,
            validation_steps=10
        )
    
    model.save("/home/raj/Hand-Hygiene-Monitoring-System-using-Gesture-Recognition/saved_model")

def main():
    train_data_gen, val_data_gen = make_data_gen(directory='/home/raj/Downloads/hand_wash/train_data')
    train_and_save_model(train_data_gen=train_data_gen,
                         val_data_gen=val_data_gen,
                         fine_tune=True)

if __name__ == "__main__":
    main()
