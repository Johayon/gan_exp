import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import pickle as pickle

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import initializers

from tqdm import tqdm_notebook as tqdm


def pickle_load(fpath):
    with open(fpath, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model
        
    
def pickle_dump(model, path):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)

        
class BaseGanModel:
    
    def __init__(self, noise_dimension=100, output_size=784, generator_activation='tanh',
                 discriminator_dropout=0.3, discriminant_likeness=0.9):
        self.noise_dimension = noise_dimension
        self.output_size = output_size
        self.generator_activation = generator_activation
        self.discriminator_dropout = discriminator_dropout
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.generator = None
        self.discriminator = None
        self.gan_model = None
        self.discriminator_losses = None
        self.generator_losses = None
        self.discriminant_likeness = discriminant_likeness
        
    def init_model(self):
        self.generator = self._create_generator(self.generator_activation)
        self.discriminator = self._create_discriminator(self.discriminator_dropout)
        self.gan_model = self._combine_generator_discriminator()
        self.discriminator_losses = []
        self.generator_losses = []
        
    def _create_generator(self, activation):
        # basic generator model
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.noise_dimension, 
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(self.output_size, activation=activation))
        generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return generator
    
    def _create_discriminator(self, dropout):
        # discriminator model
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=self.output_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(dropout))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return discriminator
    
    def _combine_generator_discriminator(self):
        # Combined network
        self.discriminator.trainable = False
        ganInput = Input(shape=(self.noise_dimension,))
        x = self.generator(ganInput)
        ganOutput = self.discriminator(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return gan
    
    def load_model(self, generator_pickle_path, discriminator_pickle_path):
        generator = pickle_load(generator_pickle_path)
        discriminator = pickle_load(discriminator_pickle_path)
        self.edit_model(generator, discriminator)
        
    def edit_model(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.gan_model = self._combine_generator_discriminator()
        self.discriminator_losses = []
        self.generator_losses = []
    
    def save_model(self, generator_pickle_path, discriminator_pickle_path):
        pickle_dump(self.generator, generator_pickle_path)
        pickle_dump(self.discriminator, discriminator_pickle_path)
        
    def generate(self, batch_size):
        # Generate fake samples using generator
        noise = np.random.normal(0, 1, size=[batch_size, self.noise_dimension])
        return self.generator.predict(noise)
        
    def _fit_discriminator(self, X_train_batch):
        # train the discriminator 
        
        batch_size = X_train_batch.shape[0]
        X_gen = self.generate(batch_size)
        X = np.concatenate([X_train_batch, X_gen])
        
        # Labels for generated and real data
        y_discriminant = np.zeros(2 * batch_size)
        y_discriminant[:batch_size] = self.discriminant_likeness

        self.discriminator.trainable = True
        dis_loss = self.discriminator.train_on_batch(X, y_discriminant)
        self.discriminator.trainable = False
        return dis_loss
        
    def _fit_generator(self, batch_size):
        # train the generator
        noise = np.random.normal(0, 1, size=[batch_size, self.noise_dimension])
        y_generator = np.ones(batch_size)
        gen_loss = self.gan_model.train_on_batch(noise, y_generator)
        return gen_loss
    
    def fit(self, X_train, y_train, epochs=100, batch_size=128):
        if self.generator is None or self.discriminator is None or self.gan_model is None:
            raise ValueError("model is not intialized correctly, please call init_model")
            

        for _ in tqdm(range(1, epochs+1), leave=False):
            # train discriminator and generator
            X_train_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            dis_loss = self._fit_discriminator(X_train_batch)
            gen_loss = self._fit_generator(batch_size)
            
            # store loss
            self.discriminator_losses.append(dis_loss)
            self.generator_losses.append(gen_loss)
    

class MnistGanModel(BaseGanModel):
    
    def _create_generator(self, activation):
        # generator model
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.noise_dimension, 
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(self.output_size, activation=activation))
        generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return generator
    
    def _create_discriminator(self, dropout):
        # discriminator model
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=self.output_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(dropout))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(dropout))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(dropout))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return discriminator
