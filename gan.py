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
    """ This class is a basic Gan model
    
    it is possible to change the discriminant and generator directly.
    
    label_noise::float apply noise during the training of the discriminator to avoid overfitting 
                       allow the generator to have more diversity.
    
    separate_fit::boolean fit the discriminator on the true and fake observation separatly.
    """
    
    def __init__(self, label_noise=0, separate_fit=False):
        self.label_noise = label_noise
        self.separate_fit = separate_fit
        self.generator = None
        self.discriminator = None
        self.gan_model = None
        self.discriminator_losses = None
        self.generator_losses = None
        
    def init_model(self, noise_dimension=100, output_size=784):
        """ init a Gan model using the function _create_generator and _create_discriminant
        
        noise_dimension::int input dimension of the generator
        
        output_size::int output of the generator and input of the discriminator
        """
        self.generator = self._create_generator(noise_dimension, output_size)
        self.discriminator = self._create_discriminator(output_size)
        self.gan_model = self._combine_generator_discriminator()
        self.discriminator_losses = []
        self.generator_losses = []
        return self
        
    def _create_generator(self, noise_dimension, output_size):
        # basic generator model
        generator = Sequential()
        generator.add(Dense(256, input_dim=noise_dimension, 
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(output_size, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return generator
    
    def _create_discriminator(self, output_size):
        # discriminator model
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=output_size, 
                                kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return discriminator
    
    def _combine_generator_discriminator(self):
        # Combined network
        self.discriminator.trainable = False
        ganInput = Input(shape=(self.generator.get_input_at(0).get_shape()[1].value,))
        x = self.generator(ganInput)
        ganOutput = self.discriminator(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return gan
    
    def load_model(self, generator_pickle_path, discriminator_pickle_path):
        """ load a model using pickle of the generator and discriminator
        
        generator_pickle_path::String  path of the generator pickle
        
        discriminator_pickle_path::String path of the discriminator pickle
        """
        generator = pickle_load(generator_pickle_path)
        discriminator = pickle_load(discriminator_pickle_path)
        self.edit_model(generator, discriminator)
        
    def edit_model(self, generator, discriminator):
        """ edit a model with a new generator and discriminator """
        self.generator = generator
        self.discriminator = discriminator
        self.gan_model = self._combine_generator_discriminator()
        self.discriminator_losses = []
        self.generator_losses = []
    
    def save_model(self, generator_pickle_path, discriminator_pickle_path):
        """ save a model using pickle of the generator and discriminator
        
        generator_pickle_path::String  path of the generator pickle
        
        discriminator_pickle_path::String path of the discriminator pickle
        """
        pickle_dump(self.generator, generator_pickle_path)
        pickle_dump(self.discriminator, discriminator_pickle_path)
        
    def generate(self, batch_size):
        """ Generate fake samples using generator
        
        batch_size::int number of observation to generate
        """
        noise = np.random.normal(0, 1, size=[batch_size, 
                                             self.generator.get_input_at(0).get_shape()[1].value])
        return self.generator.predict(noise)
        
    def _fit_discriminator(self, X_train_batch):
        # train the discriminator 
        
        batch_size = X_train_batch.shape[0]
        X_gen = self.generate(batch_size)
        X = np.concatenate([X_train_batch, X_gen])
        
        # Labels for generated and real data
        y_discriminant = np.zeros(2 * batch_size)
        y_discriminant[:batch_size] = 1
        y_discriminant = np.maximum(0, y_discriminant + self.label_noise * 
                                    np.random.uniform(-1, 1, 2*batch_size))

        self.discriminator.trainable = True
        if self.separate_fit:
            dis_loss = self.discriminator.train_on_batch(X[:batch_size], y_discriminant[:batch_size])
            dis_loss = self.discriminator.train_on_batch(X[batch_size:], y_discriminant[batch_size:])
        else:
            dis_loss = self.discriminator.train_on_batch(X, y_discriminant)
        self.discriminator.trainable = False
        return dis_loss
        
    def _fit_generator(self, batch_size):
        # train the generator
        noise = np.random.normal(0, 1, size=[batch_size, self.noise_dimension])
        y_generator = np.ones(batch_size)
        gen_loss = self.gan_model.train_on_batch(noise, y_generator)
        return gen_loss
    
    def fit(self, X, y=None, epochs=100, batch_size=128):
        """
        fit a gan model by alternating between the training of the discriminator and generator
        
        X::array like observations to imitate
        
        epochs::int number of batches to execute
        
        batch_size::int size of a batch
        """
        if self.generator is None or self.discriminator is None or self.gan_model is None:
            raise ValueError("model is not intialized correctly, please call init_model")
            

        for _ in tqdm(range(1, epochs+1), leave=False):
            # train discriminator and generator
            X_batch = X_train[np.random.randint(0, X.shape[0], size=batch_size)]
            dis_loss = self._fit_discriminator(X_batch)
            gen_loss = self._fit_generator(batch_size)
            
            # store loss
            self.discriminator_losses.append(dis_loss)
            self.generator_losses.append(gen_loss)
        
        return self
    

class MnistGanModel(BaseGanModel):
    """ This class is a Mnist Gan model
    
    label_noise::float apply noise during the training of the discriminator to avoid overfitting 
                       allow the generator to have more diversity.
    
    separate_fit::boolean fit the discriminator on the true and fake observation separatly.
    """
    
    def _create_generator(self, noise_dimension, output_size):
        # generator model
        generator = Sequential()
        generator.add(Dense(256, input_dim=noise_dimension, 
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(output_size, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return generator
    
    def _create_discriminator(self, output_size):
        # discriminator model
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=output_size, 
                                kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return discriminator
