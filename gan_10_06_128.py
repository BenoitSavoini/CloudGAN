# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:54:17 2021

@author: Alkios
"""



from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Reshape, LocallyConnected2D, Flatten, Input, ReLU, Conv2D, Dropout, Conv2DTranspose, BatchNormalization, UpSampling2D, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



class GAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.latent_dim = 128
        
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

  
    def build_generator(self):

        noise_shape = (self.latent_dim,)
        
        model = Sequential()
        
        model.add(Dense(32*32*128, activation="relu", input_shape=noise_shape))
        model.add(Reshape((32, 32, 128)))

        #model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(ReLU())
        #model.add(UpSampling2D())
        
        model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        
        model.add(Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        
        #model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(ReLU())
        
        #model.add(UpSampling2D())
        #model.add(Conv2D(32, (3, 3), padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU())

    
        model.add(Conv2DTranspose(1, (3, 3), padding='same', activation='tanh'))
        
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()
        
        model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
          
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        
        #model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.4))
        
        #model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.4))
        
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    
    def train(self, epochs, X_train, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)
        dl = []
        gl = []
        acc = []
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            print('{} [D loss: {}, accuracy: {:.2f}] [G loss: {}]'.format(epoch, d_loss[0], 100 * d_loss[1], g_loss))
            
            dl.append(d_loss[0])
            acc.append(100*d_loss[1])
            gl.append(g_loss)
            
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        
        if (epoch == epochs-1) :
            plt.plot(range(epochs), dl, 'b', label='Discriminator loss')
            plt.plot(range(epochs), gl, 'g', label='Generator loss')
            
            plt.title('G_loss and D_loss in function of the number of epochs')
            plt.legend()
            
            plt.show()
            plt.close()
            
            plt.plot(range(epochs), acc, 'or', label='Accuracy')
            plt.title('Accuracy in function of the number of epochs')
            plt.legend()
            
            plt.show()
            plt.close()

    def save_imgs(self, epoch):

        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        plt.imshow(gen_imgs[0].reshape(self.img_rows,self.img_cols) * 127.5 + 127.5, cmap = 'hot')

        plt.savefig("C://Users/Alkios/Downloads/ProjetM1/timeresults/Particles_%d.png" % epoch)
        plt.close()

if __name__ == "__main__":
	gan = GAN()
	gan.train(epochs=500, X_train = train_images, batch_size=32, save_interval=25)
    
    