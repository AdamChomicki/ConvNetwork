import os
import cv2
import tensorflow.keras as keras
import numpy as np


def read_images(dir:str, input_shape:tuple):
  return np.array([cv2.resize(cv2.imread(os.path.join(dir, path)), input_shape[0:2]) for path in os.listdir(dir)])


class createAugment(keras.utils.Sequence):

    def __init__(self, X, y, batch_size=32, dim=(160, 160), n_channels=3, shuffle=True):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Oznacza liczbę batchy na epokę'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generuje jeden batch danych'
        # Generuj indeksy batchy
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generuj dane
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Aktualizuj indeksy po każdej epoce'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs):
        # Masked_images jest macierzą zamaskowanych obrazów używanych jako dane wejściowe
        Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))  # Zamaskowany obraz
        # Mask_batch jest macierzą masek binarnych używanych jako dane wejściowe
        Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))  # Maski binarne
        # y_batch jest macierzą oryginalnych obrazów używanych do obliczania błędu z zrekonstruowanego obrazu
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))  # Obraz oryginalny

        ## Iteracja przez losowe indeksy
        for i, idx in enumerate(idxs):
            image_copy = self.X[idx].copy()

            ## Pobierz maskę związaną z tym obrazem
            masked_image, mask = self.__createMask(image_copy)

            Masked_images[i,] = masked_image / 255
            Mask_batch[i,] = mask / 255
            y_batch[i] = self.y[idx] / 255

        ## Return mask as well because partial convolution require the same.
        return [Masked_images, Mask_batch], y_batch

    def __createMask(self, img):
        ## Przygotuj maskującą macierz
        mask = np.full((*self.dim, 3), 255, np.uint8)  ## Białe tło
        for _ in range(np.random.randint(1, 10)):
            # Pobierz losowe lokalizacje x do linii startu
            x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[0])
            # Pobierz losowe lokalizacje y do linii startu
            y1, y2 = np.random.randint(1, self.dim[1]), np.random.randint(1, self.dim[1])
            # Uzyskaj losową grubość rysowanej linii
            thickness = np.random.randint(1, 3)
            # Narysuj czarną linię na białej masce
            cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)

        ## Maska zdjęcia
        masked_image = img.copy()
        masked_image[mask == 0] = 255

        return masked_image, mask