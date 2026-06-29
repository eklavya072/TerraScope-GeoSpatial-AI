import os
import hashlib
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataProcessor:
    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape

    @staticmethod
    def check_image_size(image_path):
        """Check dimensions of an image file"""
        with Image.open(image_path) as img:
            return img.size

    @staticmethod
    def check_image_dimensions(dataset_path):
        """Check dimensions of all images in dataset"""
        all_dimensions = set()
        for folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    width, height = DataProcessor.check_image_size(image_path)
                    all_dimensions.add((width, height))
        return all_dimensions

    @staticmethod
    def get_data_generators():
        """Get data generators for training and validation"""
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        test_gen = ImageDataGenerator(rescale=1./255)
        
        return train_gen, test_gen

    @staticmethod
    def get_image_hash(image_path):
        """Calculate MD5 hash of an image file"""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def check_duplicates(dataset_path):
        """Find duplicate images in dataset"""
        seen_hashes = set()
        duplicates = []
        for folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    img_hash = DataProcessor.get_image_hash(image_path)
                    if img_hash in seen_hashes:
                        duplicates.append(image_path)
                    else:
                        seen_hashes.add(img_hash)
        return duplicates