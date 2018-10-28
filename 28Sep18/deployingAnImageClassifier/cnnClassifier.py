
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initializing CNN

def main():
        classifier = Sequential()

        # Step 1: Convolution
        '''
        Conv2D(filter,shape,imageShape,activationFunction)
        '''
        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

        # Step 2 : Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 : Flatenning
        classifier.add(Flatten())

        # Step 4 : Full Connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Image Pre-Processing 
        '''
        Source of Image Processing https://keras.io/preprocessing/image/
        '''

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory('dataSets/training_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

        test_set = test_datagen.flow_from_directory('dataSets/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

        classifier.fit_generator(training_set,
                                steps_per_epoch = 8000,
                                epochs = 25,
                                validation_data = test_set,
                                validation_steps = 2000)


        # ------------------------------------
        # Saving Model to Local
        # ------------------------------------

        # serialize model to JSON
        model_json = classifier.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        classifier.save_weights("model.h5")
        print("Saved model to disk")

        # ------------------------------------
        # Loading Model
        # ------------------------------------

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Testing With Data
        import numpy as np
        from keras.preprocessing import image

        test_image = image.load_img('dataSets/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        #result = classifier.predict(test_image)

        result = loaded_model.predict(test_image)

        training_set.class_indices

        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'

if __name__ == "__main__":
    main()
