from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model

model = load_model('models/DeepLens_sudoku_model3.hdf5')
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'GeneratedDB\\test',
        target_size=(32, 32),
        batch_size=600,
        color_mode='grayscale',
        class_mode='categorical')

scores = model.evaluate_generator(validation_generator,
                                  steps=None,
                                  callbacks=None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False,
                                  verbose=0)
print("Score: ", scores)
#model.save("DeepLens_sudoku_model.hdf5")