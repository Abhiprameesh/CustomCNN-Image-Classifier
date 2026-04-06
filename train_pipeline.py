import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Constraints and Parameters
IMG_SIZE = (96, 96) # 96x96 optimal for <400KB size
BATCH_SIZE = 32
EPOCHS = 40
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def build_model(input_shape=(96, 96, 3), num_classes=4):
    """
    Builds a lightweight custom CNN pipeline.
    Goal: Maximize accuracy while keeping parameters under ~400,000 for a <400KB INT8 TFLite model.
    """
    model = models.Sequential([
        # Input layer
        layers.InputLayer(input_shape=input_shape),
        
        # Block 1
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(), # reduces to 48x48
        
        # Block 2
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(), # reduces to 24x24
        
        # Block 3
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(), # reduces to 12x12
        
        # Block 4
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(), # reduces to 6x6
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # dropout for reducing overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_data_generators():
    """
    Returns data generators with strong augmentations suitable for training.
    """
    # Strong augmentation for training, rescaling is required.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for val and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_datagen

def train_and_evaluate():
    print("Loading data generators...")
    train_gen, val_gen, test_gen, _ = get_data_generators()
    
    # Class tracking
    labels = list((train_gen.class_indices).keys())
    with open('labels.txt', 'w') as f:
        for label in labels:
            f.write(f'{label}\n')
            
    print(f"Classes: {labels}")

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(labels))
    print("\n--- Model Summary ---")
    model.summary()
    
    # Size check (Rough check: 1 float32 parameter = 4 bytes. For INT8, it will be 1 byte)
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    est_int8_size_kb = trainable_count / 1024
    print(f"\nEstimated TFLite Size (INT8 Quantized): {est_int8_size_kb:.2f} KB")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
    ]
    
    print("\n--- Starting Training ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    print("\n--- Evaluating on Test Set ---")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    return model, history, train_gen

def representative_data_gen(train_gen):
    """ generator yielding un-batched data for quantization """
    def data_gen():
        # yields a small subset to calibrate ranges
        for i in range(10): 
            # Get a batch
            images, _ = next(train_gen)
            # Yield image by image inside the batch
            for img in images:
                yield [np.expand_dims(img, axis=0).astype(np.float32)]
    return data_gen

def convert_to_tflite_int8(model, train_gen):
    print("\n--- Converting to TFLite (INT8) ---")
    
    # Required conversion to get specific input/output shapes for ESP32
    # Convert Keras model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen(train_gen)
    
    # Ensure full INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_model_quant = converter.convert()
        
        tflite_name = 'model_quant.tflite'
        with open(tflite_name, 'wb') as f:
            f.write(tflite_model_quant)
            
        real_size = os.path.getsize(tflite_name) / 1024
        print(f"Quantization Successful! Final size: {real_size:.2f} KB")
    except Exception as e:
        print(f"Quantization failed with error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Only print model summary and exit')
    args = parser.parse_args()
    
    if args.dry_run:
        model = build_model()
        model.summary()
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        est_int8_size_kb = trainable_count / 1024
        print(f"\nEstimated TFLite Size (INT8 Quantized): {est_int8_size_kb:.2f} KB")
        print("\nDry run completed.")
    else:
        model, history, train_gen = train_and_evaluate()
        convert_to_tflite_int8(model, train_gen)
