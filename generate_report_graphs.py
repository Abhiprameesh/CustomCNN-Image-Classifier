import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def main():
    print("Loading model...")
    # Load the trained model
    model = tf.keras.models.load_model('best_model.keras')
    
    # Dataset path
    test_dir = os.path.join('dataset', 'test')
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found at {test_dir}. Please ensure dataset is present.")
        return

    print("Loading test data...")
    # Create a test generator (important to have shuffle=False for confusion matrix)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(96, 96),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get labels
    labels = list(test_gen.class_indices.keys())
    
    print("Generating predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved 'confusion_matrix.png'")
    
    # 2. Plot Model Architecture (if pydot/graphviz is installed)
    try:
        print("Generating Model Architecture Diagram...")
        tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
        print("Saved 'model_architecture.png'")
    except Exception as e:
        print(f"Could not generate 'model_architecture.png'. Ensure graphviz and pydot are installed. Error: {e}")

    # Generate classification report text
    report = classification_report(y_true, y_pred, target_names=labels)
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report:\n\n")
        f.write(report)
    print("Saved 'classification_report.txt'")
    print("\nAll graphs and reports generated successfully!")

if __name__ == '__main__':
    main()
