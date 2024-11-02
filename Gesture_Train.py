import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import json

class GestureModelTrainer:
    def __init__(self, data_path='./gesture/', img_size=50, validation_split=0.2, test_split=0.2):
        self.data_path = data_path
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.model = None
        self.history = None
        
        # Create directories for outputs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"training_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess image data with augmentation"""
        print("Loading and preprocessing data...")
        x_data, y_data = [], []
        self.class_names = []
        
        # Load images and labels
        for gesture_class in os.listdir(self.data_path):
            if gesture_class.startswith('.'):
                continue
                
            self.class_names.append(gesture_class)
            gesture_path = os.path.join(self.data_path, gesture_class)
            
            for img_name in os.listdir(gesture_path):
                if img_name.startswith('.'):
                    continue
                    
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = np.array(img).reshape((self.img_size, self.img_size, 1))
                img = img / 255.0
                
                x_data.append(img)
                y_data.append(int(gesture_class))
        
        self.x_data = np.array(x_data)
        self.y_data = to_categorical(np.array(y_data))
        self.num_classes = self.y_data.shape[1]
        
        # Save class mapping
        class_mapping = {i: name for i, name in enumerate(self.class_names)}
        with open(f"{self.output_dir}/class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=4)
        
        # Split data into train, validation, and test sets
        x_temp, self.x_test, y_temp, self.y_test = train_test_split(
            self.x_data, self.y_data, test_size=self.test_split, random_state=42
        )
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_temp, y_temp, test_size=self.validation_split, random_state=42
        )
        
        print(f"Training samples: {len(self.x_train)}")
        print(f"Validation samples: {len(self.x_val)}")
        print(f"Test samples: {len(self.x_test)}")
        
    def create_model(self):
        """Create an enhanced CNN model with modern architecture"""
        print("Creating model...")
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), padding='same', input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Dense Layers
            Dense(512),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            ModelCheckpoint(
                f"{self.output_dir}/models/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            TensorBoard(
                log_dir=f"{self.output_dir}/logs"
            )
        ]
        return callbacks
    
    def create_data_generators(self):
        """Create data generators for training augmentation"""
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model with data augmentation"""
        print("Training model...")
        
        # Setup optimizers and compile model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Setup data generators
        train_datagen, val_datagen = self.create_data_generators()
        
        # Train the model
        self.history = self.model.fit(
            train_datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
            validation_data=val_datagen.flow(self.x_val, self.y_val, batch_size=batch_size),
            epochs=epochs,
            callbacks=self.setup_callbacks(),
            verbose=1
        )
        
    def evaluate_model(self):
        """Evaluate model performance and generate visualizations"""
        print("Evaluating model...")
        
        # Evaluate on test set
        test_scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print(f"Test accuracy: {test_scores[1]*100:.2f}%")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/training_history.png")
        
        # Generate confusion matrix
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/confusion_matrix.png")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(f"{self.output_dir}/training_history.csv")
        
        # Save model summary
        with open(f"{self.output_dir}/model_summary.txt", 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
        # Save final model
        self.model.save(f"{self.output_dir}/models/final_model.h5")
        
        return test_scores

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create and run trainer
    trainer = GestureModelTrainer()
    trainer.load_and_preprocess_data()
    trainer.create_model()
    trainer.train_model()
    test_scores = trainer.evaluate_model()
    
    print("\nTraining completed!")
    print(f"Final test accuracy: {test_scores[1]*100:.2f}%")
    print(f"All outputs saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()
