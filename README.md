# ClassifyAircraftDefect

**Part 1 - Classification Problem: Classifying the defect on the aircraft as 'dent' or 'crack'**

**1.1	Dataset Preparation**

The first step is to load and prepare the dataset of aircraft images. These images are labeled either as 'dent' or 'crack'. We will also split the dataset into training, validation, and test sets.

**1.2	Data Preprocessing**

Create data generators for training, validation, and testing datasets.

First,we will create ImageDataGenerators used for training, validation and testing.
The ImageDataGenerator class is part of Keras. It is a powerful utility for real-time image data augmentation, preprocessing, and feeding data into deep learning models during training. This class is particularly useful when working with image datasets that are too large to fit into memory all at once, or when you want to augment your dataset  to improve model generalization.

**1.3	Model Definition**

Here, we define the model architecture by using a pre-trained VGG16 model as the base, adding custom layers on top for binary classification of 'dent' and 'crack' types of damage.

First, Load the pre-trained model VGG16  and then modify the VGG16 model for our specific classification task. We extract the output from the last layer of the pre-trained VGG16 model, and then create a new model with this modified output. Then we will freeze the base VGG16 model layers so that their weights will not be updated during training.

After using VGG16 as a feature extractor, we add our own classifier on top of the VGG16 model. This involves adding fully connected layers (Dense), activation functions (like ReLU), and sometimes Dropout layers to avoid overfitting.
Here, we are adding two dense layers with 512 units each, followed by a Dropout layer, and finally, a Dense layer with one unit and a sigmoid activation to output the probability for binary classification ("dent" vs "crack").

**1.4	Compile the model**

Compile the model using following parameters
*   loss: `'binary_crossentropy'`.
*   optimizer: `=Adam(learning_rate=0.0001)`.
*   metrics: `['accuracy']`.

**1.5	Model Training**

Now that the model is compiled, you can train it using the .fit() method. This step involves passing in the training and validation datasets along with the number of epochs you want to train the model for.


**1.6	Visualizing Training Results**
 <img width="975" height="702" alt="image" src="https://github.com/user-attachments/assets/0995a667-5c34-4970-a9bd-680f76a8c3d7" />

 <img width="975" height="694" alt="image" src="https://github.com/user-attachments/assets/dd29d293-1474-4dab-9bac-09b7a5626b3d" />

<img width="975" height="706" alt="image" src="https://github.com/user-attachments/assets/c3a20ce4-e8a8-4821-9644-95eba715e7f2" />

 
**1.7	Model Evaluation**

Now we evaluate the trained model on the test dataset. Calculates test loss and accuracy by evaluating the test generator. Predictions are made for the test dataset, and the results are compared to true labels.
1.8	Visualizing the Predictions
Display test images alongside their true and predicted labels. True labels and predictions are retrieved.
<img width="965" height="828" alt="image" src="https://github.com/user-attachments/assets/ec46e32b-94b5-43a8-a61a-ab1860d920ff" />

 



**Part 2: Image Captioning and Summarization using BLIP Pretrained Model**

BLIP (Bootstrapping Language-Image Pretraining) is an advanced vision-and-language 
model designed to generate natural language descriptions for images. By leveraging both visual and textual information, BLIP can produce human-readable text that accurately reflects the content and context of an image. It is specifically trained to understand images and their relationships to summarizing text, making it ideal for tasks like image captioning, summarization, and visual question answering.

In this project, learners will utilize the BLIP model to build a system capable of automatically generating captions and summary for images. The code will integrate the BLIP model within a custom Keras layer. This allows the user to input an image and specify a task, either "caption" or "summary", to receive a textual output that describes or summarizes the content of the image.

**Key Steps:**

•	**Image Loading and Preprocessing:** The code will begin by loading images from a file path, then converting and processing them into a format suitable for input to the BLIP model.
•	**Text Generation:** Depending on the task, whether generating a caption or summary, the BLIP model will generate corresponding text based on the processed image.
•	**Custom Keras Layer:** A custom Keras layer is a user-defined layer that extends Keras' built-in functionality.Here custom Keras layer will be implemented to wrap the BLIP model. This layer will handle the task-specific processing (captioning or summarizing) and integrate smoothly into a TensorFlow/Keras environment.

**2.1 Loading BLIP Model**

Load the BLIP Model and Processor from Hugging Face
Hugging Face is an open-source platform that provides pre-trained machine learning models, datasets, and tools, primarily focused on natural language processing, computer vision, and other AI tasks. It offers easy access to powerful models through its Transformers library.
•	**BlipProcessor** - This handles the preprocessing of images and text. It converts images to the format that the BLIP model can understand.
•	**BlipForConditionalGeneration** - This is the model itself, responsible for generating captions or summaries based on the processed image.
**Custom Keras Layer: BlipCaptionSummaryLayer**
Next, we define a custom `tf.keras.layers.Layer` class that takes in an image and a task input (either caption or summary) and processes the image using the BLIP model. To create a custom Keras layer, we need to subclass `tf.keras.layers.Layer` and implement the required methods.
Also, we will need to implement a helper function `generate_text` that utilizes the custom `BlipCaptionSummaryLayer` Keras layer to generate captions or summaries for a given image. The function will accept an image path and a task type (caption or summary), process the image using the BLIP model, and return the generated text.

**2.2 Generating Captions and Summaries**

It processes the image and prints the corresponding text output in a human-readable format.
The function makes it easy to switch between generating captions and summaries based on the task type you provide.
Note: Generated captions/summary may not always be accurate, as the model is limited by its training data and may not fully understand new or specific images.
