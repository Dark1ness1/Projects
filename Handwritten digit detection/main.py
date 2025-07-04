from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

''' the CNN model was trained on Kaggle notebook
link: https://www.kaggle.com/code/talha1nasir/digit-recognizer-cnn-2-0 '''

# import the trained model
model = load_model('best_model.keras')

# load the image
img_path = 'images/my_digit_4.png'
img = Image.open(img_path).convert('L') # Converts the image to grayscale (L= luminace)

# Resize to 28x28 if not already
img = img.resize((28, 28))

# Convert to NumPy array
img_array = np.array(img)

"""
IMP: model show correct prediction only with black background and white pixel digit
and with white background and white digit pixel its gives wrong predictions
"""
# Invert colors if needed (you trained on white digits on black?)
# If your original training images had black digits on white background:
img_array = 255 - img_array

# Normalize to (o to 1)
img_array = img_array/255.0

# Reshape to (1, 28, 28, 1) for CNN input
image_array = img_array.reshape(1,28,28,1)

# visulization
plt.imshow(img_array, cmap='gray')
plt.title("processed Digit")
plt.show()

# predict
prediction = model.predict(image_array)
predict_digit = np.argmax(prediction)

print(f"The predicited digit is {predict_digit}")
