from tensorflow.keras.models import load_model
import numpy as np

model = load_model('best_model.keras')

# Fake image for test
dummy_input = np.random.rand(1, 28, 28, 1)
prediction = model.predict(dummy_input)

print("Prediction:", prediction)
print("Predicted digit:", np.argmax(prediction))

# issue the model is not printing the correct result for paint file so to test it I run dummy code
# the model is correctly loading and predicting
# going back to main file