from tensorflow import keras

# Load the model without loading the weights
dummy_model = keras.models.load_model('../../Models/combined_digit_model1.keras')

# Print the summary of the model
dummy_model.summary()

# Print layer information
for i, layer in enumerate(dummy_model.layers):
    print(f"Layer {i}: {layer.name}, {layer.__class__.__name__}")
    print(f"   Input shape: {layer.input_shape}, Output shape: {layer.output_shape}")
    print(f"   Config: {layer.get_config()}")
    print("")

# Also, print the names of all the layers in the model
print("All layer names in the model:")
print([layer.name for layer in dummy_model.layers])
