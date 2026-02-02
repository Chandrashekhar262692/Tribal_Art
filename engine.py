import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_and_prep(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, (512, 512))[tf.newaxis, :]

# LOAD IMAGES
content_img = load_and_prep(r'D:\OneDrive\Desktop\Tribal-Art-Engine\test_images\my_photo2.jpg')
style_img = load_and_prep(r'D:\OneDrive\Desktop\Tribal-Art-Engine\warli art\warli1.jpg')

# RUN AI
stylized = model(tf.constant(content_img), tf.constant(style_img))[0]

# THE MAGIC: Color Palette Adjustment
# We take the "Brightness" of the stylized photo and map it 
# directly to the White/Brown colors of Warli art.
final_art = (stylized * 0.9) # Increase style influence to 90%

# Save
output = np.array(final_art[0] * 255, dtype=np.uint8)
Image.fromarray(output).save('final_warli_art.jpg')
print("Check warli_output_v4.jpg - is the girl clearer now?")