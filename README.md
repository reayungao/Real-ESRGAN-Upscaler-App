**[WORK IN PROGRESS]**

**Real-ESRGAN Image Upscaler**

A simple desktop application with a graphical user interface (GUI) for upscaling images using the powerful Real-ESRGAN models.

This application allows you to easily select an image, preview it, choose the best upscaling model for your needs, and save the high-resolution result.

(Note: You will need to upload your own screenshot and replace this link)

**Features**

- Simple GUI: An intuitive 3-step interface (Select, Upscale, Save).

- Image Preview: See the image you've selected before you upscale.

- Model Selection: Choose between different Real-ESRGAN models to get the perfect result for your image.

- Cross-Platform: Built with Python and Tkinter, it should run on Windows, macOS, and Linux (once dependencies are installed).

- Model Caching: Models are loaded into memory only once, making subsequent upscales much faster.
<br>

**Models Explained**

This app is configured to use two primary models. You must download them to use the app.

1. RealESRGAN_x4plus.pth (The "Artist")

Best For: Digital art, anime, screenshots, and video game textures.

This is a GAN-based model that produces extremely sharp and detailed results. It "invents" textures to make the image look sharp, but this can sometimes create artifacts or an "painted" look on photographs.

2. RealESRNet_x4plus.pth (The "Restorer")

Best For: Photographs, portraits, and natural scenes.

This model is trained to be as mathematically faithful to the original as possible. It produces a cleaner, more natural-looking result with far fewer artifacts and is much better at respecting existing textures (like skin) and blur (like bokeh).

<br>

**Requirements**

- Python 3

- PyTorch (torch)

- OpenCV (opencv-python)

- NumPy (numpy)

- Pillow (PIL)

<br>


**How to Run**

1. Download the Models

Before you can run the application, you must download the pre-trained models. Place the downloaded .pth files in the same directory as the app-realesrgan.py script.

Model 1 (Sharp): RealESRGAN_x4plus.pth

Model 2 (Natural): RealESRNet_x4plus.pth

2. Install Dependencies

Open your terminal or command prompt and install the required Python libraries:

pip install torch torchvision opencv-python-headless numpy Pillow


(Note: torchvision is included as it's often helpful with torch. opencv-python-headless is used instead of the full opencv-python as we only need the processing functions, not its own GUI capabilities.)

3. Run the Application

Once the dependencies and models are in place, run the script from your terminal:

python app-realesrgan.py

<br>


**How to Use**

- Select Image: Click the "1. Select Image" button to open a file dialog. Choose the image you want to upscale. A preview will appear in the window.

- Select Model: Use the dropdown menu to choose the best model for your image (see "Models Explained" above).

- Upscale Image: Click the "2. Upscale Image" button. The app will freeze momentarily while processing (this is faster if you have a cuda-compatible GPU). A "Complete" message will pop up when it's done.

- Save Upscaled: Click the "3. Save Upscaled" button. A save dialog will appear, suggesting a new file name that includes the model you used (e.g., my_image_RealESRNet_x4plus.png).

<br>

**Credits**

This application is a simple GUI wrapper for the models developed by Xintao Wang et al.

Original Project: https://github.com/xinntao/Real-ESRGAN
