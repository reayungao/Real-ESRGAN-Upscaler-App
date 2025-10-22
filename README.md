<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-ESRGAN Image Upscaler Readme</title>
    <!-- We'll load Tailwind CSS to quickly style it like a GitHub readme -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* A little extra style for the code blocks */
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
        }
        code {
            font-family: monospace;
            font-size: 0.9em;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
    </style>
</head>
<body class="bg-white text-gray-900 leading-normal">

    <main class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        
        <!-- Main content container with GitHub-like styling -->
        <article class="prose prose-lg max-w-none p-8 border border-gray-200 rounded-lg shadow-sm">
            
            <h1 class="text-4xl font-bold border-b pb-4 mb-6">Real-ESRGAN Image Upscaler</h1>
            
            <p>A simple desktop application with a graphical user interface (GUI) for upscaling images using the powerful Real-ESRGAN models.</p>
            
            <p>This application allows you to easily select an image, preview it, choose the best upscaling model for your needs, and save the high-resolution result.</p>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">Features</h2>
            <ul>
                <li><strong>Simple GUI:</strong> An intuitive 3-step interface (Select, Upscale, Save).</li>
                <li><strong>Image Preview:</strong> See the image you've selected before you upscale.</li>
                <li><strong>Model Selection:</strong> Choose between different Real-ESRGAN models to get the perfect result for your image.</li>
                <li><strong>Cross-Platform:</strong> Built with Python and Tkinter, it should run on Windows, macOS, and Linux (once dependencies are installed).</li>
                <li><strong>Model Caching:</strong> Models are loaded into memory only once, making subsequent upscales much faster.</li>
            </ul>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">Models Explained</h2>
            <p>This app is configured to use two primary models. You must download them to use the app.</p>
            
            <h3 class="text-2xl font-semibold mt-6 mb-4">1. <code>RealESRGAN_x4plus.pth</code> (The "Artist")</h3>
            <ul>
                <li><strong>Best For:</strong> Digital art, anime, screenshots, and video game textures.</li>
                <li>This is a GAN-based model that produces extremely sharp and detailed results. It "invents" textures to make the image look sharp, but this can sometimes create artifacts or an "painted" look on photographs.</li>
            </ul>
            
            <h3 class="text-2xl font-semibold mt-6 mb-4">2. <code>RealESRNet_x4plus.pth</code> (The "Restorer")</h3>
            <ul>
                <li><strong>Best For:</strong> Photographs, portraits, and natural scenes.</li>
                <li>This model is trained to be as mathematically faithful to the original as possible. It produces a cleaner, more natural-looking result with far fewer artifacts and is much better at respecting existing textures (like skin) and blur (like bokeh).</li>
            </ul>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">Requirements</h2>
            <ul>
                <li>Python 3</li>
                <li>PyTorch (<code>torch</code>)</li>
                <li>OpenCV (<code>opencv-python</code>)</li>
                <li>NumPy (<code>numpy</code>)</li>
                <li>Pillow (<code>PIL</code>)</li>
            </ul>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">How to Run</h2>
            
            <h3 class="text-2xl font-semibold mt-6 mb-4">1. Download the Models</h3>
            <p>Before you can run the application, you <strong>must</strong> download the pre-trained models. Place the downloaded <code>.pth</code> files in the <strong>same directory</strong> as the <code>app-realesrgan.py</code> script.</p>
            <ul>
                <li><strong>Model 1 (Sharp):</strong> <a href="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" class="text-blue-600 hover:underline">RealESRGAN_x4plus.pth</a></li>
                <li><strong>Model 2 (Natural):</strong> <a href="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" class="text-blue-600 hover:underline">RealESRNet_x4plus.pth</a></li>
            </ul>
            
            <h3 class="text-2xl font-semibold mt-6 mb-4">2. Install Dependencies</h3>
            <p>Open your terminal or command prompt and install the required Python libraries:</p>
            <pre><code>pip install torch torchvision opencv-python-headless numpy Pillow</code></pre>
            <p><em>(Note: <code>torchvision</code> is included as it's often helpful with <code>torch</code>. <code>opencv-python-headless</code> is used instead of the full <code>opencv-python</code> as we only need the processing functions, not its own GUI capabilities.)</em></p>
            
            <h3 class="text-2xl font-semibold mt-6 mb-4">3. Run the Application</h3>
            <p>Once the dependencies and models are in place, run the script from your terminal:</p>
            <pre><code>python app-realesrgan.py</code></pre>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">How to Use</h2>
            <ol class="list-decimal list-inside">
                <li><strong>Select Image:</strong> Click the "1. Select Image" button to open a file dialog. Choose the image you want to upscale. A preview will appear in the window.</li>
                <li><strong>Select Model:</strong> Use the dropdown menu to choose the best model for your image (see "Models Explained" above).</li>
                <li><strong>Upscale Image:</strong> Click the "2. Upscale Image" button. The app will freeze momentarily while processing (this is faster if you have a <code>cuda</code>-compatible GPU). A "Complete" message will pop up when it's done.</li>
                <li><strong>Save Upscaled:</strong> Click the "3. Save Upscaled" button. A save dialog will appear, suggesting a new file name that includes the model you used (e.g., <code>my_image_RealESRNet_x4plus.png</code>).</li>
            </ol>
            
            <h2 class="text-3xl font-bold border-b pb-3 mt-10 mb-5">Credits</h2>
            <ul>
                <li>This application is a simple GUI wrapper for the models developed by Xintao Wang et al.</li>
                <li><strong>Original Project:</strong> <a href="https://github.com/xinntao/Real-ESRGAN" class="text-blue-600 hover:underline">https://github.com/xinntao/Real-ESRGAN</a></li>
            </ul>

        </article>

    </main>

</body>
</html>
