import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import urllib.request
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback  # For detailed error logging
import functools
from PIL import Image, ImageTk

# ---
#
# START: Real-ESRGAN Model Definition (RRDBNet)
# (This code is adapted from the official Real-ESRGAN repository: https://github.com/xinntao/Real-ESRGAN)
#
# ---

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization logic (can be omitted for inference if loading pre-trained weights)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the trunk network.
        num_grow_ch (int): Channels for growth.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# ---
#
# END: Real-ESRGAN Model Definition
#
# ---


class RealESRGAN_Upscaler:
    """Helper class to manage the upscaling process."""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the Real-ESRGAN model."""
        # The RealESRGAN_x4plus model uses these parameters
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        
        try:
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")

        # Support for different state_dict formats
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
            
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        return model.to(self.device)

    def process(self, img):
        """Upscale a single image (must be a cv2 image BGR)."""
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output


class App:
    """The main Tkinter application GUI."""
    
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x650") # Increased size to accommodate preview and dropdown
        self.root.minsize(700, 550) # Minimum size

        # --- Model Management ---
        self.models_info = {
            # Model name: Download URL
            "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRNet_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        }
        self.selected_model_name = tk.StringVar(value=list(self.models_info.keys())[0]) # Default to first one
        self.self_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_path = "" # Will be set when model is selected
        
        self.input_path = ""
        self.original_image = None # Stores the original PIL image
        self.upscaled_image = None # Stores the upscaled cv2 image
        self.upscalers = {} # Cache for loaded models, e.g., {"model_name.pth": upscaler_instance}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Configure Style ---
        style = ttk.Style()
        style.theme_use('clam') # Modern theme
        style.configure('TFrame', background='#e8f0f7')
        style.configure('TButton', padding=8, relief='flat', background='#007acc', foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', '#005f99')])
        style.configure('TLabel', background='#e8f0f7', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#333333')
        style.configure('Status.TLabel', font=('Segoe UI', 9, 'italic'), foreground='#555555')
        style.configure('TProgressbar', thickness=10, troughcolor='#cccccc', background='#007acc')

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="15", style='TFrame')
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.columnconfigure(0, weight=1) # Allow column 0 to expand
        main_frame.columnconfigure(1, weight=1) # Allow column 1 to expand
        main_frame.rowconfigure(4, weight=1) # Allow image preview row to expand (changed from 3)

        # --- Header ---
        header_label = ttk.Label(main_frame, text="Real-ESRGAN Upscaler", style='Header.TLabel', anchor=tk.CENTER)
        header_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        device_label = ttk.Label(main_frame, text=f"Running on: {str(self.device).upper()}", style='Status.TLabel', anchor=tk.CENTER)
        device_label.grid(row=1, column=0, columnspan=2, pady=(0, 15))

        # --- Model Selection Dropdown ---
        model_frame = ttk.Frame(main_frame, style='TFrame')
        model_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        model_frame.columnconfigure(1, weight=1)
        
        model_label = ttk.Label(model_frame, text="Select Model:", style='TLabel')
        model_label.grid(row=0, column=0, padx=(0, 10))

        model_dropdown = ttk.OptionMenu(
            model_frame,
            self.selected_model_name,
            list(self.models_info.keys())[0], # Default selection
            *self.models_info.keys() # All other options
        )
        model_dropdown.grid(row=0, column=1, sticky="ew")


        # --- Buttons Frame ---
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.grid(row=3, column=0, columnspan=2, pady=(5, 10)) # Changed row to 3
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        self.select_btn = ttk.Button(button_frame, text="1. Select Image", command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.upscale_btn = ttk.Button(button_frame, text="2. Upscale Image", command=self.start_upscaling_thread, state=tk.DISABLED)
        self.upscale_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.save_btn = ttk.Button(button_frame, text="3. Save Upscaled", command=self.save_upscaled_image, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # --- Image Preview ---
        preview_frame = ttk.Frame(main_frame, relief="solid", borderwidth=1, style='TFrame')
        preview_frame.grid(row=4, column=0, columnspan=2, pady=(10, 15), sticky="nsew") # Changed row to 4
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.image_label = tk.Label(preview_frame, bg='#dddddd') # Use a plain Label for image display
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Status and Progress ---
        self.status_label = ttk.Label(main_frame, text="Ready: Select an image to begin.", style='Status.TLabel', wraplength=750, anchor=tk.W)
        self.status_label.grid(row=5, column=0, columnspan=2, pady=(5, 5), sticky="ew") # Changed row to 5

        self.progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress_bar.grid(row=6, column=0, columnspan=2, pady=(0, 10), sticky="ew") # Changed row to 6

        # --- Model Note ---
        model_note_text = (
            "Please download your desired models and place them in the same directory as this script.\n\n"
            "• RealESRGAN (Sharp, default): ...releases/download/v0.1.0/RealESRGAN_x4plus.pth\n"
            "• RealESRNet (More natural, less artifacts): ...releases/download/v0.1.1/RealESRNet_x4plus.pth\n"
            "(Full GitHub URL is in the code)"
        )
        note_label = ttk.Label(main_frame, text=model_note_text, style='Status.TLabel', anchor=tk.CENTER, justify=tk.CENTER)
        note_label.grid(row=7, column=0, columnspan=2, pady=(10, 0), sticky="ew") # Changed row to 7

    def select_image(self):
        """Open a file dialog to select an image and display its preview."""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if path:
            self.input_path = path
            self.status_label.config(text=f"Selected: {os.path.basename(path)}")
            self.upscale_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED) # Disable save until upscaled
            self.upscaled_image = None # Clear previously upscaled image

            # Display preview
            try:
                # Read with OpenCV for consistency with model input
                img_cv2 = cv2.imread(self.input_path, cv2.IMREAD_COLOR)
                if img_cv2 is None:
                    raise IOError(f"Failed to read image: {self.input_path}")
                
                # Convert BGR (OpenCV) to RGB (Pillow)
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                self.original_image = Image.fromarray(img_rgb)
                
                self.display_image(self.original_image)

            except Exception as e:
                messagebox.showerror("Error", f"Could not load image preview: {e}")
                self.image_label.config(image='', text='Error loading image preview.')
                self.original_image = None
                self.input_path = ""
                self.status_label.config(text="No file selected or error loading preview.")
                self.upscale_btn.config(state=tk.DISABLED)
        else:
            self.input_path = ""
            self.original_image = None
            self.upscaled_image = None
            self.status_label.config(text="No file selected.")
            self.upscale_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.image_label.config(image='', text='No image selected.')


    def display_image(self, pil_image):
        """Displays a PIL Image object in the image_label, scaled to fit."""
        img_width, img_height = pil_image.size
        
        # Get label dimensions (approximate, or bind to configure event for exact)
        # For simplicity, let's use a fixed max size for the preview area
        preview_widget_height = self.image_label.winfo_height()
        preview_widget_width = self.image_label.winfo_width()
        
        # Use a sensible default if the widget hasn't been drawn yet
        if preview_widget_height <= 1 or preview_widget_width <= 1:
            preview_widget_height = 400
            preview_widget_width = 750 # Approx width of frame

        ratio_w = preview_widget_width / img_width
        ratio_h = preview_widget_height / img_height
        ratio = min(ratio_w, ratio_h)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # Ensure minimum size if image is tiny, or prevent upscale of tiny images
        if new_width == 0 or new_height == 0:
            new_width = min(img_width, max_preview_width)
            new_height = min(img_height, max_preview_height) # This might distort aspect ratio
            if new_width/img_width < new_height/img_height:
                new_height = int(img_height * (new_width/img_width))
            else:
                new_width = int(img_width * (new_height/img_height))
            if new_width == 0: new_width = 1 # Prevent zero division for extremely thin/short images
            if new_height == 0: new_height = 1

        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image, text='')
        self.root.update_idletasks() # Update GUI to show image immediately

    def start_upscaling_thread(self):
        """Start the upscaling process in a separate thread to avoid freezing the GUI."""
        if not self.input_path or self.original_image is None:
            messagebox.showwarning("No Image", "Please select an image file first.")
            return

        self.select_btn.config(state=tk.DISABLED)
        self.upscale_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.status_label.config(text="Upscaling, please wait...")

        # Run the heavy computation in a new thread
        threading.Thread(target=self.run_upscaling, daemon=True).start()

    def run_upscaling(self):
        """The core logic for loading the model and upscaling the image."""
        try:
            # --- 1. Load Model (if not already loaded) ---
            model_name = self.selected_model_name.get()
            self.model_path = os.path.join(self.self_dir, model_name)

            if model_name not in self.upscalers:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model not found. Please download '{model_name}' to the script directory.")
                
                self.status_label.config(text=f"Loading model: {model_name}...")
                # Create and cache the upscaler instance
                self.upscalers[model_name] = RealESRGAN_Upscaler(self.model_path, self.device)

            # Get the correct upscaler from our cache
            current_upscaler = self.upscalers[model_name]

            # --- 2. Read and Process Image ---
            self.status_label.config(text="Processing image...")
            # Use self.original_image (PIL) and convert to CV2 BGR for the model
            img_cv2_rgb = np.array(self.original_image)
            img_cv2_bgr = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2BGR)

            output_img_bgr = current_upscaler.process(img_cv2_bgr)
            self.upscaled_image = output_img_bgr # Store the upscaled image

            # Display success and enable save button
            self.status_label.config(text=f"Upscaling complete! Click 'Save Upscaled' to save.")
            self.save_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Upscaling Complete", "Image successfully upscaled. You can now save it.")

        except Exception as e:
            # --- 6. Show Error ---
            error_details = traceback.format_exc()
            print(f"Error during upscaling: {error_details}")
            self.status_label.config(text="An error occurred. Check console.")
            messagebox.showerror("Error", f"An error occurred:\n\n{e}\n\nSee console for full traceback.")
        
        finally:
            # --- 7. Reset GUI State ---
            self.select_btn.config(state=tk.NORMAL) # Allow selecting new image
            if self.input_path: # If an image was selected
                self.upscale_btn.config(state=tk.NORMAL) # Re-enable upscale if another image is selected
            self.progress_bar.stop()

    def save_upscaled_image(self):
        """Saves the upscaled image to a user-specified location."""
        if self.upscaled_image is None:
            messagebox.showwarning("No Image", "No upscaled image available to save. Please upscale an image first.")
            return
        
        base, ext = os.path.splitext(os.path.basename(self.input_path))
        save_dir = os.path.dirname(self.input_path)
        
        # Add model name to the output file
        model_name_base, _ = os.path.splitext(self.selected_model_name.get())
        save_name = f"{base}_{model_name_base}{ext}" # More descriptive name

        save_path_user = filedialog.asksaveasfilename(
            title="Save upscaled image",
            initialdir=save_dir,
            initialfile=save_name,
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )

        if save_path_user:
            try:
                cv2.imwrite(save_path_user, self.upscaled_image)
                self.status_label.config(text=f"Saved: {os.path.basename(save_path_user)}")
                messagebox.showinfo("Save Successful", f"Upscaled image saved to:\n{save_path_user}")
                self.save_btn.config(state=tk.DISABLED) # Disable save until next upscale
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {e}")
        else:
            self.status_label.config(text="Save operation cancelled.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()


