import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import os

# --- 1. CONFIGURATION ---
# Paths to your models (Local paths)
GENDER_MODEL_PATH = "gender_model_final.pth"
AGE_MODEL_PATH    = "age_model_best_66acc.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITIONS (Must match training exactly) ---

def get_gender_model():
    """Standard ResNet18 for Gender"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def get_age_model():
    """Robust ResNet18 with Dropout for Age"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # We must match the Sequential structure from Cell 16/14
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )
    return model

# --- 3. LOAD MODELS ---
print("Loading models...")
try:
    # Load Gender
    gender_net = get_gender_model()
    gender_net.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
    gender_net.to(DEVICE)
    gender_net.eval()
    
    # Load Age
    age_net = get_age_model()
    age_net.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
    age_net.to(DEVICE)
    age_net.eval()
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Model files not found. Check your paths!")
    # We create dummy models so the GUI doesn't crash immediately (for debugging)
    gender_net = get_gender_model().to(DEVICE)
    age_net = get_age_model().to(DEVICE)

# --- 4. PREDICTION FUNCTION ---
# Image transformation (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_handwriting(image):
    if image is None:
        return "No Image", "No Image"
    
    # Preprocess
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Gender Prediction
        g_out = gender_net(img_t)
        g_probs = torch.softmax(g_out, 1)
        g_conf, g_pred = torch.max(g_probs, 1)
        
        gender_labels = ['Male', 'Female']
        gender_res = gender_labels[g_pred.item()]
        gender_str = f"{gender_res} ({g_conf.item()*100:.1f}%)"
        
        # Age Prediction
        a_out = age_net(img_t)
        a_probs = torch.softmax(a_out, 1)
        a_conf, a_pred = torch.max(a_probs, 1)
        
        age_labels = ['< 15', '16 - 25', '26 - 50', '> 50']
        age_res = age_labels[a_pred.item()]
        age_str = f"{age_res} years ({a_conf.item()*100:.1f}%)"
        
    return gender_str, age_str

# --- 5. DESKTOP GUI (Tkinter) ---
class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KHATT Handwriting Analyzer")
        self.root.geometry("600x650")
        # self.root.resizable(False, False) # Allow resizing
        
        # Style configuration
        title_font = ("Helvetica", 20, "bold")
        label_font = ("Helvetica", 12)
        btn_font = ("Helvetica", 10, "bold")
        
        # --- Main Container ---
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        tk.Label(self.main_frame, text="🖊️ KHATT Analyzer", font=title_font, fg="#333", bg="white").pack(pady=10)
        
        # Image Display Area
        self.img_frame = tk.Frame(self.main_frame, width=500, height=400, bg="#f0f0f0", highlightbackground="#ccc", highlightthickness=1)
        self.img_frame.pack_propagate(False)
        self.img_frame.pack(pady=10)
        
        self.display_lbl = tk.Label(self.img_frame, text="Select an image or upload one.", bg="#f0f0f0", fg="#888", font=label_font)
        self.display_lbl.pack(expand=True, fill=tk.BOTH)
        
        # Control Buttons
        btn_frame = tk.Frame(self.main_frame, bg="white")
        btn_frame.pack(pady=10)
        
        self.btn_select = tk.Button(btn_frame, text="📂 Image", command=self.select_image, font=btn_font, bg="#007bff", fg="white", width=15)
        self.btn_select.grid(row=0, column=0, padx=10)
        
        self.btn_analyze = tk.Button(btn_frame, text="⚡ Analyze Current", command=self.analyze, font=btn_font, bg="#28a745", fg="white", width=15)
        self.btn_analyze.grid(row=0, column=1, padx=10)
        
        # Results Section
        res_frame = tk.Frame(self.main_frame, pady=20, bg="white")
        res_frame.pack(fill=tk.X, padx=50)
        
        # Gender Result
        tk.Label(res_frame, text="Predicted Gender:", font=("Helvetica", 12, "bold"), fg="#555", bg="white").grid(row=0, column=0, sticky="w", pady=5)
        self.lbl_gender = tk.Label(res_frame, text="---", font=("Helvetica", 16, "bold"), fg="#007bff", bg="white")
        self.lbl_gender.grid(row=0, column=1, sticky="w", padx=20)
        
        # Age Result
        tk.Label(res_frame, text="Predicted Age:", font=("Helvetica", 12, "bold"), fg="#555", bg="white").grid(row=1, column=0, sticky="w", pady=5)
        self.lbl_age = tk.Label(res_frame, text="---", font=("Helvetica", 16, "bold"), fg="#28a745", bg="white")
        self.lbl_age.grid(row=1, column=1, sticky="w", padx=20)
        
        self.current_image = None



    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.load_image(file_path)
            self.analyze()

    def load_image(self, path):
        try:
            # Open with PIL
            pil_img = Image.open(path).convert("RGB")
            self.current_image = pil_img
            
            # Resize for display (keep aspect ratio)
            display_img = pil_img.copy()
            # Calculate ratio to fit in 500x400
            display_img.thumbnail((480, 380))
            
            # Convert to Tkinter Image
            self.tk_img = ImageTk.PhotoImage(display_img)
            
            # Update Label
            self.display_lbl.config(image=self.tk_img, text="")
            
            # Clear previous results until analyzed
            self.lbl_gender.config(text="Analyzing...")
            self.lbl_age.config(text="Analyzing...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def analyze(self):
        if self.current_image is None:
            # messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            # Run prediction
            gender_str, age_str = analyze_handwriting(self.current_image)
            
            # Update GUI
            self.lbl_gender.config(text=gender_str)
            self.lbl_age.config(text=age_str)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()