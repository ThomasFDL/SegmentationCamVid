import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import torch
import cv2  # pip install opencv-python
from PIL import Image, ImageTk
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# CONFIGURATION LOCALE (PLUS BESOIN D'API)
# ==========================================
CSV_FILE_PATH = "./CamVid/class_dict.csv"
MODEL_PATH = "./mon_modele_final" 
NUM_CLASSES = 32  

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CamVid Segmentation - Mode Local Image & Vidéo")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # --- CHARGEMENT DU MODÈLE EN LOCAL ---
        print("Chargement du modèle SegFormer (MiT-B3)...")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        self.model.eval()
        
        # Détection du matériel (Utilise le GPU des puces Apple Silicon si disponible)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Modèle chargé sur le périphérique : {self.device}")

        # --- CHARGEMENT DE LA PALETTE DE COULEURS ---
        self.class_colors = self.load_class_colors(CSV_FILE_PATH)

        # --- VARIABLES DE CONTRÔLE ---
        self.filepaths = []
        self.current_index = 0
        self.video_cap = None
        self.is_playing_video = False

        # --- INTERFACE GRAPHIQUE (UI) ---
        title = tk.Label(root, text="Segmentation CamVid en Temps Réel", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=10)

        # Boutons de sélection
        self.frame_buttons = tk.Frame(root, bg="#f0f0f0")
        self.frame_buttons.pack(pady=5)
        
        self.btn_select_img = tk.Button(self.frame_buttons, text="Mode Photos", command=self.ouvrir_images, font=("Helvetica", 11), bg="#007bff", fg="black", padx=10, pady=5)
        self.btn_select_img.pack(side="left", padx=10)
        
        self.btn_select_vid = tk.Button(self.frame_buttons, text="Mode Vidéo Live", command=self.ouvrir_video, font=("Helvetica", 11), bg="#28a745", fg="black", padx=10, pady=5)
        self.btn_select_vid.pack(side="left", padx=10)

        # Navigation multi-photos
        self.frame_nav = tk.Frame(root, bg="#f0f0f0")
        self.frame_nav.pack(pady=5)
        self.btn_prev = tk.Button(self.frame_nav, text="◀ Précédent", command=self.image_precedente, state="disabled")
        self.btn_prev.pack(side="left", padx=10)
        self.lbl_index = tk.Label(self.frame_nav, text="Aucun fichier chargé", bg="#f0f0f0")
        self.lbl_index.pack(side="left", padx=10)
        self.btn_next = tk.Button(self.frame_nav, text="Suivant ▶", command=self.image_suivante, state="disabled")
        self.btn_next.pack(side="left", padx=10)

        # Conteneur principal
        self.main_container = tk.Frame(root, bg="#f0f0f0")
        self.main_container.pack(expand=True, fill="both", padx=20, pady=10)

        # Écrans de visualisation
        self.frame_images = tk.Frame(self.main_container, bg="#f0f0f0")
        self.frame_images.pack(side="left", expand=True, fill="both")

        self.panel_origin = tk.Label(self.frame_images, text="Original", bg="#e0e0e0", borderwidth=1, relief="solid")
        self.panel_origin.pack(side="left", expand=True, padx=10, fill="both")

        self.panel_seg = tk.Label(self.frame_images, text="Segmentation", bg="#e0e0e0", borderwidth=1, relief="solid")
        self.panel_seg.pack(side="right", expand=True, padx=10, fill="both")

        # Légende
        self.charger_legende_classes()

        # Statut
        self.status_label = tk.Label(root, text="Prêt", bd=1, relief="sunken", anchor="w", bg="#e0e0e0", fg="#555")
        self.status_label.pack(side="bottom", fill="x")

    def load_class_colors(self, csv_path):
        colors = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                colors.append([int(row['r']), int(row['g']), int(row['b'])])
        return np.array(colors, dtype=np.uint8)

    def charger_legende_classes(self):
        frame_legend_wrapper = tk.LabelFrame(self.main_container, text=" Légende ", font=("Helvetica", 10, "bold"), bg="#f0f0f0", fg="black")
        frame_legend_wrapper.pack(side="right", fill="y", padx=10)
        canvas = tk.Canvas(frame_legend_wrapper, bg="#f0f0f0", width=150, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame_legend_wrapper, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        if os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = {k.lower(): k for k in reader.fieldnames}
                name_key = headers.get('name', headers.get('class_name', reader.fieldnames))
                for row in reader:
                    hex_color = f"#{int(row['r']):02x}{int(row['g']):02x}{int(row['b']):02x}"
                    item_frame = tk.Frame(scrollable_frame, bg="#f0f0f0")
                    item_frame.pack(fill="x", anchor="w", pady=1)
                    tk.Label(item_frame, width=2, height=1, bg=hex_color, relief="solid").pack(side="left", padx=3)
                    # Modification ici : fg="black" pour forcer l'écriture en noir
                    tk.Label(item_frame, text=row[name_key], font=("Helvetica", 8), bg="#f0f0f0", fg="black").pack(side="left")
    # ==========================================
    # LOGIQUE DE PRÉDICTION CORE
    # ==========================================
    def predire_frame(self, image_pil):
        """Prend une image PIL, exécute le modèle et renvoie le masque PIL."""
        width, height = image_pil.size
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        
        prediction_indices = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        prediction_indices = np.clip(prediction_indices, 0, NUM_CLASSES - 1)
        color_mask = self.class_colors[prediction_indices]
        return Image.fromarray(color_mask)

    # ==========================================
    # MODE IMAGES STATIC
    # ==========================================
    def ouvrir_images(self):
        self.arreter_video()
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not files: return
        self.filepaths = list(files)
        self.current_index = 0
        self.charger_image_actuelle()

    def charger_image_actuelle(self):
        filepath = self.filepaths[self.current_index]
        self.lbl_index.configure(text=f"Photo {self.current_index + 1} / {len(self.filepaths)}")
        self.btn_prev.configure(state="normal" if self.current_index > 0 else "disabled")
        self.btn_next.configure(state="normal" if self.current_index < len(self.filepaths) - 1 else "disabled")

        img_org = Image.open(filepath).convert("RGB")
        img_resized = img_org.copy()
        img_resized.thumbnail((450, 400))
        
        img_tk_org = ImageTk.PhotoImage(img_resized)
        self.panel_origin.configure(image=img_tk_org, text="")
        self.panel_origin.image = img_tk_org

        self.status_label.configure(text="Calcul du masque...")
        self.root.update_idletasks()

        img_seg = self.predire_frame(img_org)
        img_seg.thumbnail((450, 400))
        img_tk_seg = ImageTk.PhotoImage(img_seg)
        self.panel_seg.configure(image=img_tk_seg, text="")
        self.panel_seg.image = img_tk_seg
        self.status_label.configure(text="Prêt")

    def image_precedente(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.charger_image_actuelle()

    def image_suivante(self):
        if self.current_index < len(self.filepaths) - 1:
            self.current_index += 1
            self.charger_image_actuelle()

    # ==========================================
    # MODE VIDÉO EN DIRECT (LIVE STREAMING)
    # ==========================================
    def ouvrir_video(self):
        self.arreter_video()
        video_path = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov")])
        if not video_path: return

        self.video_cap = cv2.VideoCapture(video_path)
        self.is_playing_video = True
        self.lbl_index.configure(text="Lecture Vidéo Live")
        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        
        # Lancer la boucle de lecture
        self.mettre_a_jour_video()

    def mettre_a_jour_video(self):
        if not self.is_playing_video or self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if not ret:
            self.status_label.configure(text="Fin de la vidéo.")
            self.arreter_video()
            return

        # 1. Traitement de la frame originale
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil_org = Image.fromarray(frame_rgb)
        
        # Redimensionnement pour affichage fluide
        # 1. Traitement de la frame originale (suite)
        img_display_org = img_pil_org.copy()
        img_display_org.thumbnail((450, 400))
        img_tk_org = ImageTk.PhotoImage(img_display_org)
        self.panel_origin.configure(image=img_tk_org, text="")
        self.panel_origin.image = img_tk_org

        # 2. Prédiction de la segmentation en direct
        img_pil_seg = self.predire_frame(img_pil_org)
        img_pil_seg.thumbnail((450, 400))
        img_tk_seg = ImageTk.PhotoImage(img_pil_seg)
        self.panel_seg.configure(image=img_tk_seg, text="")
        self.panel_seg.image = img_tk_seg

        self.status_label.configure(text="Lecture et segmentation en cours...")
        
        # Rappeler la fonction immédiatement après 1 milliseconde pour la frame suivante
        self.root.after(1, self.mettre_a_jour_video)

    def arreter_video(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    # Gérer la fermeture de la fenêtre pour couper proprement la vidéo
    root.protocol("WM_DELETE_WINDOW", lambda: [app.arreter_video(), root.destroy()])
    root.mainloop()