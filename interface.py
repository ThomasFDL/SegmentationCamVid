import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog  
import numpy as np
import torch
import cv2  
from PIL import Image, ImageTk
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import coremltools as ct  
import time


# =================================
#    INTERFACE CODÉE PAR UNE IA
# =================================


CSV_FILE_PATH = "./CamVid/class_dict.csv"
MODEL_PATH = "./model" 
COREML_PATH = "./model_CoreML.mlpackage"
NUM_CLASSES = 32  

class SelectionDialog(simpledialog.Dialog):
    """Boîte de dialogue simplifiée pour choisir l'architecture d'inférence."""
    def body(self, master):
        self.title("Sélection du Modèle")
        tk.Label(master, text="Choisissez l'architecture à utiliser :", font=("Helvetica", 11, "bold")).pack(pady=10)
        self.choice = tk.StringVar(value="normal")
        tk.Radiobutton(master, text="Modèle Normal (PyTorch MPS)", variable=self.choice, value="normal", font=("Helvetica", 10)).pack(anchor="w", padx=20, pady=5)
        tk.Radiobutton(master, text="Modèle Réduit (CoreML Neural Engine)", variable=self.choice, value="coreml", font=("Helvetica", 10)).pack(anchor="w", padx=20, pady=5)
        return master

    def apply(self):
        self.result = self.choice.get()

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CamVid Segmentation - Mode Local Image & Vidéo")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        dialog = SelectionDialog(self.root)
        self.model_type = dialog.result if dialog.result else "normal"
        
        print(f"Option sélectionnée : {self.model_type.upper()}")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if self.model_type == "coreml":
            print("⚡ Chargement du modèle réduit CoreML...")
            if not os.path.exists(COREML_PATH):
                messagebox.showerror("Erreur", f"Modèle introuvable à : {COREML_PATH}\nBascule sur PyTorch.")
                self.model_type = "normal"
            else:
                self.model = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.CPU_AND_GPU)
                print("✅ Modèle CoreML chargé avec succès.")

        if self.model_type == "normal":
            print("Chargement du modèle SegFormer PyTorch (MiT-B3)...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
            self.model.eval()
            self.model.to(self.device)
            print(f"✅ Modèle PyTorch chargé sur : {self.device}")

        self.class_colors = self.load_class_colors(CSV_FILE_PATH)

        self.filepaths = []
        self.current_index = 0
        self.video_cap = None
        self.is_playing_video = False

        # --- CONSTRUTION DE L'INTERFACE GRAPHIQUE (UI) ---
        label_text = f"Segmentation CamVid - Mode [{self.model_type.upper()}]"
        title = tk.Label(root, text=label_text, font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=10)

        self.frame_buttons = tk.Frame(root, bg="#f0f0f0")
        self.frame_buttons.pack(pady=5)
        
        self.btn_select_img = tk.Button(self.frame_buttons, text="Mode Photos", command=self.ouvrir_images, font=("Helvetica", 11), bg="#007bff", fg="black", padx=10, pady=5)
        self.btn_select_img.pack(side="left", padx=10)
        
        self.btn_select_vid = tk.Button(self.frame_buttons, text="Mode Vidéo Live", command=self.ouvrir_video, font=("Helvetica", 11), bg="#28a745", fg="black", padx=10, pady=5)
        self.btn_select_vid.pack(side="left", padx=10)

        self.frame_nav = tk.Frame(root, bg="#f0f0f0")
        self.frame_nav.pack(pady=5)
        self.btn_prev = tk.Button(self.frame_nav, text="◀ Précédent", command=self.image_precedente, state="disabled")
        self.btn_prev.pack(side="left", padx=10)
        self.lbl_index = tk.Label(self.frame_nav, text="Aucun fichier chargé", bg="#f0f0f0")
        self.lbl_index.pack(side="left", padx=10)
        self.btn_next = tk.Button(self.frame_nav, text="Suivant ▶", command=self.image_suivante, state="disabled")
        self.btn_next.pack(side="left", padx=10)

        self.main_container = tk.Frame(root, bg="#f0f0f0")
        self.main_container.pack(expand=True, fill="both", padx=20, pady=10)

        self.frame_images = tk.Frame(self.main_container, bg="#f0f0f0")
        self.frame_images.pack(side="left", expand=True, fill="both")

        self.panel_origin = tk.Label(self.frame_images, text="Original", bg="#e0e0e0", borderwidth=1, relief="solid")
        self.panel_origin.pack(side="left", expand=True, padx=10, fill="both")

        self.panel_seg = tk.Label(self.frame_images, text="Segmentation", bg="#e0e0e0", borderwidth=1, relief="solid")
        self.panel_seg.pack(side="right", expand=True, padx=10, fill="both")

        self.charger_legende_classes()

        self.status_label = tk.Label(root, text=f"Prêt - Moteur d'inférence : {self.model_type.upper()}", bd=1, relief="sunken", anchor="w", bg="#e0e0e0", fg="#555")
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
                    tk.Label(item_frame, text=row[name_key], font=("Helvetica", 8), bg="#f0f0f0", fg="black").pack(side="left")

    def predire_frame(self, image_pil):
        """Prend une image PIL, gère l'inférence selon le modèle sélectionné et renvoie le masque coloré PIL."""
        width, height = image_pil.size
        
        # ------------------------------------------
        # PIPELINE BRANCHÉ SUR LE MOTEUR NATIVE COREML
        # ------------------------------------------
        if self.model_type == "coreml":
            inputs = self.processor(images=image_pil, return_tensors="np")
            np_images = inputs["pixel_values"]
            
            predictions = self.model.predict({"pixel_values": np_images})
            output_key = list(predictions.keys())[0] 
            logits_np = predictions[output_key].squeeze(0)
            
            logits_resized = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
            for c in range(NUM_CLASSES):
                logits_resized[c] = cv2.resize(logits_np[c], (width, height), interpolation=cv2.INTER_LINEAR)
            
            prediction_indices = np.argmax(logits_resized, axis=0)
            
        # ------------------------------------------
        # PIPELINE BRANCHÉ SUR PYTORCH MPS (NORMAL)
        # ------------------------------------------
        else:
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.device.type == "mps":
                    with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
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
    # MODE IMAGES STATIQUES
    # ==========================================
    def ouvrir_images(self):
        self.arreter_video()
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not files: return
        self.filepaths = list(files)
        self.current_index = 0
        self.charger_image_actuelle()

    def arreter_video(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.panel_origin.configure(image="", text="Original")
        self.panel_seg.configure(image="", text="Segmentation")

    def image_precedente(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.charger_image_actuelle()

    def image_suivante(self):
        if self.current_index < len(self.filepaths) - 1:
            self.current_index += 1
            self.charger_image_actuelle()

    def charger_image_actuelle(self):
        if not self.filepaths: return
        path = self.filepaths[self.current_index]
        self.lbl_index.configure(text=f"Image {self.current_index + 1} / {len(self.filepaths)}")
        
        self.btn_prev.configure(state="normal" if self.current_index > 0 else "disabled")
        self.btn_next.configure(state="normal" if self.current_index < len(self.filepaths) - 1 else "disabled")
        
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize((550, 350))
        
        mask_color_pil = self.predire_frame(img_pil)
        mask_resized = mask_color_pil.resize((550, 350))
        
        tk_img_origin = ImageTk.PhotoImage(img_resized)
        tk_img_seg = ImageTk.PhotoImage(mask_resized)
        
        self.panel_origin.configure(image=tk_img_origin, text="")
        self.panel_origin.image = tk_img_origin
        self.panel_seg.configure(image=tk_img_seg, text="")
        self.panel_seg.image = tk_img_seg
        self.status_label.configure(text=f"Fichier affiché : {os.path.basename(path)}")

    # ==========================================
    # MODE VIDÉO LIVE CADENCÉ SUR L'INFÉRENCE
    # ==========================================

    def statut_boutons_navigation(self, state):
        self.btn_prev.configure(state=state)
        self.btn_next.configure(state=state)



    # ==========================================
    # MODE VIDÉO LIVE : TEMPS RÉEL AVEC SAUT DYNAMIQUE
    # ==========================================
    def ouvrir_video(self):
        self.arreter_video()
        file = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv")])
        if not file: return
        
        self.video_cap = cv2.VideoCapture(file)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0  # Sécurité par défaut
        
        self.is_playing_video = True
        self.statut_boutons_navigation("disabled")
        self.lbl_index.configure(text="Lecture Vidéo en Temps Réel (Saut dynamique)...")
        
        # Enregistrement du moment précis où la vidéo commence à jouer
        self.start_time = time.time()
        self.frame_count = 0
        
        self.update_video_stream()

    def update_video_stream(self):
        if not self.is_playing_video or self.video_cap is None: return
        
        # 1. Calcul du temps écoulé réel depuis le début du clic "Play"
        elapsed_time = time.time() - self.start_time
        
        # 2. Déduction de la frame théorique qui DEVRAIT être affichée à cet instant précis
        target_frame = int(elapsed_time * self.fps)
        
        # 3. LE COEUR DU SKIPPING : Si l'inférence a été trop longue, on saute les frames en retard
        # cap.grab() est bcp plus rapide que cap.read() car il ne décompresse pas l'image en RAM
        while self.frame_count < target_frame:
            ret_grab = self.video_cap.grab()
            if not ret_grab: break  # Fin de vidéo pendant le saut
            self.frame_count += 1
            
        # 4. On lit enfin la bonne frame sur laquelle on a rattrapé le retard
        ret, frame = self.video_cap.read()
        if not ret:
            self.arreter_video()
            self.status_label.configure(text="Fin de la vidéo.")
            return
            
        self.frame_count += 1
        
        # --- PANNEAU GAUCHE : Original Pur ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_resized = img_pil.resize((550, 350))
        tk_img_origin = ImageTk.PhotoImage(img_resized)
        
        # --- PANNEAU DROIT : Inférence Réelle ---
        mask_color_pil = self.predire_frame(img_pil)
        mask_resized = mask_color_pil.resize((550, 350))
        tk_img_seg = ImageTk.PhotoImage(mask_resized)
        
        # Mise à jour immédiate de l'affichage Tkinter
        self.panel_origin.configure(image=tk_img_origin, text="")
        self.panel_origin.image = tk_img_origin
        
        self.panel_seg.configure(image=tk_img_seg, text="")
        self.panel_seg.image = tk_img_seg
        
        # Calcul du nombre de frames sautées à afficher dans la barre de statut pour info
        frames_sautees = target_frame - self.frame_count + 1
        self.status_label.configure(
            text=f"Frame : {self.frame_count} | Retard rattrapé : {max(0, frames_sautees)} frames"
        )
        
        # On relance la boucle immédiatement (1 ms) pour calculer le nouveau décalage temporel
        self.root.after(1, self.update_video_stream)


    def arreter_video(self):
        """Arrête la vidéo et libère proprement le flux de capture."""
        self.is_playing_video = False
        if hasattr(self, 'video_cap') and self.video_cap:
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = None
        self.panel_origin.configure(image="", text="Original")
        self.panel_seg.configure(image="", text="Segmentation")


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
