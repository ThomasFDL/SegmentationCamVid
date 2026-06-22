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

# ==========================================
# CONFIGURATION LOCALE 
# ==========================================
CSV_FILE_PATH = "./CamVid/class_dict.csv"
MODEL_PATH = "./model" 
COREML_PATH = "./model_CoreML.mlpackage"
NUM_CLASSES = 32  

class SelectionDialog(simpledialog.Dialog):
    """Boîte de dialogue personnalisée pour choisir le modèle au démarrage."""
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

        # --- SÉLECTION DU MODÈLE AU DÉMARRAGE ---
        dialog = SelectionDialog(self.root)
        self.model_type = dialog.result if dialog.result else "normal"
        
        print(f"Option sélectionnée : {self.model_type.upper()}")
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")
        
        # Détection du matériel principal
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if self.model_type == "coreml":
            print("⚡ Chargement du modèle réduit CoreML sur le Neural Engine...")
            if not os.path.exists(COREML_PATH):
                messagebox.showerror("Erreur", f"Le modèle CoreML est introuvable à l'adresse :\n{COREML_PATH}\n\nBascule automatique sur le modèle Normal.")
                self.model_type = "normal"
            else:
                # Chargement du modèle Apple avec allocation matérielle optimisée
                self.model = ct.models.MLModel(COREML_PATH, compute_units=ct.ComputeUnit.CPU_AND_GPU)
                print("✅ Modèle CoreML chargé avec succès.")

        if self.model_type == "normal":
            print("Chargement du modèle SegFormer PyTorch (MiT-B3)...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
            self.model.eval()
            self.model.to(self.device)
            print(f"✅ Modèle PyTorch chargé sur le périphérique : {self.device}")

        # --- CHARGEMENT DE LA PALETTE DE COULEURS ---
        self.class_colors = self.load_class_colors(CSV_FILE_PATH)

        # --- VARIABLES DE CONTRÔLE ---
        self.filepaths = []
        self.current_index = 0
        self.video_cap = None
        self.is_playing_video = False

        # --- INTERFACE GRAPHIQUE (UI) ---
        label_text = f"Segmentation CamVid en Temps Réel - Mode [{self.model_type.upper()}]"
        title = tk.Label(root, text=label_text, font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333")
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

    # ==========================================
    # LOGIQUE DE PRÉDICTION CORE (ADAPTÉE PARTICULIÈREMENT POUR COREML)
    # ==========================================
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
            
            # 🌟 CORRECTION ICI : On prend le premier élément [0] de la liste de clés
            output_key = list(predictions.keys())[0] 
            
            # Maintenant output_key est un texte propre (ex: "linear_104") et non une liste
            logits_np = predictions[output_key].squeeze(0) # Forme : (32, 128, 128)
            
            # --- ÉTAPE DE LISSAGE ANTI-PIXELISATION ---
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

            # Redimensionnement lourd via interpolation PyTorch
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            prediction_indices = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Coloration finale commune du masque
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
        
        # Appel de la fonction de prédiction commune
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
    # MODE VIDÉO LIVE 
    # ==========================================
    def ouvrir_video(self):
        self.arreter_video()
        file = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv")])
        if not file: return
        
        # 1. Fenêtre de progression
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Traitement de la vidéo...")
        progress_win.geometry("400x150")
        progress_win.transient(self.root)
        progress_win.grab_set()
        
        lbl_prog = tk.Label(progress_win, text="Initialisation du traitement...", font=("Helvetica", 10))
        lbl_prog.pack(pady=15)
        
        progress = ttk.Progressbar(progress_win, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=10)
        self.root.update()

        # 2. Ouverture de la vidéo source
        cap = cv2.VideoCapture(file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Chemins des deux fichiers de sortie temporaires
        self.output_video_origin = "./temp_video_origin.mp4"
        self.output_video_seg = "./temp_video_seg.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_origin = cv2.VideoWriter(self.output_video_origin, fourcc, fps, (width, height))
        out_seg = cv2.VideoWriter(self.output_video_seg, fourcc, fps, (width, height))
        
        print("Démarrage du traitement de la vidéo...")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Écriture immédiate de la frame originale pure pour le panneau gauche
            out_origin.write(frame)
            
            # Prédiction du masque pour le panneau droit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            mask_color_pil = self.predire_frame(img_pil)
            
            # Conversion du masque coloré pur en BGR pour OpenCV
            frame_seg_out = cv2.cvtColor(np.array(mask_color_pil), cv2.COLOR_RGB2BGR)
            out_seg.write(frame_seg_out)
            
            # Progression
            frame_count += 1
            pct = int((frame_count / total_frames) * 100)
            progress['value'] = pct
            lbl_prog.config(text=f"Traitement de la frame {frame_count} / {total_frames} ({pct}%)")
            if frame_count % 10 == 0:
                progress_win.update()

        cap.release()
        out_origin.release()
        out_seg.release()
        progress_win.destroy()

        # 3. Chargement des deux flux synchronisés pour la lecture fluide
        self.cap_origin = cv2.VideoCapture(self.output_video_origin)
        self.cap_seg = cv2.VideoCapture(self.output_video_seg)
        self.is_playing_video = True
        self.statut_boutons_navigation("disabled")
        
        self.video_delay = int(1000 / fps) if fps > 0 else 33
        self.update_video_stream()

    def statut_boutons_navigation(self, state):
        self.btn_prev.configure(state=state)
        self.btn_next.configure(state=state)

    def update_video_stream(self):
        if not self.is_playing_video or not self.cap_origin or not self.cap_seg: return
        
        ret_origin, frame_origin = self.cap_origin.read()
        ret_seg, frame_seg = self.cap_seg.read()
        
        # Si une des deux vidéos se termine, on arrête proprement
        if not ret_origin or not ret_seg:
            self.arreter_video()
            self.status_label.configure(text="Fin de la vidéo.")
            
            # Nettoyage des deux fichiers temporaires sur le disque
            for path in [self.output_video_origin, self.output_video_seg]:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
            return
            
        # --- PANNEAU GAUCHE : Original Pur ---
        rgb_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
        pil_origin = Image.fromarray(rgb_origin).resize((550, 350))
        tk_img_origin = ImageTk.PhotoImage(pil_origin)
        
        self.panel_origin.configure(image=tk_img_origin, text="")
        self.panel_origin.image = tk_img_origin
        
        # --- PANNEAU DROIT : Segmentation Pure ---
        rgb_seg = cv2.cvtColor(frame_seg, cv2.COLOR_BGR2RGB)
        pil_seg = Image.fromarray(rgb_seg).resize((550, 350))
        tk_img_seg = ImageTk.PhotoImage(pil_seg)
        
        self.panel_seg.configure(image=tk_img_seg, text="")
        self.panel_seg.image = tk_img_seg
        
        # Boucle de rafraîchissement synchrone
        self.root.after(self.video_delay, self.update_video_stream)

    def arreter_video(self):
        self.is_playing_video = False
        if hasattr(self, 'cap_origin') and self.cap_origin:
            self.cap_origin.release()
            self.cap_origin = None
        if hasattr(self, 'cap_seg') and self.cap_seg:
            self.cap_seg.release()
            self.cap_seg = None


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
