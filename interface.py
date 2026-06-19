import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import torch
import cv2  
from PIL import Image, ImageTk
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ==========================================
# CONFIGURATION LOCALE 
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
        
        # Détection du matériel (Optimisé spécifiquement pour le Mac M4)
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
                    tk.Label(item_frame, text=row[name_key], font=("Helvetica", 8), bg="#f0f0f0", fg="black").pack(side="left")

    # ==========================================
    # LOGIQUE DE PRÉDICTION CORE (MODIFIÉE)
    # ==========================================
    def predire_frame(self, image_pil):
        """Prend une image PIL, exécute le modèle avec MPS Autocast (FP16) et renvoie le masque PIL."""
        width, height = image_pil.size
        
        # Le processeur d'image crée un tenseur initialement en FP32
        inputs = self.processor(images=image_pil, return_tensors="pt")
        
        # Transfert des pixels sur le GPU MPS
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inférence avec optimisation MPS FP16 automatique
        with torch.no_grad():
            if self.device.type == "mps":
                with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
                
            logits = outputs.logits

        # Redimensionnement des logits à la taille originale de l'image
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        
        # Extraction de la classe dominante par pixel
        prediction_indices = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        prediction_indices = np.clip(prediction_indices, 0, NUM_CLASSES - 1)
        
        # Coloration du masque
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
        
        # Lancer la prédiction accélérée
        mask_pil = self.predire_frame(img_org)
        mask_resized = mask_pil.copy()
        mask_resized.thumbnail((450, 400))
        
        img_tk_seg = ImageTk.PhotoImage(mask_resized)
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

    def arreter_video(self):
        self.is_playing_video = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

    # ==========================================
    # MODE VIDÉO EN DIRECT (LIVE STREAMING)
    # ==========================================
    # ==========================================
    # MODE VIDÉO OPTIMISÉ (PRE-CALCUL EN LOCAL)
    # ==========================================
    def ouvrir_video(self):
        self.arreter_video()
        video_path = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov")])
        if not video_path: return

        # 1. Préparer les chemins pour la vidéo temporaire segmentée
        self.video_temp_predite = "./temp_video_output.mp4"
        
        # 2. Désactiver l'interface et afficher un message d'attente
        self.lbl_index.configure(text="Traitement de la vidéo...")
        self.panel_origin.configure(image="", text="Génération de la vidéo en cours...\nVeuillez patienter.")
        self.panel_seg.configure(image="", text="Calcul de l'inférence SegFormer...\nCette étape peut prendre 1 à 2 minutes.")
        self.status_label.configure(text="Inférence globale lancée sur le GPU/CPU...")
        
        # Bloquer les boutons pour éviter les doubles clics pendant le calcul
        self.btn_select_img.configure(state="disabled")
        self.btn_select_vid.configure(state="disabled")
        
        # Forcer Tkinter à mettre à jour l'affichage du texte avant de figer pour le calcul
        self.root.update()

        # 3. Lancer le calcul complet de la vidéo
        self.pre_calculer_video_complete(video_path, self.video_temp_predite)

        # 4. Réactiver l'interface et lancer la lecture fluide côte à côte
        self.btn_select_img.configure(state="normal")
        self.btn_select_vid.configure(state="normal")
        
        # Ouvrir les deux flux de lecture (la vidéo d'origine et la vidéo prédite)
        self.video_cap_org = cv2.VideoCapture(video_path)
        self.video_cap_seg = cv2.VideoCapture(self.video_temp_predite)
        
        # Récupérer les FPS d'origine pour caler la vitesse de lecture (ex: 30 FPS = ~33ms entre chaque frame)
        fps = self.video_cap_org.get(cv2.CAP_PROP_FPS)
        self.vitesse_lecture = int(1000 / fps) if fps > 0 else 33
        
        self.is_playing_video = True
        self.lbl_index.configure(text="Lecture Fluide (Pré-calculée)")
        
        # Lancer la boucle de lecture synchronisée
        self.lire_videos_synchronisees()

    def pre_calculer_video_complete(self, chemin_entree, chemin_sortie):
        """Parcourt toute la vidéo en amont pour générer le fichier de segmentation."""
        cap = cv2.VideoCapture(chemin_entree)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        
        # Encodeur vidéo pour sauvegarder les masques colorés
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chemin_sortie, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 10 == 0 or frame_count == total_frames:
                self.status_label.configure(text=f"Inférence : Image {frame_count}/{total_frames} calculée...")
                self.root.update()

            # Inférence standard
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil_org = Image.fromarray(frame_rgb)
            img_pil_seg = self.predire_frame(img_pil_org)
            
            # Reconversion en BGR pour l'écriture OpenCV
            frame_seg_bgr = cv2.cvtColor(np.array(img_pil_seg), cv2.COLOR_RGB2BGR)
            out.write(frame_seg_bgr)
            
        cap.release()
        out.release()

    def lire_videos_synchronisees(self):
        """Lit les deux fichiers vidéos en même temps, sans faire d'inférence (ultra fluide)."""
        if not self.is_playing_video or self.video_cap_org is None or self.video_cap_seg is None:
            return

        ret_org, frame_org = self.video_cap_org.read()
        ret_seg, frame_seg = self.video_cap_seg.read()

        # Si l'une des vidéos arrive à la fin, on arrête
        if not ret_org or not ret_seg:
            self.status_label.configure(text="Fin de la vidéo pré-calculée.")
            self.arreter_video()
            return

        # 1. Affichage de la frame originale à gauche
        frame_org_rgb = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        img_org = Image.fromarray(frame_org_rgb)
        img_org.thumbnail((450, 400))
        img_tk_org = ImageTk.PhotoImage(img_org)
        self.panel_origin.configure(image=img_tk_org)
        self.panel_origin.image = img_tk_org

        # 2. Affichage de la frame segmentée pré-calculée à droite
        frame_seg_rgb = cv2.cvtColor(frame_seg, cv2.COLOR_BGR2RGB)
        img_seg = Image.fromarray(frame_seg_rgb)
        img_seg.thumbnail((450, 400))
        img_tk_seg = ImageTk.PhotoImage(img_seg)
        self.panel_seg.configure(image=img_tk_seg)
        self.panel_seg.image = img_tk_seg

        self.status_label.configure(text="Lecture en cours...")
        
        # Cadencer la fonction sur les vrais FPS de la vidéo au lieu de 1ms
        self.root.after(self.vitesse_lecture, self.lire_videos_synchronisees)

    def arreter_video(self):
        self.is_playing_video = False
        if hasattr(self, 'video_cap_org') and self.video_cap_org:
            self.video_cap_org.release()
            self.video_cap_org = None
        if hasattr(self, 'video_cap_seg') and self.video_cap_seg:
            self.video_cap_seg.release()
            self.video_cap_seg = None
            
        # Supprimer le fichier vidéo temporaire pour ne pas encombrer le disque
        if hasattr(self, 'video_temp_predite') and os.path.exists(self.video_temp_predite):
            try: os.remove(self.video_temp_predite)
            except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    # Gérer la fermeture de la fenêtre pour couper proprement la vidéo
    root.protocol("WM_DELETE_WINDOW", lambda: [app.arreter_video(), root.destroy()])
    root.mainloop()