# SegmentationCamVid


**SegmentationCamVid** est un petit projet personnel que j'ai développé pour découvrir le domaine de la segmentation sémantique. L'objectif est de classifier chaque pixel d'une scène routière en temps réel parmi 32 classes (route, trottoir, piéton, voiture, etc.) en utilisant le jeu de données **CamVid**.

Le projet s'appuie sur l'architecture de Transformer **SegFormer (mit-b1)**. Pour gérer la disparité entre les classes, j'ai implémenté une fonction de perte combinée (**Dice Loss + Focal Loss**). J'obtiens un mIoU de 43,85% sur le dataset de test, notamment à cause de certaines classes qui sont trop peu présentes dans le dataset pour pouvoir les segmenter correctement. 

---

## Fonctionnalités
* **Modèle :** Architecture SegFormer-B1 (mit-1) adaptée pour 32 classes.
* **Loss Hybride :** Dice loss + Focal loss pour forcer la précision sur les classes peu représentées.
* **Interface Vidéo :** Un script dédié pour tester visuellement le modèle sur des fichiers vidéo ou photo.
* **Portage pour accélerer l'inférence :** Modèle optimisé et converti au format CoreML pour les appareils Apple.

---

## Téléchargement des Modèles Entrainés

Pas besoin de relancer l'entraînement complet pour tester le projet ! Je mets à disposition mes deux meilleurs modèles entraînés. Prenez le modèle CoreML si vous avez un Macbook avec une puce M1,M2,M3,m4 ou M5, et l'autre sinon.

* [Télécharger le modèle PyTorch](https://drive.google.com/file/d/1RcA1CBHiVqKSDdVP2KQZ0KgyndXYsyyU/view?usp=sharing)
* [Télécharger le modèle optimisé CoreML (.mlpackage)](https://drive.google.com/file/d/16L7ud9_b-NajKaw4J3ngNHA_Mu2xiLy5/view?usp=drive_link)
  
*Une fois téléchargé et décompressé, placez le modèle à la racine du projet.
---

## Structure du Projet

```text
├── CamVid/              # Le dataset (train, val, test, class_dict.csv)
├── src/
│   ├── dataset.py       # Dataset PyTorch & pipeline d'augmentation
│   ├── model.py         # Configuration mit-b3 et custom Trainer
│   └── utils.py         # Métriques d'évaluation
├── train.py             # Script pour lancer l'entraînement
├── test_models.py       # Script d'évaluation quantitative sur le dataset de test
├── interface.py         # Petite interface utilisateur pour tester sur des vidéos
└── README.md
```

---

##  Utilisation

### 1. Installation
Télechargez le dataset CamVid en cliquant [ici](https://www.kaggle.com/datasets/carlolepelaars/camvid).

Commencez par cloner le dépôt :
```bash
git clone https://github.com/ThomasFDL/SegmentationCamVid.git
```
Installez ensuite les dépendances nécessaires au projet :
```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle (`train.py`)
Pour entraîner le modèle et générer les courbes d'apprentissages :
```bash
python train.py
```

### 3. Évaluer sur le jeu de test (`test_models.py`)
Pour calculer le mIou et l'IoU par classe sur les données de test de CamVid :
```bash
python test_models.py
```

### 4. Tester sur des vidéos (`interface.py`)
J'ai développé une petite interface graphique pour tester le modèle sur une vidéo et afficher la segmentation sémantique en direct :
```bash
python interface.py
```

---

## Mes Résultats

<img width="640" height="358" alt="démo" src="https://github.com/user-attachments/assets/fa9c1d56-5ab1-406f-b031-b9515e52742d" />


