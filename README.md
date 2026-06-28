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

* [Télécharger le modèle PyTorch (.pth ou HuggingFace format) 🔗](https://google.com)
* [Télécharger le modèle optimisé CoreML (.mlpackage) 🔗](https://google.com)

*Une fois téléchargé, placez le modèle dans un dossier nommé `model` à la racine du projet pour le modèle pytorch, ou laissez le tel quel pour le modèle CoreML.*

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
git clone https://github.com/h2p6smyjkv-ops/SegmentationCamVid.git
```
Installez ensuite les dépendances nécessaires au projet :
```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle (`train.py`)
Pour entraîner le modèle à partir de zéro :
```bash
python train.py
```

### 3. Évaluer sur le jeu de test (`test_models.py`)
Pour calculer les performances des deux modèles sur les données de test de CamVid :
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

<img width="1450" height="747" alt="ExempleSegmentation" src="https://github.com/user-attachments/assets/1cc80ae9-4783-4423-be16-4e82087ceed2" />



---

## Ce que ce projet m'a appris
* L'utilisation de transformers comme modèle de segmentation sémantique.
* L'utilisation de fonctions de pertes combinées. 
* L'importance de la data augmentation pour les petits datasets.
* L'importance du choix des hyperparamètres.
* L'importance du portage pour limiter l'inférence.
* La maitrise de librairies comme pytorch, transformers, albumentations.
