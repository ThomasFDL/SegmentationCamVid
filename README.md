# SegmentationCamVid

[![Python](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)
[![HuggingFace](https://shields.io)](https://huggingface.co)
[![CoreML](https://shields.io)](https://apple.com)

**SegmentationCamVid** est un petit projet personnel que j'ai développé pour découvrir le domaine de la segmentation sémantique. L'objectif est de classifier chaque pixel d'une scène routière en temps réel parmi 32 classes (route, trottoir, piéton, voiture, etc.) en utilisant le jeu de données **CamVid**.

Le projet s'appuie sur l'architecture de Transformer **SegFormer (mit-b3)**. Pour gérer la disproportionnalité entre les classes, j'ai implémenté une fonction de perte combinée (**Dice Loss + Cross-Entropy Loss**).

---

## Fonctionnalités
* **Modèle :** Architecture SegFormer-B3 (mit-b3) adaptée pour 32 classes.
* **Loss Hybride :** `ComboDiceCrossEntropyLoss` pour forcer la précision sur les frontières inter-classes.
* **Interface Vidéo :** Un script dédié pour tester visuellement le modèle sur des fichiers vidéo ou photo.
* **Portage pour accélerer l'inférence :** Modèle optimisé et converti au format CoreML pour les appareils Apple.

---

## Téléchargement des Modèles Éduqués

Pas besoin de relancer l'entraînement complet pour tester le projet ! Je mets à disposition mes deux meilleurs modèles entraînés. Prenez le modèle CoreML si vous avez un Macbook avec une puce M1,M2,M3,m4 ou M5, et l'autre sinon.

* 💾 [Télécharger le modèle PyTorch (.pth ou HuggingFace format) 🔗](https://google.com)
* 🍏 [Télécharger le modèle optimisé CoreML (.mlpackage) 🔗](https://google.com)

*Une fois téléchargé, placez le modèle dans un dossier nommé `model` à la racine du projet.*

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
Installez d'abord les dépendances nécessaires au projet :
```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle (`train.py`)
Pour entraîner le modèle à partir de zéro :
```bash
python train.py
```

### 3. Évaluer sur le jeu de test (`test_models.py`)
Pour calculer précisément les performances et le temps d'inférence des deux modèles sur les données de test officielles de CamVid :
```bash
python test_models.py
```

### 4. Tester sur des vidéos (`interface.py`)
J'ai développé une petite interface graphique / script pour charger une vidéo de votre choix et afficher la segmentation sémantique en direct :
```bash
python interface.py
```

---

## Mes Résultats d'Apprentissage (mIoU)


---

## 🎓 Ce que ce projet m'a appris
* L'utilisation de transformers comme modèle de segmentation sémantique
* L'utilisation de fonctions de pertes combinées 
* Le suivi de l'entraînement pour éviter l'overfitting.
* L'importance du portage pour limiter l'inférence 
