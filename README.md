# T-shirt Detection Using Mask R-CNN

## Description du Projet
Ce projet implémente un système de détection et de segmentation de T-shirts utilisant Mask R-CNN. 

## Structure du Projet
```
project/
│
├── dataset.py          # Définition du dataset personnalisé
├── train.py           # Script d'entraînement du modèle
├── inference.py       # Script pour les prédictions
├── visualize.py       # Outils de visualisation
├── Dockerfile         # Configuration Docker
├── .gitignore        # Configuration Git
│
└── dataset/
    ├── images/        # Images d'entraînement
    └── annotations.json # Annotations des images
```

## Configuration Technique

### Prérequis
- Python 3.x
- TensorFlow
- Mask R-CNN
- Docker

### Configuration de l'Environnement
Le projet utilise Docker pour assurer une configuration cohérente. Pour construire l'image :

```bash
docker build -t tshirt-detection .
```

### Dataset
- **Taille** : 202 images de T-shirts
- **Format** : Annotations au format JSON par VIA pour l'annotation 
- **Structure** : Images labélisées avec des masques polygonaux

## Hyperparamètres et Configuration

### Configuration 1 du Modèle
```python
IMAGES_PER_GPU = 1
NUM_CLASSES = 1 + 1  # Fond + T-shirt
STEPS_PER_EPOCH = 50
IMAGE_MAX_DIM = 256
IMAGE_MIN_DIM = 256
```

### Hyperparamètres d'Entraînement
- **Learning Rate** : 0.01
- **Optimiseur** : Adam
- **Epochs** : 5
- **Layers** : "heads" (uniquement les couches de tête)

## Entraînement

### Procédure
1. Chargement des poids pré-entraînés COCO
2. Fine-tuning des couches de tête
3. Entraînement sur le dataset personnalisé

### Résultats
  image
- Loss d'entraînement
- Loss de validation
- Matrice de confusion
- Images de test avec prédictions

## Utilisation

### Entraînement
```bash
python train.py
```

### Inférence
```bash
python inference.py --image path/to/image
```



