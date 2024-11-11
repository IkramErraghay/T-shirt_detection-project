# T-shirt Detection Using Mask R-CNN

## Description du Projet
Ce projet implémente un système de détection et de segmentation de T-shirts utilisant Mask R-CNN. 
re entrainer les head layers du coco model


### Préparation de l'Environnement

Pour assurer une reproduction fiable de notre environnement de développement et vu les probèmes de compatibilité, nous avons mis en place un environnement Docker incluant toutes les dépendances nécessaires à Mask R-CNN avec les versions spécifiquese. Cette approche nous permet de garantir la cohérence des versions des bibliothèques et des dépendances.
Nous avons configuré un environnement avec les versions suivantes :

- Python 3.5.4
- TensorFlow 1.5.0
- Keras 2.1.5
- Mask R-CNN (Implémentation Matterport)

1. **Construction de l'image Docker**
```bash
docker build -t tshirt-detection .
```

2. **Lancement du conteneur**
```bash
docker run -it mask-rcnn-python35
```

le Dockerfile gère automatiquement l'installation de :
- Les packages Python requis
- La bibliothèque Mask R-CNN
- Le téléchargement du modèle COCO
- Les dépendances du projet

### Dataset
- **Taille** : 202 images de T-shirts
- **Répartition** :
   Entraînement : 162 images (80%)
   Validation : 40 images (20%)
- **Format** : Masques polygonaux pour une segmentation précise 
- **Structure** : Images labélisées avec des masques polygonaux

## Structure du Projet
```
project/
├── dataset.py          # Gestion du dataset et configuration du modèle  (`CustomConfig`)
├── train.py           # Script d'entraînement
├── inference.py       # Script pour les prédictions
├── dataset/
│   ├── images/        # Images d'entraînement
│   └── annotations/   # Fichiers d'annotation JSON
└── logs/             # Logs d'entraînement et modèles sauvegardés
```

1. Les images et annotations sont chargées via `dataset.py` et on a la génération des masques pour la segmentation
2. Le Chargement et division du dataset (80% train, 20% validation) et l'entraînement est géré par `train.py`
3. Le Chargement du modèle entraîné et les prédictions sur nouvelles images sont faites via `inference.py`

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



