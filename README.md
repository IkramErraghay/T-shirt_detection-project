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
2. Le Chargement et division du dataset (80% train, 20% validation) et l'entraînement du model est géré par `train.py`
```bash
python train.py
```
3. Le Chargement du modèle entraîné et les prédictions sur nouvelles images sont faites via `inference.py`
```bash
python inference.py --image path/to/image
```

## Hyperparamètres et Configuration
la première configuration pour l'entraînement du modèle est la suivante:
### Configuration 1 du Modèle
```python
IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Arrière-plan + T-shirt
    STEPS_PER_EPOCH = 100
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    LEARNING_RATE = 0.001
    OPTIMIZER = 'sgd'
    EPOCHS = 10
```
Voici les differents résultats d'entrainement obtenus:

#### 1. Courbe de Perte (Loss)

Le graphe ci-dessous montre les training loss et validation loss.

![Courbe d'entraînement](config1/training_validation_loss.png)
*Figure 1: Évolution des pertes d'entraînement et de validation sur 10 époques*

La perte de validation reste inférieure à la perte d'entraînement donc notre modèle généralise bien sans overfitting.

#### 2. Dernière Ligne des Logs
Les logs complets de l'entraînement sont sauvegardés dans deux fichiers :
- log.txt : Contient tous les détails d'entraînement
- training_logs.txt : Enregistre spécifiquement les losses et validation_losses de chaque époque

Voici les derniers logs d'entraînement (Époque 10) :
```
 93/100 [==========================>...] - ETA: 1:52 - loss: 0.5784 - rpn_class_loss: 0.0215 - rpn_bbox_loss: 0.2328 - mrcnn_class_loss: 0.0207 - mrcnn_bbox_loss: 0.1109 - mrcnn_mask_loss: 0.1926
 94/100 [===========================>..] - ETA: 1:36 - loss: 0.5751 - rpn_class_loss: 0.0213 - rpn_bbox_loss: 0.2312 - mrcnn_class_loss: 0.0206 - mrcnn_bbox_loss: 0.1105 - mrcnn_mask_loss: 0.1916
 95/100 [===========================>..] - ETA: 1:20 - loss: 0.5760 - rpn_class_loss: 0.0211 - rpn_bbox_loss: 0.2332 - mrcnn_class_loss: 0.0204 - mrcnn_bbox_loss: 0.1108 - mrcnn_mask_loss: 0.1904
 96/100 [===========================>..] - ETA: 1:04 - loss: 0.5713 - rpn_class_loss: 0.0209 - rpn_bbox_loss: 0.2310 - mrcnn_class_loss: 0.0202 - mrcnn_bbox_loss: 0.1102 - mrcnn_mask_loss: 0.1890
 97/100 [============================>.] - ETA: 48s - loss: 0.5674 - rpn_class_loss: 0.0208 - rpn_bbox_loss: 0.2292 - mrcnn_class_loss: 0.0201 - mrcnn_bbox_loss: 0.1093 - mrcnn_mask_loss: 0.1879 
 98/100 [============================>.] - ETA: 32s - loss: 0.5641 - rpn_class_loss: 0.0206 - rpn_bbox_loss: 0.2273 - mrcnn_class_loss: 0.0200 - mrcnn_bbox_loss: 0.1093 - mrcnn_mask_loss: 0.1869
 99/100 [============================>.] - ETA: 16s - loss: 0.5597 - rpn_class_loss: 0.0204 - rpn_bbox_loss: 0.2254 - mrcnn_class_loss: 0.0198 - mrcnn_bbox_loss: 0.1085 - mrcnn_mask_loss: 0.1856
100/100 [==============================] - 1709s 17s/step - loss: 0.5557 - rpn_class_loss: 0.0202 - rpn_bbox_loss: 0.2235 - mrcnn_class_loss: 0.0197 - mrcnn_bbox_loss: 0.1078 - mrcnn_mask_loss: 0.1845 - val_loss: 0.3947 - val_rpn_class_loss: 0.0094 - val_rpn_bbox_loss: 0.1267 - val_mrcnn_class_loss: 0.0123 - val_mrcnn_bbox_loss: 0.1065 - val_mrcnn_mask_loss: 0.1396
```

#### 3. Exemples de Détection

On a utilisé le modèle entrainé sur de nouveaux images de test voici le résultat:
![Exemple de détection](sample_detection.png)
*Figure 2: Exemples de détections sur des images de test*
### Hyperparamètres d'Entraînement
- **Learning Rate** : 0.01
- **Optimiseur** : Adam
- **Epochs** : 5
- **Layers** : "heads" (uniquement les couches de tête)






- Images de test avec prédictions







