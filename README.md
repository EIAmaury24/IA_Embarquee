# Rapport - IA Embarquée

## Martin HAMEL  
## Amaury MIQUEL  

### Développement d’un réseau de neurones pour la maintenance prédictive

### Organisation du projet
Parler de la construction du projet et des différents dossiers


## 📌 Introduction

Ce projet s'inscrit dans le cadre du déploiement d'un réseau de neurones profonds (DNN) pour la **maintenance prédictive**. Il se divise en **trois grandes étapes** :

1. **Conception et entraînement du modèle** sur un fichier Google Colab.
2. **Exportation et intégration du modèle** dans **STM32CubeIDE**.
3. **Communication avec la carte STM32L4R9**, à l’aide d’un script Python.

---

## 🧠 Partie 1 - Développement sur Google Colab

L’objectif de cette première étape est de **créer un modèle capable de prédire les défaillances de machines industrielles à partir de leurs conditions de fonctionnement**.

### 🔧 Données d'entrée
Les conditions de fonctionnement sont définies par plusieurs paramètres :
- Température de l'air
- Température du processus
- Couple
- Taux de rotation
- Usure des outils

À partir de ces données, le but est de prédire **si la machine est en défaillance ou non**, et **d’identifier le type d’erreur** en cas de dysfonctionnement.

### 📊 Modèle MLP (Multi-Layer Perceptron)

Nous avons construit un **MLP simple** avec la bibliothèque Keras :

```python
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.05),
    Dense(64, activation='relu'),
    Dropout(0.05),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid')
])
