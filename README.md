# Projet-Dockerfile-Prediction-Immobilier
Prévision du prix de l'immobilier avec Machine Learning en Python
Utilisation du modèle de régression Ridge, Bayesian, Lasso, Elastic Net et OLS pour la prédiction
Pour prédire les prix de vente, nous allons utiliser les algorithmes de régression linéaire suivants : algorithme Ordinal Least Square (OLS), algorithme de régression Ridge, algorithme de régression Lasso, algorithme de régression bayésienne et enfin algorithme de régression Elastic Net. Ces algorithmes peuvent être facilement implémentés en python avec l'utilisation du package scikit-learn.
Enfin, nous concluons quel modèle est le mieux adapté au cas donné en évaluant chacun d'eux à l'aide des métriques d'évaluation fournies par le package scikit-learn.
Étapes impliquées
1.	Importation des packages requis dans notre environnement python
2.	Importer les données sur le prix de l'immobilier et faire de l'EDA dessus
3.	Visualisation des données sur les données du prix de l'immobilier
4.	Sélection des fonctionnalités et fractionnement des données
5.	Modélisation des données à l'aide des algorithmes
6.	Évaluation du modèle construit à l'aide des métriques d'évaluation
7.	Création d'un fichier Dockerfile:
FROM jupyter/scipy-notebook

RUN pip install joblib

COPY House_Data.csv ./House_Data.csv


COPY Lab_2_house_prices.py ./Lab_2_house_prices.py

RUN python3 Lab_2_house_prices.py

9.	Passons maintenant à notre partie codage !
Importation des packages requis
