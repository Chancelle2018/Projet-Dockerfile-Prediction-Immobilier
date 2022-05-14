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

Nous allons maintenant essayer de créer de créer docker compose afin de lancer notre application via n’importe quel système d’exploitation compatible ou non.
Une fois que notre code python est a été réalisé, nous allons nous créer le dockerfile. Pour cela, créez un fichier sans extension nommé dockerfile dans notre dossier puis ajouter les instructions suivantes

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

Le syntaxe pour spécifier la version de docker à utiliser et les mises à jour autorisées ;
FROM : pour indiquer l’image de base que nous souhaitons utiliser à savoir Python 3.9 ;
WORKDIR : pour communiquer l’emplacement dans laquelle souhaitons-nous que les commandes qui suivent se basent ;
COPY pour copier les fichiers nécessaires à l’exécution de l’image à savoir requirements.txt ;
Nous allons ensuite installer les packages nécessaires à l’image à l’aide de la commande RUN. Ici, nous avons demandé à docker daemon d’aller puiser dans le fichier texte copié ci-dessus le nom des packages à installer à savoir Flask ;
Après cela, nous copions tout vers le système de fichier de l’image à créer à l’aide de la commande COPY . . ;
Nous allons indiquer le port 8000 que nous allons utiliser avec la commande PORT ;
La dernière ligne spécifie l’action à exécuter en premier lors du lancement du conteneur. Il s’agit de la même commande que nous avons utilisée pour tester notre application dans la section précédente.
Nous avons le dockerfile qui va nous servir à créer notre propre image.
docker build --tag python-docker .
docker images
Comme nous pouvons le constater, notre image a bien été créée.

Si nous supprimons cette image en utilisant la commande docker rmi <ID_de_l_image> et qu’on le reconstruit, nous allons voir que certaines commandes ont été mises en cache et ne sont plus exécutées.

Pour vérifier que l’on peut accéder à notre application, nous allons lancer un conteneur de l’image avec la commande suivante :

docker run -p 8000:5000 python-docker
Ici, nous avons indiqué que l’application exposée au port 5000, c’est-à-dire notre application, sera accessible via le port 8000. Vérifions cela en tapant http://127.0.0.1:8000/ sur un navigateur.
Malheureusement, nous n’avons pas réussir à effectuer cette dernière partie du projet.
