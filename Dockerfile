# Partir de la MÊME image que votre master/worker
# Cela inclut déjà pyspark et java, ce qui résout votre erreur de build.
FROM docker.io/apache/spark:3.5.0

WORKDIR /app

# Passer en root temporairement pour installer les paquets
USER root

# Copier les requirements (Assurez-vous d'enlever 'pyspark' de requirements.txt)
COPY requirements.txt .

# Installer vos bibliothèques (streamlit, pandas, etc.)
# C'est maintenant très rapide car pyspark n'est pas téléchargé
RUN pip install -r requirements.txt

# Copier le script d'attente et le rendre exécutable
COPY wait-for-spark.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/wait-for-spark.sh

# Copier le reste de votre application
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Revenir à l'utilisateur par défaut (bonne pratique)
USER spark

# La commande CMD est redondante car compose.yaml la fournit,
# mais en voici une correcte pour référence :
CMD ["/usr/local/bin/wait-for-spark.sh", "streamlit", "run", "main.py"]