# Partir de la MÊME image que votre master/worker
FROM docker.io/apache/spark:3.5.0

WORKDIR /app

# Passer en root temporairement pour installer les paquets
USER root

# --- CORRECTION ---
# Mettre à jour apt et installer les outils de build (pour numpy/pandas)
# et les en-têtes de développement Python
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
# --- FIN CORRECTION ---

# Copier les requirements
COPY requirements.txt .

# Mettre à jour pip et installer les bibliothèques
# Utiliser --no-cache-dir est une bonne pratique dans Docker
RUN pip install --upgrade pip --timeout=1000
RUN pip install --no-cache-dir -r requirements.txt --timeout=1000 # <-- Ligne 15, devrait maintenant fonctionner

# Copier le script d'attente et le rendre exécutable
COPY wait-for-spark.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/wait-for-spark.sh

# Copier le reste de votre application
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Revenir à l'utilisateur par défaut (bonne pratique)
USER spark


CMD ["/usr/local/bin/wait-for-spark.sh", "python", "run", "app.py"]