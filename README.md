# Memory OS AI

Memory OS AI est un système d'IA avancé pour la gestion et l'analyse de documents localement. Il prend en charge divers formats (PDF, TXT, DOCX, images, audio, etc.) et utilise des technologies comme FAISS, SentenceTransformers, et Mistral-7B pour la recherche sémantique et la génération de rapports. Ce logiciel est conçu pour fonctionner entièrement en local, garantissant la sécurité des données.

## Fonctionnalités
- Traitement de documents multi-formats (PDF, TXT, DOCX, images, audio, etc.).
- Recherche sémantique avec FAISS.
- Génération de résumés et de rapports avec Mistral-7B.
- Fonctionnement local pour garantir la sécurité des données.
- Adaptation automatique CPU/GPU selon la machine de l'utilisateur.

## Structure du projet
```
memory-os-ai/
│
├── emb/                          # Dossier pour les modèles d'embeddings
│   ├── all-MiniLM-L6-v2/         # Modèle SentenceTransformers (téléchargé automatiquement)
│   └── whisper_base.pt           # Modèle Whisper (téléchargé automatiquement)
│
├── mistral-7b/                   # Dossier pour le modèle Mistral-7B
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf  # Modèle Mistral-7B (téléchargé automatiquement)
│
├── pdfs/                         # Dossier pour les fichiers à traiter (PDFs, images, audio, etc.)
│
├── embeddings_cache.pkl          # Cache des embeddings (généré automatiquement)
│
├── memory_os_ai.py               # Script principal
├── memory_os_ai_gui.py           # Script de l'interface GUI
├── requirements_gpu.txt          # Dépendances pour GPU
├── requirements_cpu.txt          # Dépendances pour CPU
├── README.md                     # Instructions
└── LICENSE                       # Licence LGPL v3
```

## Installation

### Prérequis
- Python 3.9 ou supérieur.

- Outils supplémentaires :

  - Tesseract OCR (pour `pytesseract`) : 
    ```
    brew install tesseract  # Sur macOS
    sudo apt-get install tesseract-ocr  # Sur Ubuntu
FFmpeg (pour whisper) :

```
brew install ffmpeg  # Sur macOS
sudo apt-get install ffmpeg  # Sur Ubuntu
```

Antiword (pour textract, si nécessaire) :
```
brew install antiword  # Sur macOS
sudo apt-get install antiword  # Sur Ubuntu
```

Étapes

Clonez le dépôt :
```
git clone https://github.com/[ton-utilisateur]/memory-os-ai.git
```

cd memory-os-ai
Installez les dépendances selon votre machine :
Sur une machine avec GPU (ex. sur serveur interne avec RTX3090) :
```
pip install -r requirements_gpu.txt
```
Assurez-vous que CUDA est installé (version 12.1 recommandée). Ajustez les URLs dans requirements_gpu.txt si vous utilisez une version différente de CUDA.

Sur une machine sans GPU (ex. MacBook Pro) :
```
pip install -r requirements_cpu.txt
```

Lancez le script principal :
```
python memory_os_ai.py
```

Note sur les modèles : Lors de la première exécution, le script téléchargera automatiquement les modèles nécessaires :

SentenceTransformers (all-MiniLM-L6-v2) : 
-Téléchargé dans emb/all-MiniLM-L6-v2/.
Si le modèle ne fonctionne pas le convertir en sentence avec le script:  
```
convert_to_sentence_models.py
```

Mistral-7B (mistral-7b-instruct-v0.2.Q4_K_M.gguf) : 
-Téléchargé dans mistral-7b/.

Whisper (base) : Téléchargé dans emb/whisper_base.pt. (Choisissez le modèle 2 meilleurs compris, moins volumineux)

Ces modèles seront stockés localement et utilisés pour toutes les exécutions suivantes.

VERIFIER BIEN LES EMPLACEMENTS DES MODELES POUR LE BON FONCTIONNEMENT.
ILS DOIVENT ETRE COHERENTS AVEC LE SCRIPT :)


(Optionnel) Lancez l'interface GUI :
```
python memory_os_ai_gui.py
```
Utilisation:

Placez vos fichiers (PDFs, images, audio, etc.) dans le dossier pdfs/.

Avec le script principal (memory_os_ai.py), suivez les instructions en ligne de commande :
recherche <mot-clé> : Recherche un mot-clé dans les documents.
rapport <sujet> : Génère un rapport structuré sur un sujet.
exit : Quitte le programme.

Avec l'interface GUI (memory_os_ai_gui.py), utilisez les boutons pour charger les fichiers, rechercher, ou générer des rapports.
Licence

Memory OS AI est sous licence GNU Lesser General Public License v3.0 (LGPL v3) pour une utilisation non commerciale. Cela signifie que vous pouvez utiliser, modifier, et redistribuer le logiciel gratuitement pour des usages non commerciaux, à condition de respecter les termes de la LGPL v3 (voir le fichier LICENSE pour plus de détails).

Pour une utilisation commerciale (ex. intégration dans un produit commercial, utilisation en entreprise), vous devez acheter une licence commerciale. Les revenus générés par les licences commerciales contribuent directement au développement et à la maintenance du projet. Contactez-moi à romainsantoli@gmail.com pour plus d'informations sur les tarifs et les conditions.

Support
Besoin d'aide pour installer ou configurer Memory OS AI ? Je propose des services de support et de personnalisation payants. Contactez-moi à romainsantoli@gmail.com pour plus d'informations.

Sponsor
Soutenez le développement de Memory OS AI en faisant un don ! [Lien vers Patreon/Open Collective]

Contribution
Les contributions sont les bienvenues ! Veuillez consulter le fichier CONTRIBUTING.md pour plus d'informations sur la manière de contribuer.
