# Memory OS AI

Memory OS AI est un système d'IA avancé pour la gestion et l'analyse de documents localement. Il prend en charge divers formats (PDF, TXT, DOCX, images, audio, etc.) et utilise des technologies comme FAISS, SentenceTransformers, et Mistral-7B pour la recherche sémantique et la génération de rapports.

## Fonctionnalités
- Traitement de documents multi-formats (PDF, TXT, DOCX, images, audio, etc.).
- Recherche sémantique avec FAISS.
- Génération de résumés et de rapports avec Mistral-7B.
- Fonctionnement local pour garantir la sécurité des données.
- Adaptation automatique CPU/GPU selon la machine de l'utilisateur.

## Prérequis
- Python 3.9 ou supérieur.
- Dépendances : `numpy`, `faiss-cpu` (ou `faiss-gpu` si GPU disponible), `sentence-transformers`, `llama-cpp-python`, `pymupdf`, `pytesseract`, `python-pptx`, `docx`, `textract`, `pillow`, `whisper`.
- Pour l'interface GUI : `PySide2`.
- Modèles : Téléchargez les modèles nécessaires (SentenceTransformers, Mistral-7B, Whisper) et placez-les dans les chemins indiqués dans le script.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/[ton-utilisateur]/memory-os-ai.git
   cd memory-os-ai
