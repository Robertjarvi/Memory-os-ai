# Contributing to Memory OS AI

Merci de votre intérêt pour contribuer à **Memory OS AI** ! Ce projet est open-source et nous accueillons avec plaisir les contributions de la communauté pour améliorer ses fonctionnalités, corriger des bugs, ou ajouter de nouvelles idées. Voici un guide pour vous aider à contribuer de manière efficace et organisée.

## Comment contribuer ?

Il existe plusieurs façons de contribuer à Memory OS AI :

- **Signaler des bugs** : Si vous trouvez un bug, ouvrez une issue pour le signaler.
- **Proposer des améliorations** : Si vous avez une idée pour une nouvelle fonctionnalité ou une amélioration, ouvrez une issue pour en discuter.
- **Corriger des bugs ou ajouter des fonctionnalités** : Si vous souhaitez contribuer du code, suivez les étapes ci-dessous pour soumettre une pull request.
- **Améliorer la documentation** : Si vous trouvez des erreurs ou des lacunes dans la documentation (README, instructions, etc.), proposez des améliorations.
- **Fournir des retours** : Vos retours sur l’utilisation du logiciel sont précieux. Partagez vos expériences ou suggestions via une issue.

## Prérequis pour contribuer

Avant de commencer, assurez-vous d’avoir les éléments suivants :

- **Python 3.9 ou supérieur** installé.
- Les dépendances du projet installées (voir `requirements_gpu.txt` ou `requirements_cpu.txt` dans le `README.md`).
- Un environnement de développement configuré (ex. un éditeur comme VSCode, PyCharm, etc.).
- Git installé pour cloner le dépôt et soumettre des pull requests.

## Étapes pour contribuer

### 1. Forker le dépôt
- Forkez le dépôt en cliquant sur le bouton "Fork" sur la page GitHub de [Memory OS AI](https://github.com/[ton-utilisateur]/memory-os-ai).
- Clonez votre fork localement :
  ```
  git clone https://github.com/[votre-utilisateur]/memory-os-ai.git
  cd memory-os-ai


2. Créer une branche pour vos modifications
Créez une nouvelle branche pour vos modifications :
```
git checkout -b [nom-de-votre-branche]
```
Par exemple :
fix-bug-ocr pour corriger un bug dans l’OCR.
feature-add-new-format pour ajouter un nouveau format de fichier.


3. Installer les dépendances
Selon votre machine, installez les dépendances :

Sur une machine avec GPU :
```
pip install -r requirements_gpu.txt
```

Sur une machine sans GPU :
```
pip install -r requirements_cpu.txt
```
Assurez-vous que les prérequis système (Tesseract, FFmpeg, Antiword) sont installés, comme indiqué dans le README.md.


4. Faire vos modifications
Apportez vos modifications au code ou à la documentation.

Si vous ajoutez une nouvelle fonctionnalité, assurez-vous qu’elle est compatible avec les deux modes (CPU et GPU).

Si vous modifiez des fichiers existants, respectez le style de code actuel (ex. indentation, conventions de nommage).


5. Tester vos modifications
Testez vos modifications localement pour vous assurer qu’elles fonctionnent correctement :
Lancez le script principal :
```
python memory_os_ai.py
```

Si votre modification concerne l’interface GUI, testez également avec :
```
python memory_os_ai_gui.py
```
Vérifiez que vos modifications ne cassent pas les fonctionnalités existantes (ex. recherche, génération de rapports, traitement des fichiers).


6. Soumettre une pull request
Ajoutez et commitez vos modifications :
```
git add .
git commit -m "Description claire de vos modifications (ex. 'Fix bug in OCR for PNG files')"
```
Poussez votre branche sur votre fork :
```
git push origin [nom-de-votre-branche]
```

Créez une pull request (PR) :
Allez sur la page GitHub de ton fork.
Cliquez sur "Compare & pull request".

Remplissez le formulaire de PR avec :
Un titre descriptif (ex. "Fix bug in OCR for PNG files").

Une description détaillée de vos modifications, incluant :
Le problème que vous avez résolu ou la fonctionnalité que vous avez ajoutée.
Les tests que vous avez effectués.

Toute information pertinente pour le relecteur (ex. "J’ai testé sur CPU et GPU, aucun impact sur les performances").
Soumettez la PR.

7. Révision et intégration
Je (ou un autre mainteneur) examinerai votre pull request.
Si des modifications sont nécessaires, je vous ferai des commentaires. Apportez les changements demandés et mettez à jour votre PR.

Une fois approuvée, votre contribution sera intégrée dans la branche principale.


8.Règles et bonnes pratiques
Respectez la licence : Memory OS AI est sous licence LGPL v3 pour une utilisation non commerciale. Si vous contribuez, votre code sera également sous cette licence. Pour une utilisation commerciale, une licence payante est requise (voir README.md).

Style de code :
Utilisez une indentation de 4 espaces (pas de tabulations).
Suivez les conventions PEP 8 pour le style de code Python.
Ajoutez des commentaires pour expliquer les sections complexes.

Commits clairs : Rédigez des messages de commit descriptifs (ex. "Add support for new file format: ODT").

Tests : Assurez-vous que vos modifications ne cassent pas les fonctionnalités existantes. Testez sur CPU et GPU si possible.
Respect de la communauté : Soyez respectueux dans vos interactions (issues, PR, commentaires).
Types de contributions bienvenues

Corrections de bugs : Si vous trouvez un bug dans le traitement des fichiers, la recherche sémantique, ou la génération de rapports, proposez une correction.

Nouvelles fonctionnalités :
Ajout de nouveaux formats de fichiers pris en charge.
Amélioration des performances (ex. optimisation pour CPU/GPU).
Ajout de nouvelles options de recherche ou de génération de rapports.
Améliorations de l’interface GUI : Si vous utilisez memory_os_ai_gui.py, proposez des améliorations (ex. nouveaux boutons, meilleure ergonomie).

Documentation : Améliorez le README.md, ajoutez des exemples d’utilisation, ou corrigez des erreurs.

Tests : Ajoutez des tests unitaires pour valider les fonctionnalités (actuellement, le projet n’a pas de tests automatisés).
Signaler un bug ou proposer une amélioration
Si vous ne souhaitez pas contribuer directement au code, vous pouvez signaler un bug ou proposer une amélioration en ouvrant une issue :

Allez sur la page GitHub du projet et cliquez sur l’onglet "Issues".
Cliquez sur "New issue".
Choisissez le type d’issue ("Bug report" ou "Feature request").

Remplissez le formulaire avec :
Une description claire du bug ou de la fonctionnalité proposée.
Les étapes pour reproduire le bug (si applicable).
Votre environnement (ex. "MacBook Pro sans GPU, Python 3.9").
Toute information supplémentaire (ex. captures d’écran, logs d’erreurs).

Contact
Pour toute question sur la contribution, contactez-moi à romainsantoli@gmail.com.

Merci de contribuer à Memory OS AI ! Ensemble, nous pouvons faire de ce projet un outil encore plus puissant pour la gestion et l’analyse de documents.
