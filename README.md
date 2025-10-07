############ DEPENDENCIES ############

There are 2 dependency related files:
    - requirements.txt       # Pip dependency list file
    - environment.yml        # Conda environment file

If using conda, it is easy to create the environment using
    conda env create -f environment.yml
and all dependencies should become installed. 

To check one can list the available environments:
    conda env list
The environment is called 'StatLearn', it should become available
to choose as the python interpreter in VS.



############ FOLDER SCHEMA ############

└── MyProject/  # Root: Project Folder (e.g., "ECG_Analysis")
├── Arrhythmia-Classification/      # Code repository folder
    │   ├── Experiments/                # Folder with Jupyter notebooks
    │   │   ├── data-vis.ipynb              # Data visualizations (Check if augmentations work, etc)
    │   │   └── transf.ipynb                # Model output visualization (check if model work)
    │   │
    │   ├── src/                        # Main code corpus
    │   │   ├── data_handling/              # Data related code
    │   │   │   ├── dataset.py                  # AUGMENTATION Code. Create dataset and fetch data for training, eval, etc.
    │   │   │   ├── patiet_grouping.py          # Separate patients into train, test, etc.
    │   │   │   ├── preprocess_utils.py
    │   │   │   └── preprocess.py
    │   │   └── model/
    │   │       ├── MLP_head.py                 # CLASSIFICATION model. MLP architecture
    │   │       ├── model_utils.py              # Functionns and classes required for transformer
    │   │       ├── training_utils.py           # Training functions includingg training loop
    │   │       └── transformer.py              # ENCODER - DECODER model. MAE Transformer architecture
    │   ├── README.md                   # Project README file
    │   ├── requirements.txt            # Pip dependency list file
    │   └── environment.yml             # Conda environment file
    │
    └── Data/                       # Data folder
        ├── input/                      # Input data and metadata for experiments
        │   ├── experiment/             # Will contain json files with metadata for each experiment
        │   │   ├── leads_MLII_V1_tst0.25_rd42.json/    (e.g. patients with leads MLII, V1; test_split 25%; random_seed 42)
        │   │   └── ...
        │   └── preprocessed_data/      # Will contain folders with data preprocessed in different ways
        │       ├── fs250_bp0.5-45Hz/                   (e.g resampling at 250Hz bandpass filter (0.5,45))
        │       └── ...
        ├── output/                     # Output from models and results
        │   ├── experiment/  **TODO**
        ├── mitbih_database/            # Original MIT-BIH Arrhythmia Database
        ├── AAMI_MAPPING.yaml           # Arrhythmia type mapping file
        └── patient_sensors.json        # Leads by patient file
