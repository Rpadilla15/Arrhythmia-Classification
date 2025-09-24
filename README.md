############ DEPENDENCIES ############

There are 2 dependency related files:
    - requirements.txt       # Pip dependency list file
    - environment.yml        # Conda environment file

Unfortunately the requirements.txt contains few of the modules we actually use,
you can try it but I'd recomend using conda or otherwise downloading the dependencies 
as you encounter the trouble. You can use environment.yml for guidance with the version.

If using conda, it is easy to create the environment using
    conda env create -f environment.yml
and all dependencies should become installed. 

To check one can list the available environments:
    conda env list
The environment is called 'StatLearn', it should become available
to choose as the python interpreter in VS.



############ FOLDER SCHEMA ############

└── MyProject/  # Root: Project Folder (e.g., "ECG_Analysis")
    ├── Arrhythmia-Classification/ # Code repository folder
    │   ├── Experiments/           # Folder with Jupyter notebooks
    │   │   ├── data-vis.ipynb
    │   │   └── ...
    │   ├── src/                   # Main code corpus
    │   │   ├── data_handling/     # Data related code
    │   │   │   ├── dataset.py
    │   │   │   └── ...
    │   │   └── ...
    │   ├── README.md              # Project README file
    │   ├── requirements.txt       # Pip dependency list file
    │   └── environment.yml        # Conda environment file
    │
    └── Data/                      # Data folder
        ├── input/                 # Input data and metadata for experiments
        │   ├── experiment/        # Will contain json files with metadata for each experiment
        │   │   ├── leads_MLII_V1_tst0.25_rd42.json/    (e.g. patients with leads MLII, V1; test_split 25%; random_seed 42)
        │   │   └── ...
        │   └── preprocessed_data/ # Will contain folders with data preprocessed in different ways
        │       ├── fs250_bp0.5-45Hz/                   (e.g resampling at 250Hz bandpass filter (0.5,45))
        │       └── ...
        ├── mitbih_database/       # Original MIT-BIH Arrhythmia Database
        ├── AAMI_MAPPING.yaml      # Arrhythmia type mapping file
        └── patient_sensors.json   # Leads by patient file
