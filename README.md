# multiclass_IDS_smartgrid

This repository contains the Ensemble Representation Learning Classifier (ERLC) developed for multi-label attack detection and localization in smart cyber-physical grids.

## Installation and Use
To run this model:

1. Clone the repository

   ``` 
   git clone https://github.com/jsakhnin/multiclass_IDS_smartgrid.git 
   ```
   
2. Navigate to the directory `cd multiclass_IDS_smartgrid` and then create an environment and install the requirements:

    * Using Conda:
    
       ```
       conda create --name myenv
       conda activate myenv
       ```
    
    * Using Pip:
    
       ```
       python3 -m venv myenv
       ```

       on Windows:

       ```
       myenv\Scripts\activate.bat
       ```

       on Linux/MacOS:

       ```
       source myenv/bin/activate
       ```
    
    
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
   
4. Create jupyter kernel from envrionment
   ```
   python -m ipykernel install --user --name=myEnv
   ```
   
5. For basic implementation of the classifier, go through implementation.ipynb notebook. For the attack localization component, go through attack_localization.ipynb notebook.


## Repository Architecture:

* input folder: contains the power system data acquired from [Oak Ridge National Laboratories](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)
* output folder: contains the output and results produced by the scripts in the repository, including figures and latex tables.
* `erlc.py` : The ERLC model class
* `dataPipline.py` : The data processing class
* `implementation.ipynb` : Basic implementation example of the model
* `cv_test.ipynb` : cross validation test used to evaluate the model
* `attack_localization.ipynb` : notebook that tests the efficacy of the attack localization component of the model
* `tuning.ipynb` : A notebook for tuning and logging neural network training
* `result_processing.ipyng` : Processing results and creating figures and tables.
