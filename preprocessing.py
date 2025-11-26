from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import dotenv_values
import re
config = dotenv_values(".env")

# dsmb_folder=Path("/home/pmwaniki/Dropbox/others/Ambrose/Sync study/DSMB/")

result_folder=Path(config["RESULTS_FOLDER"])
data_folder=Path(config["DATA_FOLDER"])
data=pd.read_parquet(data_folder/"outcome_data.parquet")

data=data.loc[data['n_clinicians']==1].copy()
data['arm_num']=data['arm'].map({'Intervention':1.0,'Control':.0})
data['failure_num']=data['failure_status_final'].map({'No treatment failure':0.,
                                                  'Treament failure':1.,
                                                  'Not assessed':pd.NA})
unique_clinicians=data['clinician_id'].unique().astype('int')
unique_clinicians=np.sort(unique_clinicians)
data['clinician_num']=data['clinician_id'].map({str(v):i for i,v in enumerate(unique_clinicians)})

unique_hosp=data['hosp_id'].unique()
unique_hosp=np.sort(unique_hosp)
data['hospital_num']=data['hosp_id'].map({v:i for i,v in enumerate(unique_hosp)})
data['visit_number2']=data['visit_number'].map(lambda x: re.sub(r'[^0-9]','',str(x)))


data.to_parquet(result_folder/"outcome_data_cleaned.parquet")



