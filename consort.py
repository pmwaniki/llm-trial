from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")

# dsmb_folder=Path("/home/pmwaniki/Dropbox/others/Ambrose/Sync study/DSMB/")

result_folder=Path(config["RESULTS_FOLDER"])
data_folder=Path(config["DATA_FOLDER"])

main=pd.read_parquet(data_folder/"main.parquet")
screening=pd.read_parquet(data_folder/"screening_cleaning.parquet")
screening['study_id']=screening['study_id'].map(lambda x:f"{x:.0f}")


consort_dict={}
consort_dict['screened']=screening.shape[0] #screened
consort_dict['not_eligible']=screening.loc[screening['eligible']==False].shape[0] # not eligible
consort_dict['not_consented']=screening.loc[(screening['consented2']=="No") & (screening['eligible'])].shape[0]

#unlinked to emr
consort_dict['arm_unlinked']=screening.loc[(screening['consented2']=="Yes") & (screening['eligible']) &
              (screening['study_id'].isin(main.loc[main['arm']=="Unlinked",'study_id']))].shape[0]

# consort_dict['arm_unlinked']=main.loc[(main['arm']=="Unlinked")& (main['withdrawn']=="No")].shape[0]

consort_dict['mixed']=screening.loc[(screening['consented2']=="Yes") & (screening['eligible']) &
              (screening['study_id'].isin(main.loc[main['arm']=="Mixed",'study_id']))].shape[0]


consort_dict['unconsented']=screening.loc[(screening['consented2']=="Yes") & (screening['eligible']) &
              (screening['study_id'].isin(main.loc[main['arm']=="Unknown",'study_id']))].shape[0]

randomized=main.loc[main['arm'].isin(["Intervention","Control"]) & main['study_id'].isin(screening['study_id'])]


consort_dict['randomized']=randomized.shape[0]


# consort_dict['consented']=main.shape[0]
consort_dict['withdrawn_control']=randomized.loc[(randomized['arm']=="Control") &
                                            # (randomized['complete']==True) &
               (randomized['withdrawn']=="Yes")].shape[0]

consort_dict['withdrawn_intervention']=randomized.loc[(randomized['arm']=="Intervention") &
                                            # (randomized['complete']==True) &
               (randomized['withdrawn']=="Yes")].shape[0]

#loss to follow-up
consort_dict['loss_control']=randomized.loc[(randomized['arm']=="Control") &
                                            (randomized['complete']==False) &
                                            (randomized['withdrawn']=="No")].shape[0]
consort_dict['loss_intervention']=randomized.loc[(randomized['arm']=="Intervention") &
                                            (randomized['complete']==False)&
                                            (randomized['withdrawn']=="No")].shape[0]


# exclude multiple clinicians
consort_dict['multiple_clin_control']=randomized.loc[(randomized['arm']=="Control") &
                                            (randomized['complete']==True) &
                                            (randomized['withdrawn']=="No") &
                                                     (randomized['n_clinicians'] != 1)].shape[0]
consort_dict['multiple_clin_intervention']=randomized.loc[(randomized['arm']=="Intervention") &
                                            (randomized['complete']==True)&
                                            (randomized['withdrawn']=="No") &
                                                          (randomized['n_clinicians'] != 1)].shape[0]

consort_dict['final_control']=randomized.loc[(randomized['arm']=="Control") &
                                            (randomized['complete']==True) &
                                            (randomized['withdrawn']=="No") &
                                                     (randomized['n_clinicians'] == 1)].shape[0]
consort_dict['final_intervention']=randomized.loc[(randomized['arm']=="Intervention") &
                                            (randomized['complete']==True)&
                                            (randomized['withdrawn']=="No") &
                                                          (randomized['n_clinicians'] == 1)].shape[0]
# consort_dict['arm_mixed']=main.loc[(main['arm']=="Mixed") & (main['withdrawn']=="No")].shape[0]+main.loc[(main['arm']=="Unknown")& (main['withdrawn']=="No")].shape[0]
# consort_dict['arm_control']=main.loc[(main['arm']=="Control")& (main['withdrawn']=="No")].shape[0]
# consort_dict['arm_intervention']=main.loc[(main['arm']=="Intervention")& (main['withdrawn']=="No")].shape[0]

# consort_dict['control_followup']=None
# consort_dict['intervention_followup']=None
# consort_dict['f2']=main.loc[(main['arm']=="Intervention")& (main['withdrawn']=="No") & main['status'].isin(['Completed on day 14', 'Completed on day 3'])].shape[0]
# consort_dict['f1']=main.loc[(main['arm']=="Control")& (main['withdrawn']=="No") & main['status'].isin(['Completed on day 14', 'Completed on day 3'])].shape[0]
# consort_dict['s2']=main.loc[(main['arm']=="Intervention")& (main['withdrawn']=="No") & main['status'].isin(['Completed on day 14', 'Completed on day 3']) & (main['n_clinicians']==1)].shape[0]
# consort_dict['s1']=main.loc[(main['arm']=="Control")& (main['withdrawn']=="No") & main['status'].isin(['Completed on day 14', 'Completed on day 3']) & (main['n_clinicians']==1)].shape[0]
