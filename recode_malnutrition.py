import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pygrowup
from tqdm import tqdm

load_dotenv(".env")

data_folder=Path(os.getenv("DATA_FOLDER"))
results_folder=Path(os.getenv("RESULTS_FOLDER"))


outcome_data=pd.read_parquet(data_folder / "outcome_data.parquet")
outcome_data=outcome_data.loc[outcome_data['age']<=60].copy()
clinical_data=pd.read_excel(data_folder / "clinical_data-PATH.xlsx")
clinical_data['visit_number']=clinical_data['visit_number'].map(lambda x: re.sub(r'[^0-9]','',str(x)))
outcome_data['visit_number']=outcome_data['visit_number'].map(lambda x: re.sub(r'[^0-9]','',str(x)))

outcome_data=pd.merge(outcome_data, clinical_data,how='left', on='visit_number')
outcome_data['Referrals2']=outcome_data['Referrals'].str.replace("Referrals:\n","")
# weight for height z-score
calc = pygrowup.Calculator(
    include_cdc=False,  # Use WHO standards
    # data_dir='path/to/who_data_files' # Specify if needed
)
outcome_data['whz']=pd.NA

for i, row in outcome_data.iterrows():
    if pd.isna(row['doc_wgt']) | pd.isna(row['age']) | pd.isna(row['gender']) | pd.isna(row['doc_hgt']):
        continue
    try:
        val=calc.wfh(measurement=row['doc_wgt'],age_in_months=row['age'],sex=row['gender'][0],height=row['doc_hgt'])
        outcome_data.loc[outcome_data['record_id']==row['record_id'],'whz']=float(val)
    except pygrowup.exceptions.InvalidMeasurement:
        continue
    except Exception as e:
        raise e




model_name="alibayram/medgemma:27b"

class Schema(BaseModel):
    """Schema for malnutrtion risk and treatment"""
    malnutrition_diagnosis: bool = Field(False, title="Malnutrition diagnosis")
    malnutrition_risk: bool = Field(False, description="Clinician identified a child as at risk of malnutrition")
    nutritionist_referred: bool = Field(False, description="Child referred to a nutritionist")



model=ChatOllama(model=model_name,temperature=0.)
# model with structured output
structured_model=model.with_structured_output(Schema)

sys_prompt="""
You are a professor of medicine working in a primary care setting in Kenya.
Your task is to review clinical notes of children under five for diagnoses and treatment of malnutrition.

From the clinical notes:
- Identify if a child was diagnosed as malnutrition. ie set malnutrition_diagnosis as true
- Identify whether the child was considered at risk of malnutrition. ie set malnutrition_risk as true 
- Identify if the child was refered to a nutritionist. ie set nutritionist_referred as true 
- Do not used anthropometric measurements MUAC, weight, height, etc. to make this determination unless the clinician
noted that their values were of concern.

- Output MUST be a single JSON object with ONLY these keys and boolean values (true or false) exactly as shown:
{{
  "malnutrition_diagnosis": true/false,
  "malnutrition_risk": true/false,
  "nutritionist_referred": true/false,
  
}}

"""


# use prompt template as before
prompt_template = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{text}")]
    )


extracted_diagnoses={}
error_visits=[]
for _, row in tqdm(outcome_data.iterrows(),total=outcome_data.shape[0]):
    text=row.get('clinical_documentation')
    # ensure consistent key for missing text
    key = row.get('record_id')
    if not text or pd.isna(text):
        extracted_diagnoses[key]=None
        continue


    prompt_text=text
    prompt = prompt_template.invoke({'text': prompt_text})
    result = structured_model.invoke(prompt)

    # robust parsing + logging
    try:
        # structured_model may return a pydantic model-like object
        if hasattr(result, "dict"):
            result_dict = result.model_dump()
        elif isinstance(result, dict):
            result_dict = result
        else:
            # fallback: try extracting raw content and parse JSON
            raw = getattr(result, "content", None) or str(result)
            # strip triple backticks if present
            raw = re.sub(r'```(?:json)?', '', raw).strip('` \n')
            result_dict = json.loads(raw)
        # attach metadata
        result_dict['Dx'] = row.get('Dx')
        result_dict['Referrals']=row.get('Referrals2')
        result_dict['clinical_documentation']=row.get('clinical_documentation')
        result_dict['record_id'] = row.get('record_id', key)
        extracted_diagnoses[result_dict['record_id']] = result_dict
    except Exception as e:
        # save raw output for debugging
        raw = getattr(result, "content", None) or str(result)
        print(f"PARSE ERROR record={key} error={e}")
        print("RAW OUTPUT:", raw[:1000])
        error_visits.append(key)
        extracted_diagnoses[key] = {"parse_error": True, "raw": raw, "record_id": key}

system=pd.DataFrame(extracted_diagnoses.values())
system2=pd.merge(system,outcome_data[['record_id', 'doc_muc','whz']],how="left",on='record_id')
system2.to_excel(results_folder / f'Malnutrition recode - {model_name.replace("/","__")}.xlsx',index=False)


