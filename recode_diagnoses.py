from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json
import re
from pydantic import BaseModel,Field
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
import os


# load environmental variables
load_dotenv(".env")
data_folder=Path(os.getenv("DATA_FOLDER"))
result_folder=Path(os.getenv("RESULTS_FOLDER"))


data=pd.read_parquet(data_folder / 'clinical_doc_full.parquet')

model_name="alibayram/medgemma:27b"
# define output schema which should be boolean values for the following body systems
#"Cardiovascular","Dermatologic","ENT, Dental, Ophthalmologic" , "Febrile / Infectious","Gastrointestinal","Genitourinary & Reproductive", "Musculoskeletal" ,
# "Neurologic & Psychiatric","Respiratory","Unspecified / Other"

class Schema(BaseModel):
    """Schema for body systems affected in diagnoses"""
    cardiovascular: bool = Field(False, description="Cardiovascular")
    dermatologic: bool = Field(False, description="Dermatologic")
    ent_dental_ophthalmologic: bool = Field(False, description="ENT, Dental, Ophthalmologic")
    febrile_infectious: bool = Field(False, description="Febrile / Infectious")
    gastrointestinal: bool = Field(False, description="Gastrointestinal")
    genitourinary_reproductive: bool = Field(False, description="Genitourinary & Reproductive")
    musculoskeletal: bool = Field(False, description="Musculoskeletal")
    neurologic_psychiatric: bool = Field(False, description="Neurologic & Psychiatric")
    respiratory: bool = Field(False, description="Respiratory")
    unspecified_other: bool = Field(False, description="Unspecified / Other")


model=ChatOllama(model=model_name,temperature=0.)
# model with structured output
structured_model=model.with_structured_output(Schema)

sys_prompt="""
You are a professor of medicine working in a primary care setting in Kenya.
Your task is to extract the affected body systems from free-text clinical notes.

For each note:
- Identify which of the following body systems are affected. Output MUST be a single JSON object with ONLY these keys and boolean values (true or false) exactly as shown:
{{
  "cardiovascular": true/false,
  "dermatologic": true/false,
  "ent_dental_ophthalmologic": true/false,
  "febrile_infectious": true/false,
  "gastrointestinal": true/false,
  "genitourinary_reproductive": true/false,
  "musculoskeletal": true/false,
  "neurologic_psychiatric": true/false,
  "respiratory": true/false,
  "unspecified_other": true/false
}}

- Give more weight to the "Diagnosis:" section. If a diagnosis or symptom clearly maps to a system, set that key to true. If unsure or not mentioned, set false.
- Do NOT emit any explanatory text, only the JSON object above, and no additional keys.
Examples:
Input: "Diagnosis: Acute bronchitis. Patient with cough and sputum." 
Output: {{"cardiovascular": false, "dermatologic": false, "ent_dental_ophthalmologic": false, "febrile_infectious": false, "gastrointestinal": false, "genitourinary_reproductive": false, "musculoskeletal": false, "neurologic_psychiatric": false, "respiratory": true, "unspecified_other": false}}

Input: "Assessment: urinary tract infection. Dysuria and frequency." 
Output: {{"cardiovascular": false, "dermatologic": false, "ent_dental_ophthalmologic": false, "febrile_infectious": true, "gastrointestinal": false, "genitourinary_reproductive": true, "musculoskeletal": false, "neurologic_psychiatric": false, "respiratory": false, "unspecified_other": false}}
"""

# use prompt template as before
prompt_template = ChatPromptTemplate.from_messages(
        [("system", sys_prompt), ("user", "{text}")]
    )


extracted_diagnoses={}
error_visits=[]
for _, row in tqdm(data.iterrows(),total=data.shape[0]):
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

system2=pd.merge(system, data[['record_id',"clinical_documentation"]], on='record_id')
system2.to_excel(result_folder / f'Body system - {model_name.replace("/","__")}.xlsx')
# with open('/home/pmwaniki/Dropbox/others/Ambrose/Sync study/Retrospective/Data/recoded_diagnoses.json','w') as f:
#     json.dump(extracted_diagnoses,f,indent=4)