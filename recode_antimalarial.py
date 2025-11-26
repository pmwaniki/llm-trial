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

data_folder = Path(os.getenv("DATA_FOLDER"))
results_folder = Path(os.getenv("RESULTS_FOLDER"))

outcome_data = pd.read_parquet(data_folder / "outcome_data.parquet")

clinical_data = pd.read_excel(data_folder / "clinical_data-PATH.xlsx")
clinical_data['visit_number'] = clinical_data['visit_number'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))
outcome_data['visit_number'] = outcome_data['visit_number'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))

outcome_data = pd.merge(outcome_data, clinical_data, how='left', on='visit_number')




model_name = "alibayram/medgemma:27b"


class Schema(BaseModel):
    """Schema for diagnoses and treatment of malaria"""
    malaria_test:bool=Field(False, description="Malaria test done")
    test_type:str=Field(None, description="Type of test")
    malaria_positive:bool=Field(None, description="Malaria test positive")
    antimalarial_prescribed:bool=Field(False, description="Antimalaria drugs prescribed")
    antimalarial_name:str=Field(None, description="Name of antimalaria")


model = ChatOllama(model=model_name, temperature=0.)
# model with structured output
structured_model = model.with_structured_output(Schema)

sys_prompt = """
You are a professor of medicine working in a primary care setting in Kenya.
Your task is to review clinical notes of patients in an outpatient department for prescription of malaria drugs.


From the clinical notes:
- Whether Antimalarial were prescribed: antimalarial_prescribed
- Specific antimalarial prescribed (in case antimalarial were prescribed): antimalarial_name
- Identify whether malaria test was done : malaria_test
- Type of malaria test done in case one was done (eg RDT, or Microscopy): test_type
- Whether malaria test was positive (for patients with test done): malaria_positive




- Output MUST be a single JSON object with ONLY these keys as shown:
{{
  "malarial_test": true/false,
  "test_type: "Type of test done eg RDT, Microscopy"
  "malaria_positive": true/false,
  "antimalarial_prescribed": true/false,
  "antimalarial_name": "Drug name"
}}
"""

# use prompt template as before
prompt_template = ChatPromptTemplate.from_messages(
    [("system", sys_prompt), ("user", "{text}")]
)

extracted_diagnoses = {}
error_visits = []
for _, row in tqdm(outcome_data.iterrows(), total=outcome_data.shape[0]):
    text = row.get('clinical_documentation')
    # ensure consistent key for missing text
    key = row.get('record_id')
    if not text or pd.isna(text):
        extracted_diagnoses[key] = None
        continue

    prompt_text = text
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

        result_dict['clinical_documentation'] = row.get('clinical_documentation')
        result_dict['record_id'] = row.get('record_id', key)
        extracted_diagnoses[result_dict['record_id']] = result_dict
    except Exception as e:
        # save raw output for debugging
        raw = getattr(result, "content", None) or str(result)
        print(f"PARSE ERROR record={key} error={e}")
        print("RAW OUTPUT:", raw[:1000])
        error_visits.append(key)
        extracted_diagnoses[key] = {"parse_error": True, "raw": raw, "record_id": key}

system = pd.DataFrame(extracted_diagnoses.values())
system2 = pd.merge(system, outcome_data[['record_id', 'LabTest', 'Rx',"Dx"]], how="left", on='record_id')
system2.to_excel(results_folder / f'Anti-malarial recode - {model_name.replace("/", "__")}.xlsx', index=False)