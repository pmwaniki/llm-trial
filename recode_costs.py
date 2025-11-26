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
from tqdm import tqdm

load_dotenv(".env")

data_folder = Path(os.getenv("DATA_FOLDER"))
results_folder = Path(os.getenv("RESULTS_FOLDER"))

outcome_data = pd.read_parquet(data_folder / "outcome_data.parquet")

operational_data = pd.read_excel(data_folder / "Operation_Data-PATH.xlsx")
operational_data['visit_number'] = operational_data['visit_code'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))
outcome_data['visit_number'] = outcome_data['visit_number'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))

outcome_data = pd.merge(outcome_data, operational_data, how='left', on='visit_number')


model_name = "alibayram/medgemma:27b"


class Schema(BaseModel):
    """Schema for diagnoses and treatment of hypertension"""
    consult_fees: float = Field(description="Consultation fee", ge=0)
    investigations_ordered:str=Field(description="Investigations ordered")
    investigations_fees: float = Field(description="Investigations fee", ge=0)
    medications_ordered:str=Field(description="Medications ordered")
    medications_fees: float = Field(description="Medications fee", ge=0)
    antibiotics_cost: float = Field(description="Antibiotics cost", ge=0)
    antimalarials_cost: float = Field(description="Antimalarials cost", ge=0)
    antihistamines_cost: float = Field(description="antihistamines cost", ge=0)
    analgestics_cost: float = Field(description="Analgesics cost", ge=0)
    respiratory_cost: float = Field(description="Respiratory medicines cost", ge=0)
    gastrointestinal_cost: float = Field(description="Gastrointestinal medicines cost", ge=0)
    chronic_cost: float = Field(description="Chronic disease medicines cost", ge=0)
    reproductive_cost: float = Field(description="Reproductive health and STI medicines cost", ge=0)
    dermatological_cost: float = Field(description="Dermatological medicines cost", ge=0)
    neurology_cost: float = Field(description="Mental Health and Neurological medicines cost", ge=0)
    supplement_cost: float = Field(description="Vitamins and supplements cost", ge=0)
    supportive_cost: float = Field(description="Miscellaneous and supportive treatments cost", ge=0)

    other_fees: float = Field(description="Other fees", ge=0)
    description_others: str=Field(description="Description of other fees")






model = ChatOllama(model=model_name, temperature=0.)
# model with structured output
structured_model = model.with_structured_output(Schema)

sys_prompt = """
You are a health records information officer working in a primary care setting in Kenya.
Your task is to review medical bills in an outpatient department.

From the bill notes:
- Identify the amount of money paid for consultation: consult_fees
- Which Lab investigations were ordered, eg Chest radiographs, Malaria test, Blood glucose: investigations_ordered
- The total amount paid for investigations: investigations_fees
- A list of all the medications ordered: medications_ordered
- The total amount paid for medications: medications_fees
- A breakdown of drug cost by drug categories:

    1. Antibiotics (antibiotics_cost) e.g. amoxicillin, amoxicillin–clavulanate, azithromycin, doxycycline, ceftriaxone, ciprofloxacin, metronidazole (oral), trimethoprim–sulfamethoxazole, nitrofurantoin, erythromycin, clindamycin.
    2. Antihistamines (antihistamines_cost) e.g. cetirizine, loratadine, chlorpheniramine, fexofenadine, levocetirizine, promethazine.
    3. Antimalarials (antimalarials_cost) e.g. artemether–lumefantrine, quinine, injectable artesunate, atovaquone–proguanil, doxycycline (when used for prophylaxis).
    4. Analgesics, Antipyretics and Anti-inflammatories (analgestics_cost) e.g.  paracetamol, ibuprofen, diclofenac, naproxen, tramadol, aspirin, indomethacin, meloxicam.
    5. Respiratory Medicines (Non-antibiotic, Non-antihistamine) (respiratory_cost) e.g. salbutamol inhalers, salbutamol nebuliser solution, budesonide inhaler, beclomethasone inhaler, montelukast, ipratropium bromide, guaifenesin, dextromethorphan, carbocisteine.
    6. Gastrointestinal Medicines (gastrointestinal_cost) e.g. antacids (magnesium trisilicate, aluminium hydroxide), PPIs (omeprazole, pantoprazole, esomeprazole), H2 blockers (ranitidine, famotidine), antiemetics (ondansetron, metoclopramide, domperidone), ORS with zinc, loperamide, lactulose, bisacodyl, dicycloverine, simethicone.
    7. Chronic Disease Medicines (chronic_cost) e.g. amlodipine, enalapril, lisinopril, hydrochlorothiazide, nifedipine, atenolol, metformin, glibenclamide, gliclazide, sitagliptin, insulin (regular, NPH), simvastatin, atorvastatin, rosuvastatin, allopurinol (for gout), levothyroxine (thyroid).
    8. Reproductive Health, STI and Genitourinary Medicines (reproductive_cost) e.g. contraceptive pills (combined, progestin-only), Depo-Provera injections, hormonal implants (Jadelle, Implanon), emergency contraception (levonorgestrel), vaginal pessaries (clotrimazole pessaries, metronidazole vaginal tablets), intravaginal creams (butoconazole, miconazole), syndromic STI kits, acyclovir (for genital herpes).
    9. Dermatological Medicines (skin only) (dermatological_cost) e.g. topical steroids (hydrocortisone, betamethasone, clobetasol), topical antifungals (clotrimazole cream, ketoconazole cream), topical antibiotics (mupirocin, fusidic acid), acne treatments (benzoyl peroxide, adapalene), emollients, calamine lotion.
    10. Mental Health and Neurology Medicines (neurology_cost) e.g. SSRIs (fluoxetine, sertraline, escitalopram), SNRIs (venlafaxine), tricyclics (amitriptyline), benzodiazepines (diazepam, lorazepam), anticonvulsants (carbamazepine, sodium valproate, phenytoin), antipsychotics (risperidone, haloperidol), propranolol (for anxiety tremor).
    11. Vitamins and Supplements (supplement_cost) e.g. iron tablets, folic acid, ferrous fumarate, multivitamins, vitamin B complex, vitamin D, calcium supplements, zinc tablets/syrup, oral rehydration + micronutrient combos.
    12. Miscellaneous and supportive Treatments (supportive_cost)  e.g. IV fluids (normal saline, Ringer’s lactate, dextrose), eye drops (chloramphenicol, timolol, artificial tears), ear drops (chloramphenicol, clotrimazole ear drops), nasal saline spray, wound-care dressings (povidone–iodine, silver sulfadiazine), oral steroids (prednisolone), nebuliser diluent, antiseptic solutions.

The aggregated cost of antibiotic, antihistamines, steroids, and other drugs MUST be equal to total medication fees (medications_fee)

- Total amount paid for services besides consultation, investigations, and medication
- Output MUST be a single JSON object with ONLY these keys as shown:
{{
  "consult_fees": 200,
  "investigations_ordered": "eg Salmonella Antigen Test, Malaria Lab test, Chest X-ray",
  "investigations_fees": 2500,
  "medications_ordered": "Paracetamol, Amoxicillin, Cetrizine",
  "medications_fees":1100,
  "antibiotics_costs": 800,
  "antimalarials_cost": 0,
  "antihistamines_cost": 100,
  "analgestics_cost": 200,
  "respiratory_cost": 0,
  "gastrointestinal_cost": 0,
  "chronic_cost": 0,
  "reproductive_cost": 0,
  "dermatological_cost": 0,
  "neurology_cost": 0,
  "supplement_cost": 0,
  "supportive_cost": 0,
  "other_fees": 50,
  "description_others": "Registration fee"
}}
"""

# use prompt template as before
prompt_template = ChatPromptTemplate.from_messages(
    [("system", sys_prompt), ("user", "{text}")]
)

extracted_diagnoses = {}
error_visits = []
for _, row in tqdm(outcome_data.iterrows(), total=outcome_data.shape[0]):
    text = row.get('InvoiceItemsWithAmount')
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

        result_dict['Original text'] = row.get('InvoiceItemsWithAmount')
        result_dict['record_id'] = row.get('record_id', key)
        extracted_diagnoses[result_dict['record_id']] = result_dict
    except Exception as e:
        # save raw output for debugging
        raw = getattr(result, "content", None) or str(result)
        print(f"PARSE ERROR record={key} error={e}")
        print("RAW OUTPUT:", raw[:1000])
        error_visits.append(key)
        extracted_diagnoses[key] = {"parse_error": True, "raw": raw, "record_id": key}


with open(results_folder / f'Medical costs - {model_name.replace("/", "__")}.json', 'w') as f:
    json.dump(extracted_diagnoses, f)

extracted_diagnoses2={}
parsing_errors = []
for k,v in extracted_diagnoses.items():
    if v is None:
        parsing_errors.append(k)
    else:
        extracted_diagnoses2[k] = v

system = pd.DataFrame(extracted_diagnoses2.values())
system2 = pd.merge(system, outcome_data[['record_id', 'TotalInvoiceAmount']], how="left", on='record_id')
system2.to_excel(results_folder / f'Medical costs - {model_name.replace("/", "__")}.xlsx', index=False)

# outcome_data[['record_id','TotalInvoiceAmount']].to_parquet(results_folder / "cost_data.parquet")
