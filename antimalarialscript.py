import pandas as pd
import re
from dotenv import load_dotenv
from pathlib import Path
import os


# load environmental variables
load_dotenv(".env")
data_folder=Path(os.getenv("DATA_FOLDER"))
results_folder=Path(os.getenv("RESULTS_FOLDER"))
# ====== FILE PATH ======
input_path = data_folder / "clinical_data-PATH.xlsx"         # your input file
output_path = results_folder/"Antimalialresults.xlsx"  # output file

# ====== LOAD DATA ======
df = pd.read_excel(input_path)

# Standardize column names
df.columns = [c.strip() for c in df.columns]

# Identify relevant columns
lab_col = "LabTest"
rx_col = "Rx"

# ====== 1. IMPROVED: Detect malaria lab tests (more specific) ======
# More specific patterns that indicate actual malaria tests
lab_pattern = re.compile(
    r'\b(malaria\s*(rdt|rapid\s*test|rapid\s*diagnostic|test|smear|blood\s*(film|slide)|microscopy|result|antigen)|'
    r'rdt\s*(for\s*)?malaria|'
    r'rapid\s*(diagnostic\s*)?test\s*(for\s*)?malaria|'
    r'blood\s*(film|slide)\s*(for\s*)?malaria|'
    r'malaria\s*parasite|'
    r'mps\b|'
    r'pf\s*antigen)',
    flags=re.IGNORECASE
)

def detect_malaria_lab(text):
    if pd.isna(text):
        return "No"
    
    text_str = str(text)
    
    # Look for specific malaria test sections
    lines = text_str.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        # Check if this line contains malaria test indicators
        if lab_pattern.search(line_lower):
            return "Yes"
    
    return "No"

df["Was a Malaria Test done?"] = df[lab_col].apply(detect_malaria_lab)

# ====== 2. Detect antimalarial prescriptions with doxycycline logic ======
def detect_antimalarial_conditional(text, malaria_flag):
    if pd.isna(text):
        return "No"
    txt = str(text).lower()

    # Other antimalarials
    other_antimalarials = ["artemether/lumefantrine", "coartem", "artemether injection", "artesunate sodium injection"]
    has_other = any(t in txt for t in other_antimalarials)

    # Doxycycline present
    has_doxy = "doxycycline" in txt

    if has_other:
        return "Yes"
    elif has_doxy and malaria_flag == "Yes":
        return "Yes"
    else:
        return "No"

df["Was Antimalarial Prescribed?"] = df.apply(
    lambda r: detect_antimalarial_conditional(r[rx_col], r["Was a Malaria Test done?"]),
    axis=1
)

# ====== 3. IMPROVED: Detect malaria test positivity/negativity ======
positive_pattern = re.compile(
    r'\b(positive|\+ve|\+\+|\+\+\+|detected|found|pf|falciparum|plasmodium|vivax|malaria\s*antigen\s*present|parasites?\s*seen)\b',
    flags=re.IGNORECASE
)
negative_pattern = re.compile(
    r'\b(negative|-ve|not\s*detected|none|absent|no\s*malaria|malaria\s*negative|neg|no\s*mps\s*seen|no\s*parasites?)\b',
    flags=re.IGNORECASE
)

def detect_test_result(text, malaria_flag):
    if malaria_flag != "Yes":
        return "Unknown"
    if pd.isna(text):
        return "Unknown"
    
    txt = str(text)
    
    # Look for malaria-related sections more broadly
    malaria_sections = []
    lines = txt.split("\n")
    
    for line in lines:
        line_lower = line.lower()
        # Only include lines that specifically mention malaria tests
        if (lab_pattern.search(line_lower) or 
            "malaria" in line_lower and any(word in line_lower for word in ["result", "test", "rdt", "microscopy", "blood film", "blood slide"])):
            malaria_sections.append(line)
    
    # If no specific malaria sections found, use the entire text
    if not malaria_sections:
        malaria_text = txt
    else:
        malaria_text = " ".join(malaria_sections)
    
    has_pos = bool(positive_pattern.search(malaria_text))
    has_neg = bool(negative_pattern.search(malaria_text))
    
    # Improved logic for conflict resolution
    if has_pos and has_neg:
        # If both positive and negative indicators found, look for clearer patterns
        if "no mps seen" in malaria_text.lower() or "no parasites seen" in malaria_text.lower():
            return "No"
        elif "parasites seen" in malaria_text.lower():
            return "Yes"
        else:
            return "Unknown"
    elif has_pos:
        return "Yes"
    elif has_neg:
        return "No"
    else:
        return "Unknown"

df["Was the malaria test positive?"] = df.apply(
    lambda r: detect_test_result(r[lab_col], r["Was a Malaria Test done?"]),
    axis=1
)

# ====== 4. Filter rows with at least one tag ======
df_tagged = df[
    (df["Was a Malaria Test done?"] == "Yes") |
    (df["Was Antimalarial Prescribed?"] == "Yes") |
    (df["Was the malaria test positive?"] == "Yes")
]

# ====== 5. Save filtered output ======
df_tagged.to_excel(output_path, index=False)

# ====== 6. Print summary ======
summary = {
    "Total rows in original data": len(df),
    "Rows with at least one tag": len(df_tagged),
    "Malaria test done": (df_tagged["Was a Malaria Test done?"] == "Yes").sum(),
    "Antimalarial prescribed": (df_tagged["Was Antimalarial Prescribed?"] == "Yes").sum(),
    "Malaria positive": (df_tagged["Was the malaria test positive?"] == "Yes").sum(),
    "Malaria negative": (df_tagged["Was the malaria test positive?"] == "No").sum(),
    "Malaria unknown result": (df_tagged["Was the malaria test positive?"] == "Unknown").sum(),
    "Both test and drug": ((df_tagged["Was a Malaria Test done?"] == "Yes") & (df_tagged["Was Antimalarial Prescribed?"] == "Yes")).sum(),
}

print("=== Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print(f"\nTagged file saved to: {output_path}")