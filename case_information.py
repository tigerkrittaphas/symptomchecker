import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
import os

load_dotenv()
API_VERSION = os.getenv("API_VERSION")

cases = pd.read_csv("data/all_vignettes.csv")
cases = cases[['disease', 'vignette']]

class CaseInformation(BaseModel):
    age: int = Field(..., description="Age of the patient in years")
    gender: Literal["male", "female"] = Field(..., description="Gender of the patient")
    body_systems: Literal[
        'Hematology',
        'Cardiovascular',
        'Neurology',
        'Endocrine',
        'ENT (Ear, Nose, and Throat)',
        'GI (Gastrointestinal)',
        'Obstetrics and gynecology',
        'Infectious',
        'Respiratory',
        'Orthopedics and rheumatology',
        'Ophthalmology',
        'Dermatology',
        'Urology',
        'Nephrology'
    ] = Field(..., description="Classification of systems involved in the case")
    underlying_diseases: List[Literal[
        "T2DM",
        "HT",
        "DLP",
        "CKD",
        "Thyroid Disease",
        "Cancer",
        "COPD",
        "Asthma",
        "Obesity",
        "Other"
    ]] = Field(..., description="List of underlying diseases or conditions")

def extract_case_information(case_text: str) -> CaseInformation:
    system_prompt = """You are a medical expert. Extract the following information from the case description:
    - Patient age
    - Patient gender (male/female)
    - Body systems involved (select from the provided list)
    - Underlying diseases or medical conditions
    
    Analyze the case carefully and provide structured output."""
    
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract information from this case: {case_text}"}
    ]
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
        api_version=API_VERSION
    )
    
    response = client.beta.chat.completions.parse(
        model=os.getenv("AZURE_OPENAI_MODEL"),
        messages=message,
        response_format=CaseInformation
    )
    
    return response.choices[0].message.parsed

# Process all cases
extracted_cases = []
for index, row in cases.iterrows():
    print(f"Processing case {index + 1}/{len(cases)}: {row['disease']}")
    case_info = extract_case_information(row['vignette'])
    extracted_cases.append({
        'disease': row['disease'],
        'age': case_info.age,
        'gender': case_info.gender,
        'body_systems': case_info.body_systems,
        'underlying_diseases': case_info.underlying_diseases
    })

# Convert to DataFrame for easy handling
results_df = pd.DataFrame(extracted_cases)

# Save the results to a CSV file
results_df.to_csv("data/case_info.csv", index=False)
