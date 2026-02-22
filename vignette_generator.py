import argparse
import json
import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field


load_dotenv()

API_VERSION = os.getenv("API_VERSION")
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-2024-11-20")

SYSTEM_PROMPT_TEMPLATE = """You are a medical professor. You will be given a diagnosis from the user.
From that diagnosis, you will generate a case vignette containing
comprehensive case information. The case must be based on the Thai
population and follow this structure:

- Demographic: age, gender
- Chief complaint
- History (Open questioning)
- History (Direct questioning)
- Family History
- Past Medical History
- Medication History
- Social History: smoking, alcohol, occupation
- Allergy History

Generate the case vignette for {{diagnosis}}."""


class Demographic(BaseModel):
    age: int = Field(..., description="Age in years")
    gender: Literal["male", "female"] = Field(..., description="Patient gender")


class ChiefComplaint(BaseModel):
    symptom: str = Field(..., description="Main symptom")
    duration: str = Field(..., description="Duration of the chief complaint")


class SocialHistory(BaseModel):
    smoking: str = Field(..., description="Smoking history")
    alcohol: str = Field(..., description="Alcohol history")
    occupation: str = Field(..., description="Occupation")


class CaseVignette(BaseModel):
    demographic: Demographic
    chief_complaint: ChiefComplaint
    history_open_questioning: List[str]
    history_direct_questioning: List[str]
    family_history: str
    past_medical_history: str
    medication_history: str
    social_history: SocialHistory
    allergy_history: str


def load_diseases(disease_file: Path) -> List[str]:
    raw_lines = disease_file.read_text(encoding="utf-8").splitlines()
    diseases = [line.strip() for line in raw_lines if line.strip()]
    # Preserve order and remove duplicates.
    return list(dict.fromkeys(diseases))


def make_safe_filename(name: str) -> str:
    safe = name.strip()
    for bad in ["/", "\\", ":"]:
        safe = safe.replace(bad, " - ")
    return f"{safe}.json"


def list_to_info_map(items: List[str]) -> dict:
    return {f"info{i + 1}": item for i, item in enumerate(items)}


def to_output_schema(vignette: CaseVignette) -> dict:
    return {
        "Demographic": {
            "Gender": vignette.demographic.gender,
            "Age": str(vignette.demographic.age),
        },
        "Chief complaint": {
            "symptom": vignette.chief_complaint.symptom,
            "duration": vignette.chief_complaint.duration,
        },
        "History (Open questioning)": list_to_info_map(
            vignette.history_open_questioning
        ),
        "History (Direct questioning)": list_to_info_map(
            vignette.history_direct_questioning
        ),
        "Family History": vignette.family_history,
        "Past Medical History": vignette.past_medical_history,
        "Medication History": vignette.medication_history,
        "Social History": {
            "smoking": vignette.social_history.smoking,
            "alcohol": vignette.social_history.alcohol,
            "occupation": vignette.social_history.occupation,
        },
        "Allergy History": vignette.allergy_history,
    }


def generate_vignette(
    client: AzureOpenAI,
    diagnosis: str,
    model: str,
) -> CaseVignette:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{{diagnosis}}", diagnosis)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": diagnosis},
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=CaseVignette,
    )
    return response.choices[0].message.parsed


def create_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")

    missing = []
    if not api_key:
        missing.append("AZURE_API_KEY")
    if not azure_endpoint:
        missing.append("OPENAI_AZURE_ENDPOINT")
    if not API_VERSION:
        missing.append("API_VERSION")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {joined}")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=API_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate case vignette JSON files from a disease list."
    )
    parser.add_argument(
        "--disease-file",
        type=Path,
        default=Path("data/disease_list.txt"),
        help="Path to disease list text file (one diagnosis per line).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/vignette"),
        help="Directory to write generated JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Azure OpenAI deployment/model name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite JSON files if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of diagnoses to process.",
    )
    args = parser.parse_args()

    diseases = load_diseases(args.disease_file)
    if args.limit is not None:
        diseases = diseases[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = create_client()

    processed = 0
    skipped = 0
    failed = 0

    for index, disease in enumerate(diseases, start=1):
        output_path = args.output_dir / make_safe_filename(disease)

        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(diseases)}] Skipping existing file: {output_path.name}")
            continue

        try:
            print(f"[{index}/{len(diseases)}] Generating vignette: {disease}")
            vignette = generate_vignette(client, disease, args.model)
            payload = to_output_schema(vignette)
            output_path.write_text(
                json.dumps(payload, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            processed += 1
        except Exception as error:
            failed += 1
            print(f"[{index}/{len(diseases)}] Failed for '{disease}': {error}")

    print(
        f"Finished. Generated: {processed}, Skipped: {skipped}, Failed: {failed}, "
        f"Output directory: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
