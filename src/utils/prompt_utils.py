"""Diagnosis labels and response normalization."""

VALID_DIAGNOSES_LIST = [
    "Possible NSTEMI / STEMI", "Spontaneous rib fracture", "Pulmonary embolism",
    "Pulmonary neoplasm", "URTI", "Croup", "Sarcoidosis", "HIV (initial infection)",
    "Inguinal hernia", "Spontaneous pneumothorax", "Bronchospasm / acute asthma exacerbation",
    "Viral pharyngitis", "Bronchiolitis", "Pancreatic neoplasm", "Guillain-Barré syndrome",
    "Chagas", "Allergic sinusitis", "Acute rhinosinusitis", "PSVT", "Panic attack",
    "Epiglottitis", "Bronchiectasis", "Bronchitis", "Pericarditis",
    "Acute COPD exacerbation / infection", "Ebola", "Chronic rhinosinusitis",
    "Acute otitis media", "Larygospasm", "Influenza", "Stable angina", "Myasthenia gravis",
    "Myocarditis", "SLE", "GERD", "Anemia", "Cluster headache", "Localized edema",
    "Anaphylaxis", "Atrial fibrillation", "Acute pulmonary edema", "Acute laryngitis",
    "Acute dystonic reactions", "Boerhaave", "Pneumonia", "Tuberculosis",
    "Whooping cough", "Unstable angina", "Scombroid food poisoning"
]
VALID_DIAGNOSES_STR = "\n".join([f"- {d}" for d in VALID_DIAGNOSES_LIST])


formatted_disease_list = "\n".join(
    [f"{i+1:02d}: {name}" for i, name in enumerate(VALID_DIAGNOSES_LIST)]
)


DIAGNOSIS_ID_MAP = {
    f"{i+1:02d}": name for i, name in enumerate(VALID_DIAGNOSES_LIST)
}


NAME_TO_ID_MAP = {
    name: f"{i+1:02d}" for i, name in enumerate(VALID_DIAGNOSES_LIST)
}


def get_diagnosis_id(name: str) -> str:

    return NAME_TO_ID_MAP.get(name)

def get_diagnosis_name(diagnosis_id: str) -> str:

    return DIAGNOSIS_ID_MAP.get(diagnosis_id, "Unknown")

def normalize_diagnosis_id(raw_id: str) -> str:

    if not raw_id:
        return raw_id

    cleaned = str(raw_id).strip().strip('"').strip("'").strip()

    if cleaned in DIAGNOSIS_ID_MAP:
        return cleaned

    try:
        num = int(cleaned)

        if 1 <= num <= 49:
            return f"{num:02d}"

        if num > 49 and len(cleaned) == 3:

            first_two = cleaned[:2]
            if first_two in DIAGNOSIS_ID_MAP:
                return first_two

            last_two = cleaned[1:]
            if last_two in DIAGNOSIS_ID_MAP:
                return last_two

            without_leading_zero = cleaned.lstrip('0')
            if len(without_leading_zero) <= 2:
                formatted = f"{int(without_leading_zero):02d}"
                if formatted in DIAGNOSIS_ID_MAP:
                    return formatted
    except ValueError:
        pass

    return raw_id

def normalize_diagnosis_id_list(raw_ids: list) -> list:

    if not raw_ids or not isinstance(raw_ids, list):
        return raw_ids

    normalized = []
    for raw_id in raw_ids:
        normalized_id = normalize_diagnosis_id(str(raw_id))
        normalized.append(normalized_id)
    return normalized

if __name__ == "__main__":
    print(formatted_disease_list)
    print("\n--- ID to Name Map ---")
    for k, v in DIAGNOSIS_ID_MAP.items():
        print(f"  {k}: {v}")

