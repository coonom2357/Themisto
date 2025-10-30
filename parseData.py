from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import json
import os


class SeverityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    NONE = "NONE"


class ParsedEntry(BaseModel):
    id: str
    description: str
    severity: SeverityLevel


unparsed_data_directory: str = "./Data/Unclean"
unparsed_data_file_names: List[str] = os.listdir(unparsed_data_directory)

unparsed_data_files: List[str] = []

for file_name in unparsed_data_file_names:
    file_path: str = os.path.join(unparsed_data_directory, file_name)

    if os.path.isfile(file_path):
        unparsed_data_files.append(file_path)

parsed_data: List[Dict] = []

for file_path in unparsed_data_files:
    with open(file_path, 'r') as file:
        data = json.load(file)

        vulnerabilities: List[Dict] = data["vulnerabilities"]

        for cve in vulnerabilities:
            descs: List[Dict[str, str]] = cve["cve"]["descriptions"]
            desc: str = ""

            # Only want English descriptions
            for description in descs:
                if description["lang"] == "en":
                    desc = description["value"]
                    break

            if desc == "":
                # print(f"{cve['cve']['id']} | NO DESCRIPTION\n")
                continue

            metrics: List[Dict] = cve["cve"].get("metrics", {}).get("cvssMetricV3", []) or cve["cve"].get("metrics", {}).get("cvssMetricV40", [])

            if not metrics:
                # print(f"{cve['cve']['id']} | NO CVSS DATA | {desc}\n")
                continue

            base_severity: SeverityLevel = SeverityLevel(metrics[0]["cvssData"]["baseSeverity"])

            if base_severity == SeverityLevel.NONE:
                # print(f"{cve[`'cve']['id']} | NO SEVERITY DATA | {desc}\n")
                continue

            parsed_data.append({
                "id": cve["cve"]["id"],
                "description": desc,
                "severity": base_severity.value
            })

df = pd.DataFrame(parsed_data)
df.to_csv("./Data/Parsed/parsed_data.csv", index=False, quoting=1)






