from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import json
import os
import gzip


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


def get_base_severity(cve_data):
    """
    Reliably extracts the baseSeverity from a CVE's 'metrics' object,
    prioritizing the latest CVSS version available.

    Args:
        cve_data (dict): A dictionary representing a single CVE
                         (e.g., from the NVD JSON feed).

    Returns:
        str: The baseSeverity string (e.g., "CRITICAL", "HIGH", "MEDIUM", "LOW")
             or None if no metrics are found.
    """
    if "metrics" not in cve_data:
        return None
        print("No metrics found in CVE data.")

    metrics = cve_data["metrics"]

    # Priority list of CVSS versions, from newest to oldest
    # Note: The user's "v41" is CVSS v4.0, with the key "cvssMetricV40"
    priority_order = [
        "cvssMetricV40",
        "cvssMetricV31",
        "cvssMetricV30",
        "cvssMetricV2"
    ]

    for key in priority_order:
        if key in metrics and metrics[key]:
            # The metric data is always in a list, even if there's only one.
            # We'll take the first entry.
            metric_entry = metrics[key][0]

            # CVSS v3.x and v4.0 store baseSeverity inside a 'cvssData' object
            if key != "cvssMetricV2":
                if "cvssData" in metric_entry and "baseSeverity" in metric_entry["cvssData"]:
                    return metric_entry["cvssData"]["baseSeverity"]

            # CVSS v2.0 stores baseSeverity directly on the metric entry
            elif key == "cvssMetricV2":
                if "baseSeverity" in metric_entry:
                    return metric_entry["baseSeverity"]

    # If no metrics with baseSeverity are found
    return None


unparsed_data_directory: str = "./Data/Unclean"
unparsed_data_file_names: List[str] = os.listdir(unparsed_data_directory)

unparsed_data_files: List[str] = []

for file_name in unparsed_data_file_names:
    file_path: str = os.path.join(unparsed_data_directory, file_name)

    if os.path.isfile(file_path):
        unparsed_data_files.append(file_path)

parsed_data: List[Dict] = []

for file_path in unparsed_data_files:
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:

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

            base_severity: SeverityLevel = SeverityLevel(get_base_severity(cve["cve"]) or "NONE")
            # print(base_severity, cve["cve"]["id"])

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






