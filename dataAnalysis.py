from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
from ydata_profiling import ProfileReport

df = pd.read_csv("./Data/Parsed/parsed_data.csv")

print(df['severity'].value_counts())
print(f"Number of samples: {len(df)}")

profile = ProfileReport(df, title="Vulnerability Data Profile Report")
profile.to_file("./Data/Parsed/vulnerability_data_profile_report.html")

# Plot severity distribution
severity_counts: Series = df['severity'].value_counts()
severity_order: List[str] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
severity_counts: Series = severity_counts.reindex(severity_order, fill_value=0)
plt.figure(figsize=(10, 6))
severity_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
plt.title('Vulnerability Severity Distribution')
plt.xlabel('Severity Level')
plt.ylabel('Number of Vulnerabilities')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('./Data/Parsed/severity_distribution.png')
plt.close()

# Plot description length distribution
df['description_length'] = df['description'].apply(len)
plt.figure(figsize=(10, 6))
plt.hist(df['description_length'], bins=30, color='blue', edgecolor='black')
plt.title('Description Length Distribution')
plt.xlabel('Description Length (characters)')
plt.ylabel('Number of Vulnerabilities')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('./Data/Parsed/description_length_distribution.png')
plt.close()

# Use log scale for y-axis
plt.figure(figsize=(10, 6))
plt.hist(df['description_length'], bins=30, color='blue', edgecolor='black', log=True)
plt.title('Description Length Distribution (Log Scale)')
plt.xlabel('Description Length (characters)')
plt.ylabel('Number of Vulnerabilities (log scale)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('./Data/Parsed/description_length_distribution_log.png')
plt.close()
