
---

📁 **Scripts Directory**

This directory contains the main executable Python scripts for the **Bayou\_Sp25** project.

---

🚀 **`generate_periodic_summaries.py`**

### 🎯 **Purpose**

This script serves as the primary data-processing pipeline for aggregating raw environmental and event data into periodic summaries based on zip codes. It reads various raw data files (rainfall 🌧️, public/private sewage overflow events 🚧, 311 service calls 📞, and demographics 📊), processes them, and aggregates them into summaries over specified time periods (daily 📅, weekly 🗓️, etc.).

---

🛠️ **How to Use**

Run the script from your project's root directory via the command line:

```bash
python scripts/generate_periodic_summaries.py -r 3 
#Example usage, change 3 to other time resolution if needed.
```

**Command-line Arguments:**

* 📌 `-r`, `--resolution`: (**Required**) The time resolution in days for data aggregation (e.g., `1` for daily, `7` for weekly summaries).

**✨ Example:**

```bash
# Generate weekly summaries 🌟
python scripts/generate_periodic_summaries.py -r 7
```

---

📥 **Input Data**

The script relies on paths specified in `config/settings.py` 🗃️ to locate the necessary input files, including:

* Raw rainfall data 🌧️ (`RAW_RAINFALL_2223_PATH`, `RAW_RAINFALL_2324_PATH`)
* Raw public/private overflow events 🚧 (`ACTUAL_RAW_PUBLIC_EVENTS_PATH`, `ACTUAL_RAW_PRIVATE_EVENTS_PATH`)
* Pre-formatted daily 311 call data 📞 (`RAW_311_CALLS_PATH`)
* Site location mappings 📍 (`SITE_LOCATIONS_PATH`)
* Demographic data 📊 (`DEMOGRAPHICS_PATH`)
* Aggregation date range 📆 (`AGGREGATION_START_DATE`, `AGGREGATION_END_DATE`)

---

📤 **Generated Outputs**

The script produces aggregated summaries and saves them to directories defined in `config/settings.py`. Output filenames depend on the chosen resolution:

* **Rainfall Summary 🌦️**:
  `{RAINFALL_DATA_DIR}/rainfall_summary_{resolution}.csv`

* **Overflow Events Summary 🚧**:
  `{EVENT_COUNT_DIR}/overflow_summary_{resolution}.csv`

* **311 Calls Summary 📞**:
  `{EVENT_COUNT_DIR}/311_summary_{resolution}.csv`

Additionally, it generates an intermediate daily rainfall file 🗂️:

* Intermediate Rainfall Data:
  `{INTERMEDIATE_RAINFALL_PATH}` (e.g., `Data/Rainfall Complete/rainfall_complete.csv`)

---

🎉 **Happy Analyzing!**
