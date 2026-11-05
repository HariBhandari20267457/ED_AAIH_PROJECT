# %%
#installing the toolkit i needed to run the script

%pip install pandas numpy matplotlib scikit_learn


# %%
#getting the tools out of the toolbox

import pandas as pd # load pandas library, handles dataframes like csv files, and structured data
import numpy as np #numpy is a library for the Python programming language, for numerical arrays and fast matrix operations
import matplotlib.pyplot as plt 
#matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
import statistics as stats 
# statistics is a built-in Python library that provides functions for calculating mathematical statistics of numeric data.


# %%
# load two scv file into the pandas dataframe

triage = pd.read_csv("triage.csv") #reading the csv file into a pandas dataframe
edstays = pd.read_csv("edstays.csv")
triage.head() #displaying the first 5 rows of the dataframe

#edstays.head() #displaying the first 5 rows of the dataframe
# will do this laterto check demographics and other features of the data

# %%
triage.tail() #displaying the last 5 rows of the dataframe


# %%
# get the basic shape of data like number of rows and coulumns in the dataframe

triage.shape


# %%
# listing all the coulmns names
triage.columns


# %%
# checking datatypes and the missing values
triage.info()


# %%
# getting the statistics vallues of the data like mean, media, mode, maximum, minimum, average, st

triage.describe()

# %%
#acuity is your ESI level score is a measure of how sick a patient is when they arrive at the emergency department. It is usually determined by a triage nurse or other healthcare professional based on the patient's vital signs, symptoms, and medical history. The acuity score can help prioritize patients for treatment and determine the level of care they need.
triage['acuity'].value_counts() # this will give the raw count of each acuity score in the dataset
triage['acuity'].value_counts(normalize=True) #this will give the percentage of each acuity score in the dataset


# %%
#checking how much data is missing , column by column
triage.isnull().sum()
 #this will give, how many rows have nothing in each coulmns
 

# %%
#Look at the chief complaint column, which is a free-text field that describes the patient's main reason for seeking care in the emergency department. This column can contain a wide variety of complaints, ranging from specific symptoms (e.g., "chest pain", "shortness of breath") to more general complaints (e.g., "fever", "injury"). The chief complaint is an important piece of information that can help guide the initial assessment and treatment of the patient.
triage['chiefcomplaint'].head(20) 
#.head(20) Shows you 20 real examples of what patients' presenting complaints actually look like as text (e.g. "chest pain", "fall from standing").
triage['chiefcomplaint'].isnull().sum() # missing values in the chief complaint  for one column


# %%
edstays.head() #displaying the first 5 rows of the dataframe

# %%
edstays.tail() #displaying the last 5 rows of the dataframe


# %%
edstays.shape #shape of the dataframe, number of rows and columns

# %%
edstays.columns #listing all the coulmns names

# %%
edstays.info() #checking datatypes and the missing values

# %%
edstays.describe() #getting the statistics vallues of the data like mean, media, mode, maximum, minimum, average, st

# %%
edstays['race'].value_counts() # this will give the raw count of how many patients fall into each category
#edstays['arrival_transport'].value_counts() # this will give the raw count of how many patients arrived by each transport method


# %%
edstays.isnull().sum() #checking how much data is missing , column by column

# %%
edstays['disposition'].value_counts() # dispostion mean the what happened to the patient at the end of therir ED visit, Admitted, discharged, transfereed etc.

# %%
# validate the merge keys before combining the tables
print("Duplicate triage stay_ids:", triage["stay_id"].duplicated().sum())
print("Duplicate edstays stay_ids:", edstays["stay_id"].duplicated().sum())

df = triage.merge(edstays, on="stay_id", how="inner", validate="one_to_one")
print("Merged shape:", df.shape)
#.merge() joins the two tables using stay_id
#how='inner' only keeps the rows where a match exists in both tables
#validate='one_to_one' prevents an accidental many-to-many row multiplication

# %%
#drop the targets missing row first
df= df.dropna(subset=['acuity'])
df.shape
# we cant train the model where ESI level is missing so  this will remove the rows missing acuity nothing else



# %%
#separate the feature from the target
x= df.drop(columns=['acuity'])
y=df['acuity']
# x = everyting we wil use to make the prediciton
#y= the acutual answer we are trying to predict the ESI level



# %%
#  spliting the data into training and test sets while trying to keep each patient in only one set and the ESI class proportions similar in both

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

groups = x["subject_id_x"]          # identifies which patient each row belongs to. If a patient has multiple rows, those rows should stay together.

# 5 folds = each fold is about 20%; we take the first fold as the test set
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# divides the data into five folds. Each fold is roughly 20% of the rows; stratification tries to keep the ESI class distribution similar across folds, while grouping keeps patients together.
# shuffle=True shuffles the groups before splitting; random_state=42 makes the split repeatable.

train_idx, test_idx = next(sgkf.split(x, y, groups=groups))
# takes the first of the five possible splits. That fold becomes the test set; the other four folds become the training set

x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---- checks (all three must look right) ----
overlap = set(x_train["subject_id_x"]) & set(x_test["subject_id_x"]) #subject_id_x is used to keep patients separated
print("Patients in BOTH train and test:", len(overlap), "(must be 0)")
print("Train rows:", len(x_train), "| Test rows:", len(x_test),
      f"| Test share: {len(x_test) / len(x):.1%}")
print("\nESI class share (train vs test, should be similar):")
print(pd.DataFrame({"train": y_train.value_counts(normalize=True).sort_index(),
                    "test":  y_test.value_counts(normalize=True).sort_index()}).round(3))

# %%

groups = x["subject_id_x"]          # identifies which patient each row belongs to. If a patient has multiple rows, those rows should stay together.

# 5 folds = each fold is about 20%; we take the first fold as the test set
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# divides the data into five folds. Each fold is roughly 20% of the rows; stratification tries to keep the ESI class distribution similar across folds, while grouping keeps patients together.
# shuffle=True shuffles the groups before splitting; random_state=42 makes the split repeatable.

train_idx, test_idx = next(sgkf.split(x, y, groups=groups))
# takes the first of the five possible splits. That fold becomes the test set; the other four folds become the training set

x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---- checks (all three must look right) ----
overlap = set(x_train["subject_id_x"]) & set(x_test["subject_id_x"]) #subject_id_x is used to keep patients separated
print("Patients in BOTH train and test:", len(overlap), "(must be 0)")
print("Train rows:", len(x_train), "| Test rows:", len(x_test),
      f"| Test share: {len(x_test) / len(x):.1%}")
print("\nESI class share (train vs test, should be similar):")
print(pd.DataFrame({"train": y_train.value_counts(normalize=True).sort_index(),
                    "test":  y_test.value_counts(normalize=True).sort_index()}).round(3))

# %%

# the cell preserves all rows, replaces out-of-range vital-sign values with missing values, and reports what it changed. The chosen limits are assumptions for this analysis, so they should be checked against the dataset’s units and domain requirements.
# cell copies the training and test features, then marks vital-sign values outside specified ranges as missing (NaN). It does not delete rows or fill missing values yet.
# Impulsive vital signs: 2,268 training values and 610 test values

import numpy as np

#  Each pair gives the accepted minimum and maximum for that column. The bounds are inclusive, so a value equal to either limit is considered valid. Values outside the ranges are treated as impossible or invalid for this cleaning step.


limits = {"heartrate": (30, 220), "temperature": (90, 110), "sbp": (50, 220),
          "dbp": (30, 140), "resprate": (8, 60), "o2sat": (50, 100)}

x_train_clean = x_train.copy()
x_test = x_test.copy()
y_train_clean = y_train.copy()          # no rows are deleted, so labels stay as they are

rows = []
for col, (lo, hi) in limits.items(): # Check each vital sign in train and test
    for set_name, frame in (("train", x_train_clean), ("test", x_test)): #The loops process every vital-sign column first in the training data and then in the test data.
        was_missing = frame[col].isna().sum() #Counts how many values were already missing before this cleaning step.
        invalid = frame[col].notna() & ~frame[col].between(lo, hi) #Finds values that are not missing and fall outside the allowed range. The ~ means “not,” so it selects values that are not between the limits.
        frame.loc[invalid, col] = np.nan   # Replaces those invalid values with NaN. The rows remain in the dataset; only the cell values are changed.
        rows.append({"set": set_name, "vital": col, "range": f"{lo}-{hi}",
                     "already_missing": was_missing, "set_to_missing": invalid.sum()})
                     #The rows.append(...) line records the dataset, vital-sign name, accepted range, number already missing, and number newly changed to missing.
print(pd.DataFrame(rows).to_string(index=False))
print("Train rows:", len(x_train_clean), "| Test rows:", len(x_test))


# %%
# flags, then training medians, then fill 
vitals = list(limits.keys())

# 1. flags FIRST (they must be made before any filling), for both sets
for col in vitals:
    x_train_clean[col + "_missing"] = x_train_clean[col].isna().astype(int)
    x_test[col + "_missing"] = x_test[col].isna().astype(int)

# 2. medians from the TRAINING set only
medians = x_train_clean[vitals].median()

# 3. fill both sets with the same training medians
x_train_clean[vitals] = x_train_clean[vitals].fillna(medians)
x_test[vitals] = x_test[vitals].fillna(medians)

# 4. checks
print("Missing left (train, test):", x_train_clean[vitals].isna().sum().sum(),
      x_test[vitals].isna().sum().sum(), "(expect 0 0)")
print("\nShare of rows flagged (train vs test, should be similar and above 0):")
flag_cols = [c + "_missing" for c in vitals]
print(pd.DataFrame({"train": x_train_clean[flag_cols].mean(),
                    "test": x_test[flag_cols].mean()}).round(4))

# %%
import re #re provides regular expressions, used to recognize patterns such as pain ranges like "6-7".
import numpy as np
import pandas as pd
 
UNABLE = {"unable", "uta", "ua", "u/a", "critical", "c"}   # Defines text values that mean pain could not be scored. Later code can check whether a pain entry—after trimming spaces and converting to lowercase—matches one of these values.
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])} #Creates a dictionary mapping abbreviated month names to month numbers,
 # The cleaning code can use this to interpret values Excel may have converted from ranges into dates, such as "7-Jun" as a possible "6-7" pain range.
 
# READ-ONLY CHECK: how does ESI differ by pain group? 
raw = x_train_clean["pain"].astype("string").str.strip().str.lower() # Makes a text version of the pain column, removing surrounding spaces and converting text to lowercase. For example, " Unable " becomes "unable".
num = pd.to_numeric(raw, errors="coerce").astype(float) # Converts values that look numeric to numbers. Any value that cannot be converted becomes NaN for this temporary num series
is_empty = raw.isna().to_numpy(dtype=bool)
is_unable = raw.isin(UNABLE).fillna(False).to_numpy(dtype=bool)
in_scale = ((num >= 0) & (num <= 10)).to_numpy(dtype=bool)
is_number = num.notna().to_numpy(dtype=bool)
group = np.select( #np.select checks the conditions in order and assigns the first matching group. For instance, a missing value becomes "empty", while 13 becomes "number
    [is_empty, is_unable, in_scale, is_number],
    ["empty", "unable/critical", "score 0-10", "number outside 0-10 (e.g. 13)"],
    default="other text")
print(pd.Series(group).value_counts()) #Show counts and ESI proportions 
print("\nShare of each ESI level within each pain group:") # Prints the number of training rows in each pain group.
print(pd.crosstab(group, y_train_clean.values, normalize="index").round(3))
 #Builds a table showing the proportion of each ESI level within each pain group. normalize="index" means each group’s row adds up to 1 (or 100%); .round(3) rounds the proportions to three decimal places.

# %%
# cleaning the pain column in both the training and test data, while keeping track of values that were missing or unusual.
#  the cell turns pain entries into usable numeric scores, preserves flags describing problematic original values, 
# fills missing scores consistently, and prepares complaint text for text processing.
def parse_pain(value):
    """ Return (pain_score_or_nan, kind) where kind is '', 'unable' or 'odd'."""
    s = str(value).strip().lower()
    if s in UNABLE:
        return np.nan, "unable"
    # Excel turned ranges like "6-7" into dates such as "7-Jun"  read them back as ranges
    m = re.fullmatch(r"(\d{1,2})-([a-z]{3})", s)
    if m and m.group(2) in MONTHS:
        lo, hi = MONTHS[m.group(2)], int(m.group(1))
        return ((lo + hi) / 2, "") if 0 <= lo <= hi <= 10 else (np.nan, "odd")
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", s)            # a plain range like "6-7"
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return ((lo + hi) / 2, "") if 0 <= lo <= hi <= 10 else (np.nan, "odd")
    try:
        v = float(s)
    except ValueError:
        return np.nan, "odd"                                # any other text
    return (v, "") if 0 <= v <= 10 else (np.nan, "odd")     # 13 etc. are not on a 0-10 scale
 
 
def clean_pain(df):   # convert pain to a clean number (train AND test)
    uniq = df["pain"].dropna().unique()
    parsed = {u: parse_pain(u) for u in uniq}
    df["pain_empty"] = df["pain"].isna().astype(int)                       # nothing recorded
    df["pain_unable"] = df["pain"].map({u: int(p[1] == "unable") for u, p in parsed.items()}).fillna(0).astype(int)
    df["pain_odd"] = df["pain"].map({u: int(p[1] == "odd") for u, p in parsed.items()}).fillna(0).astype(int)
    df["pain"] = df["pain"].map({u: p[0] for u, p in parsed.items()}).astype(float)   # flags made BEFORE filling
 
 
# Safety guard: This stops the cell if the training pain column is already numeric. It is intended to prevent accidentally cleaning the same column twice.
if pd.api.types.is_float_dtype(x_train_clean["pain"]) or pd.api.types.is_integer_dtype(x_train_clean["pain"]):
    raise RuntimeError("pain is already numeric - it was cleaned earlier in this session. "
                       "Kernel > Restart & Run All, then run this cell only once.")
 # Clean training and test data, then fill missing scores
clean_pain(x_train_clean) 
clean_pain(x_test)
 
pain_median = x_train_clean["pain"].median()                # median is then calculated from the training data only:
x_train_clean["pain"] = x_train_clean["pain"].fillna(pain_median)
x_test["pain"] = x_test["pain"].fillna(pain_median)
 
# text columns must not be empty for TF-IDF, chiefcomplaint lines replace missing text with ""
x_train_clean["chiefcomplaint"] = x_train_clean["chiefcomplaint"].fillna("")
x_test["chiefcomplaint"] = x_test["chiefcomplaint"].fillna("")
 
print("pain median (train):", pain_median) #The training-set median used for filling.
print("Share flagged (train vs test):")    #The proportion of rows flagged as missing, unable, or unusual in each dataset.
print(pd.DataFrame({"train": x_train_clean[["pain_empty", "pain_unable", "pain_odd"]].mean(),
                    "test": x_test[["pain_empty", "pain_unable", "pain_odd"]].mean()}).round(4))
                    # The resulting data type and how many missing pain scores remain.
print("dtype:", x_train_clean["pain"].dtype, "| empty pain left:",
      x_train_clean["pain"].isna().sum(), x_test["pain"].isna().sum())

# %%
# Check the cleaned features and their labels
print("Training features:", x_train_clean.shape)
print("Testing features:", x_test.shape)
print("Training labels:", y_train_clean.shape)
print("Testing labels:", y_test.shape)

print("Training feature/label rows match:",
      len(x_train_clean) == len(y_train_clean))
print("Testing feature/label rows match:",
      len(x_test) == len(y_test))

print("\nESI class share (train vs test):")
print(pd.DataFrame({
    "train": y_train_clean.value_counts(normalize=True).sort_index(),
    "test": y_test.value_counts(normalize=True).sort_index()
}).round(3))
#Replacing missing values in the chiefcomplaint column with an empty string (""):
#fillna("") means where a value is missing (NaN), use an empty string instead.
x_train_clean["chiefcomplaint"] = x_train_clean["chiefcomplaint"].fillna("")
x_test["chiefcomplaint"] = x_test["chiefcomplaint"].fillna("")

# %%
#checking that both tables have original vitals and 6 new _missing flag columns signs
print(x_train_clean.shape)
print(x_test.shape)
print(x_train_clean.columns.tolist())
# .columns.tolist()) prints all column names in the training data as a Python list

# %% [markdown]
# Train/test split: done and verified.
# Impossible vital-sign values: set to missing in both train and test using the same limits; no rows are removed.
# Missing values: flags are created before imputation; medians are learned from training data and applied to both sets.

# %%
# checking which categories exist in each field before encoding
#x_train_clean['gender'].value_counts()
#x_train_clean['race'].value_counts()
x_train_clean['arrival_transport'].value_counts()



# %%
# checking which categories exist in each field before encoding
x_train_clean['race'].value_counts()


# %%
# checking which categories exist in each field before encoding
x_train_clean['gender'].value_counts()


# %%
# checking if there are any mismatched columns
print(x_train_clean.shape)
print(x_test.shape)

missing_in_test = set (x_train_clean.columns) - set(x_test.columns)
missing_in_train = set (x_test.columns) - set(x_train_clean.columns)

print("columns in the train but not the test:", missing_in_test)
print("Columns in the  test but not train:", missing_in_train)



# %%
print(x_train_clean["pain"].value_counts(dropna=False).head(25))
print("Not convertible to a number:",
      pd.to_numeric(x_train_clean["pain"], errors="coerce").isna().sum())

# %%
#Encoding for the race, gender and arrival_transport
# converts the non-numeric categorical columns into numeric values so a machine learning model can use them.

x_train_clean['gender'] = x_train_clean['gender'].map({'M':0,'F':1})
x_test['gender'] = x_test['gender'].map({'M':0, 'F':1})
#gender is encoded manually:- - `M` → `0`and  `F` → `1`

categorical_cols=['race','arrival_transport']

x_train_clean = pd.get_dummies(x_train_clean, columns=categorical_cols, dtype=int)
x_test = pd.get_dummies(x_test, columns=categorical_cols, dtype=int)

# Match the test schema to the training schema after one-hot encoding.
x_test = x_test.reindex(columns=x_train_clean.columns, fill_value=0)

# This is a simple mapping because gender only has two categories.
#- `pd.get_dummies(...)` is used for the other categorical columns:  `race` and `arrival_transport`
#This turns a column like: race with values white, asian into multiple binary columns like: race_white, race_asian with values `0` or `1`.

print(x_train_clean.shape)
print(x_test.shape)
print("Columns aligned:", x_train_clean.columns.equals(x_test.columns))

# %%
# Saving the cleaned, split, encoded data as new files in your project folder

x_train_clean.to_csv("x_train_clean.csv", index= False)
y_train_clean.to_csv("y_train_clean.csv", index= False)
x_test.to_csv("x_test_clean.csv", index = False)
y_test.to_csv("y_test_clean.csv", index = False)


# %% [markdown]
# 
# # Summary: Data_prep
# 
# Prepare one clean, leakage_checked train/test dataset that is used by all 3 models - text-only, structured-only and hybrid. Target variables: acuity- ESI level 1-5.
# 
# Summary of the steps:
# wing steps:
# 
# 1 Loading and inspecting both MIMIC-IV-ED tables
# 2 Merging them into one working dataset
# 3 Targeting to remove the stays with no ESI level recorded found  418,100 stays kept with 334480 train and 8360 test.
# 4 Spliting the patient level used StratifiedGroupKFold to split the data by patient ID. This completely eliminates data leakage, resulting in 0 patient overlap between train and test sets. ESI shares identical in both sets (1: 5.7%, 2: 33.3%, 3: 53.8%, 4: 6.8%, 5: 0.3%) 
# 5 Vital Sign Handling and Imputaiton: Flagged the implausible vitals like impossible heart rates and imputed them alongside naturally missing values using training set medians. This strictly prevents test set information from leaking into the training data. Data Train: 2,268 values; test: 610 values, with median sets no missing vitals left (0 in train, 0 in test).
# 6 Pain Score: Converted messy free text pain entries like unable, critical into distinct flags - pain_empty (1.5%), pain_unable (2.7%), pain_odd (3.3%), where gap filled with the trainig median 4.0.
# 7 Chief Complaint: Missing text replaces with an empty string 15 train and 2 test rows which is needed for TF-IDF
# 8 Encoding: Gender as M=0, F=1, encoding of race and arrival_transport
# 9 Saved Files: X_train_clean.csv, y_train_clean.csv, x_test_clean.csv, y_test_clean.csv with shapes with train (334480, 63) and  test (83620, 63).
# 
# Here, Vital-sign limits (values outside these were set to missing): heart rate 30-220,
# temperature 90-110, systolic BP 50-220, diastolic BP 30-140, respiratory rate 8-60,
# SpO2 50-100.
# 
# Problems found and fixed
# 1. Patient overlap for data leakage. The original row-level trian_test_split placed 33,870 patients in boths sets and 64.7% of test rows came form pateients also in training. It was replaced with the patient level stratified split in step 2 and verified as 0 overlaped.
# 2. Row deletion bias. The first cleaning rule deleted 2148 trainig rows with impossible vital signs. These rows were skewed towards ESI 1(11.5% of removed rows vs 5.7% overall), so the rule was replaced by setting valuses to missing as on step 3, which keeps all pateint and cleans the test set by the same rule.
# 3. Pain recored as free text. About 5% of pain entires were not numbers. These entries differ strongly by ESI level with ESI1 empty 52%, unable/critical 39%, normal score 3.6%, so they were kept as flags rather than discarded
# 4. Order of operations. Flags are created before imputaion, and  imputation values come from the training set only, so no test information reaches the training data.
# 
# 
# 


