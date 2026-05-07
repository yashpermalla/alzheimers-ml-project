
RANDOM_STATE = 42
TARGET = "Diagnosis"

ID_COLUMNS = ["PatientID", "DoctorInCharge"]

CATEGORICAL_FEATURES = ["Ethnicity", "EducationLevel"]

BINARY_FEATURES = [
    "Gender",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

CONTINUOUS_FEATURES = [
    "Age",
    "BMI",
    "AlcoholConsumption",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "SystolicBP",
    "DiastolicBP",
    "CholesterolTotal",
    "CholesterolLDL",
    "CholesterolHDL",
    "CholesterolTriglycerides",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
]

SELECTED_FEATURES = [
    "Ethnicity",
    "EducationLevel",
    "Gender",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "Age",
    "BMI",
    "AlcoholConsumption",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "SystolicBP",
    "DiastolicBP",
    "CholesterolTotal",
    "CholesterolLDL",
    "CholesterolHDL",
    "CholesterolTriglycerides",
]

GENDER_MAP = {0: "Female", 1: "Male"}

ETHNICITY_MAP = {
    0: "Caucasian",
    1: "African American",
    2: "Asian",
    3: "Other",
}

EDUCATION_MAP = {
    0: "None",
    1: "High School",
    2: "Bachelor's",
    3: "Higher",
}
