"""
config.py
=========
Central configuration for the GAD preschool ML pipeline.
Edit TRAIN_PATH and OUTPUT_DIR before running on a new machine.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH = "/kaggle/input/datasets/bariscelebi/anskiyetedata/Training Data.xlsx"
OUTPUT_DIR = "/kaggle/working/"

# ── Experiment ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 10

# ── Figure rendering ───────────────────────────────────────────────────────
DPI        = 300
FONT_TITLE = 16
FONT_AXIS  = 14
FONT_TICK  = 12
FONT_LEGEND = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    "Naive Bayes":         "#3B8BD4",
    "Logistic Regression": "#1D9E75",
    "Decision Tree":       "#EF9F27",
    "KNN":                 "#D85A30",
    "grid":                "#E0DED8",
    "bg":                  "#F8F8F6",
}

MODEL_SHORT = {
    "Naive Bayes":         "NB",
    "Logistic Regression": "LR",
    "Decision Tree":       "DT",
    "KNN":                 "KNN",
}

# ── English short labels for every PAPA feature ────────────────────────────
FEATURE_LABELS_EN = {
    "Irritability":
        "Irritability",
    "Increased unnecessary whole body movements in specific situations":
        "Unnecessary body movements",
    "Difficulty concentrating on tasks or play activity independently":
        "Difficulty concentrating (independent)",
    "Difficulty concentrating on adult-directed tasks or play activities":
        "Difficulty concentrating (adult-directed)",
    "Inattention":
        "Inattention",
    "Fear about possible harm befalling major attachment figures":
        "Fear of harm to attachment figures",
    "Fear about calamitous separation":
        "Fear of calamitous separation",
    "Avoidance of being alone":
        "Avoidance of being alone",
    "Anticipatory distress/resistance to separation":
        "Anticipatory distress at separation",
    "Withdrawal when attachement figure absent":
        "Withdrawal when attachment figure absent",
    "Actual distress when attachment figure absent":
        "Distress when attachment figure absent",
    "Complaints of physical symptoms when separation from major attachment figure is anticipated":
        "Physical symptoms at anticipated separation",
    "Parent's plan disrupted due to child's distress at separation":
        "Parent plan disrupted by separation distress",
    "Complaints of physical symptoms when attendance at school/daycare is anticipated or occurs":
        "Physical symptoms at school/daycare attendance",
    "Fear/anxiety about daycare/school attendance screen positive":
        "Fear/anxiety about school attendance",
    "Fear/Anxiety about leaving home for daycare/school":
        "Fear of leaving home for school",
    "Anticipatory fear of daycare/school":
        "Anticipatory fear of daycare/school",
    "Daycare/school non-attendance due to anxiety":
        "School non-attendance due to anxiety",
    "Has to be taken to daycare/school":
        "Must be taken to school",
    "Has to be taken to daycare/school because of separation anxiety":
        "Must be taken to school (separation anxiety)",
    "Picked up early from daycare/school due to anxiety":
        "Picked up early from school",
    "Child tries unsuccessfully to leave daycare/school due to anxiety":
        "Tries to leave school (unsuccessful)",
    "Child leaves daycare/school due to anxiety":
        "Leaves school due to anxiety",
    "Frequency of reluctance to go to sleep":
        "Reluctance to go to sleep",
    "Frequency of sleeping with family member due to a reluctance to sleep alone":
        "Sleeps with family member",
    "Sleep resistence":
        "Sleep resistance",
    "Hours taken to fall asleep":
        "Hours to fall asleep",
    "Freqency of nights child wakes up during the night":
        "Night waking frequency",
    "How long awak per night":
        "Duration awake per night",
    "Rising at night to check on family members":
        "Rises at night to check family",
    "Increased need for sleep":
        "Increased need for sleep",
    "Restless sleep":
        "Restless sleep",
    "Inadequately rested by sleep":
        "Inadequately rested by sleep",
    "Falls asleep in carseat for unscheduled nap":
        "Falls asleep in car (unscheduled)",
    "Tiredness":
        "Tiredness",
    "Child becomes tired or \"worn out\" more easily than normal":
        "Easily worn out",
    "Separation dreams":
        "Separation dreams",
    "Nervous tension":
        "Nervous tension",
    "Anxious affect that occurs in certain situations/environments":
        "Situational anxious affect",
    "Anxiety not associated with any particular situation":
        "Non-situational anxiety",
    "Exaggerated tartle response":
        "Exaggerated startle response",
    "Concentration difficulties":
        "Concentration difficulties",
    "Easy fatigability":
        "Easy fatigability",
    "Muscle Tension":
        "Muscle tension",
    "Restlessness":
        "Restlessness",
    "Worries that cannot be stopped voluntarily and occur across more than one acctivity":
        "Uncontrollable cross-domain worries",
    "Frequency of worries":
        "Frequency of worries",
    "Hypochondriasis":
        "Hypochondriasis",
    "Worry that family members will become ill":
        "Worry about family illness",
    "Worry about the future":
        "Worry about the future",
    "Worries about natural calamity":
        "Worry about natural disaster",
    "Worries about past behavior":
        "Worry about past behaviour",
    "Worries about competence or performance":
        "Worry about competence/performance",
    "Worries about appearance":
        "Worry about appearance",
    "Worries about money/food":
        "Worry about money/food",
    "Other worries":
        "Other worries",
}


def to_en(name: str, max_len: int = 48) -> str:
    """Return the English short label for a PAPA feature name."""
    label = FEATURE_LABELS_EN.get(name, name)
    return label[:max_len]
