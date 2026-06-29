"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Stability

Description:
    Global constants used throughout the Belief Stability
    module.

---------------------------------------------------------
"""

# =========================================================
# Canonical Relation Vocabulary
# =========================================================

CANONICAL_RELATIONS = {

    # Birth
    "BORN_IN",
    "BORN_ON",

    # Death
    "DIED_IN",
    "DIED_ON",

    # Personal
    "NATIONALITY",
    "OCCUPATION",
    "SPOUSE",
    "PARENT",
    "CHILD",

    # Education
    "EDUCATED_AT",

    # Employment
    "EMPLOYER",
    "CEO_OF",

    # Organizations
    "FOUNDED",
    "MEMBER_OF",

    # Awards
    "AWARDED",

    # Location
    "HEADQUARTERED_IN",
    "LOCATED_IN",

}


# =========================================================
# Canonical Transition Labels
# =========================================================

SUPPORT = "support"
ABSENT = "absent"
CONTRADICT = "contradict"


# =========================================================
# Default Confidence
# =========================================================

DEFAULT_CONFIDENCE = 1.0


# =========================================================
# Unknown Values
# =========================================================

UNKNOWN = "UNKNOWN"


# =========================================================
# Attribute Keys
# =========================================================

YEAR = "year"

DATE = "date"

TIME = "time"

LOCATION = "location"

QUANTITY = "quantity"


# =========================================================
# Future Extension Placeholders
# =========================================================

SUPPORTED_EXTRACTORS = {

    "rebel",

    "glirel",

    "openie",

}
# =========================================================
# Relation Mapping
# =========================================================

RELATION_MAPPING = {

    # Birth
    "born_in": "BORN_IN",
    "birthplace": "BORN_IN",
    "place_of_birth": "BORN_IN",

    "born_on": "BORN_ON",
    "date_of_birth": "BORN_ON",

    # Death
    "died_in": "DIED_IN",
    "place_of_death": "DIED_IN",

    "died_on": "DIED_ON",
    "date_of_death": "DIED_ON",

    # Organization
    "founder": "FOUNDED",
    "founder_of": "FOUNDED",
    "founded": "FOUNDED",
    "established": "FOUNDED",

    # Employment
    "ceo": "CEO_OF",
    "chief_executive_officer": "CEO_OF",
    "ceo_of": "CEO_OF",

    "employer": "EMPLOYER",

    # Personal
    "nationality": "NATIONALITY",
    "occupation": "OCCUPATION",
    "spouse": "SPOUSE",
    "parent": "PARENT",
    "child": "CHILD",

    # Education
    "educated_at": "EDUCATED_AT",

    # Awards
    "award": "AWARDED",
    "awarded": "AWARDED",

    # Location
    "located_in": "LOCATED_IN",
    "headquartered_in": "HEADQUARTERED_IN",

    # Membership
    "member_of": "MEMBER_OF",
}


# =========================================================
# Entity Alias Mapping
# =========================================================

ENTITY_ALIASES = {

    "usa": "United States",
    "u.s.": "United States",
    "us": "United States",

    "uk": "United Kingdom",
    "u.k.": "United Kingdom",

}