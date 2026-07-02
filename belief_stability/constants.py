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
    "SIBLING",
    "RELIGION",
    "ETHNIC_GROUP",

    # Education
    "EDUCATED_AT",
    "DOCTORAL_ADVISOR",
    "STUDENT_OF",

    # Employment
    "EMPLOYER",
    "CEO_OF",
    "WORK_LOCATION",
    "FIELD_OF_WORK",

    # Organizations
    "FOUNDED",
    "FOUNDED_BY",
    "MEMBER_OF",
    "SUBSIDIARY_OF",
    "PARENT_ORGANIZATION",
    "OWNED_BY",
    "HEADQUARTERED_IN",
    "INDUSTRY",

    # Awards / recognition
    "AWARDED",
    "NOMINATED_FOR",

    # Location
    "LOCATED_IN",
    "COUNTRY",
    "CAPITAL_OF",

    # Sports
    "SPORT",
    "SPORTS_TEAM_MEMBER",
    "LEAGUE",
    "POSITION_PLAYED",

    # Politics / office
    "POLITICAL_PARTY",
    "POSITION_HELD",
    "POLITICAL_IDEOLOGY",
    "CANDIDATE_IN_ELECTION",

    # Creative works
    "CAST_MEMBER",
    "DIRECTOR",
    "PRODUCER",
    "COMPOSER",
    "AUTHOR",
    "SCREENWRITER",
    "GENRE",
    "NOTABLE_WORK",
    "RECORD_LABEL",
    "INSTRUMENT",
    "CREATOR",

    # Events / time
    "PARTICIPANT_IN",
    "INCEPTION",
    "PUBLICATION_DATE",

    # Military
    "MILITARY_BRANCH",
    "MILITARY_RANK",

    # Miscellaneous relational facts
    "PART_OF",
    "HAS_PART",
    "CONTAINS_LOCATION",
    "FOLLOWS",
    "FOLLOWED_BY",
    "INSTANCE_OF",
    "INFLUENCED_BY",
    "PERFORMER",
    "LIBRETTIST",
    "PRODUCTION_COMPANY",
    "WINNER",
    "CONFLICT",
    "PRESENT_IN_WORK",
    "LOCATION_OF_FORMATION",

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

    # -----------------------------------------------------------------
    # Expansion (2): curated from empirical relation-frequency data
    # measured over the existing 238-document belief cache (see
    # experiments/relation_frequency_report.py). Only true synonyms are
    # merged into an existing canonical name (same subject/object
    # direction, different surface wording) - relations that are
    # actually directional inverses of an existing one (e.g.
    # "founded_by" vs "founded") are kept as their OWN canonical name
    # and instead cross-referenced via INVERSE_RELATION_PAIRS /
    # SYMMETRIC_RELATIONS below, never silently renamed.
    # -----------------------------------------------------------------

    # Sports
    "sport": "SPORT",
    "member_of_sports_team": "SPORTS_TEAM_MEMBER",
    "league": "LEAGUE",
    "position_played_on_team_/_speciality": "POSITION_PLAYED",

    # Creative works / media
    "cast_member": "CAST_MEMBER",
    "genre": "GENRE",
    "performer": "PERFORMER",
    "director": "DIRECTOR",
    "producer": "PRODUCER",
    "composer": "COMPOSER",
    "author": "AUTHOR",
    "screenwriter": "SCREENWRITER",
    "librettist": "LIBRETTIST",
    "notable_work": "NOTABLE_WORK",
    "record_label": "RECORD_LABEL",
    "instrument": "INSTRUMENT",
    "creator": "CREATOR",
    "production_company": "PRODUCTION_COMPANY",
    "present_in_work": "PRESENT_IN_WORK",
    "publication_date": "PUBLICATION_DATE",

    # Politics / office
    "member_of_political_party": "POLITICAL_PARTY",
    "position_held": "POSITION_HELD",
    "officeholder": "POSITION_HELD",
    "chairperson": "POSITION_HELD",
    "political_ideology": "POLITICAL_IDEOLOGY",
    "candidacy_in_election": "CANDIDATE_IN_ELECTION",
    "office_contested": "CANDIDATE_IN_ELECTION",
    "candidate": "CANDIDATE_IN_ELECTION",

    # Family (gender-generalized; safe because REBEL emits each raw
    # string consistently, so matching stays internally consistent
    # regardless of the true real-world subject/object direction)
    "father": "PARENT",
    "mother": "PARENT",
    "sibling": "SIBLING",

    # Personal / demographic
    "religion": "RELIGION",
    "ethnic_group": "ETHNIC_GROUP",
    "country_of_citizenship": "NATIONALITY",

    # Education
    "student_of": "STUDENT_OF",
    "doctoral_advisor": "DOCTORAL_ADVISOR",
    "field_of_this_occupation": "FIELD_OF_WORK",
    "field_of_work": "FIELD_OF_WORK",
    "work_location": "WORK_LOCATION",

    # Organizations
    "founded_by": "FOUNDED_BY",
    "headquarters_location": "HEADQUARTERED_IN",
    "parent_organization": "PARENT_ORGANIZATION",
    "subsidiary": "SUBSIDIARY_OF",
    "owned_by": "OWNED_BY",
    "industry": "INDUSTRY",

    # Awards / recognition
    "award_received": "AWARDED",
    "nominated_for": "NOMINATED_FOR",
    "winner": "WINNER",

    # Location / geography
    "country": "COUNTRY",
    "country_of_origin": "COUNTRY",
    "location": "LOCATED_IN",
    "located_in_the_administrative_territorial_entity": "LOCATED_IN",
    "contains_administrative_territorial_entity": "CONTAINS_LOCATION",
    "location_of_formation": "LOCATION_OF_FORMATION",

    # Events / relations
    "participant_in": "PARTICIPANT_IN",
    "participant": "PARTICIPANT_IN",
    "conflict": "CONFLICT",
    "inception": "INCEPTION",
    "influenced_by": "INFLUENCED_BY",
    "has_part": "HAS_PART",
    "part_of": "PART_OF",
    "follows": "FOLLOWS",
    "followed_by": "FOLLOWED_BY",
    "instance_of": "INSTANCE_OF",

    # Military
    "military_branch": "MILITARY_BRANCH",
    "military_rank": "MILITARY_RANK",
}


# =========================================================
# Inverse / Symmetric Relation Handling
# =========================================================
#
# INVERSE_RELATION_PAIRS: relation R's inverse is a DIFFERENT
# canonical name. If (A, R, B) holds, a candidate phrased as
# (B, INVERSE_RELATION_PAIRS[R], A) asserts the exact same
# fact. Kept intentionally small and strictly logically
# entailed - curated only from relations already present in
# RELATION_MAPPING/CANONICAL_RELATIONS above, never guessed.
#
# SYMMETRIC_RELATIONS: relation R's inverse is the SAME
# canonical name with subject/object swapped, e.g. if
# (A, SPOUSE, B) holds then (B, SPOUSE, A) asserts the same
# fact.

INVERSE_RELATION_PAIRS = {

    "FOUNDED": "FOUNDED_BY",
    "FOUNDED_BY": "FOUNDED",

    "PARENT": "CHILD",
    "CHILD": "PARENT",

    "HAS_PART": "PART_OF",
    "PART_OF": "HAS_PART",

    "LOCATED_IN": "CONTAINS_LOCATION",
    "CONTAINS_LOCATION": "LOCATED_IN",

    "FOLLOWS": "FOLLOWED_BY",
    "FOLLOWED_BY": "FOLLOWS",

    "DOCTORAL_ADVISOR": "STUDENT_OF",
    "STUDENT_OF": "DOCTORAL_ADVISOR",

}

SYMMETRIC_RELATIONS = {

    "SPOUSE",
    "SIBLING",

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


# =========================================================
# Document-Scoped Entity Normalization
# =========================================================
#
# Used only by DocumentEntityNormalizer (canonicalizer/
# document_entity_normalizer.py), which operates on one
# document's full pool of subject/object strings at a time -
# unlike ENTITY_ALIASES above (global, always safe), these
# tables are applied with per-document ambiguity gating (see
# that module for the merge conditions).

PERSON_TITLES = {

    "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
    "sir", "lord", "lady", "dame",
    "prof", "prof.", "professor",
    "rev", "rev.", "reverend", "father",
    "captain", "capt", "capt.",
    "colonel", "col", "col.",
    "general", "gen", "gen.",
    "major", "maj", "maj.",
    "sergeant", "sgt", "sgt.",
    "president", "senator", "governor", "judge",

}

# Small, curated nickname -> full-first-name table. Only used
# to merge a nickname mention into the document's already-
# identified primary subject when the implied first name
# matches - never a global rename.
NICKNAME_TO_FIRST_NAME = {

    "bill": "william",
    "billy": "william",
    "will": "william",
    "bob": "robert",
    "bobby": "robert",
    "rob": "robert",
    "robbie": "robert",
    "jim": "james",
    "jimmy": "james",
    "dick": "richard",
    "rich": "richard",
    "rick": "richard",
    "ricky": "richard",
    "tom": "thomas",
    "tommy": "thomas",
    "ted": "edward",
    "eddie": "edward",
    "ed": "edward",
    "chuck": "charles",
    "charlie": "charles",
    "joe": "joseph",
    "joey": "joseph",
    "mike": "michael",
    "mickey": "michael",
    "dave": "david",
    "davy": "david",
    "steve": "steven",
    "stevie": "steven",
    "sam": "samuel",
    "sammy": "samuel",
    "alex": "alexander",
    "andy": "andrew",
    "drew": "andrew",
    "tony": "anthony",
    "ken": "kenneth",
    "kenny": "kenneth",
    "nick": "nicholas",
    "pete": "peter",
    "pat": "patrick",
    "harry": "henry",
    "hank": "henry",
    "liz": "elizabeth",
    "beth": "elizabeth",
    "betty": "elizabeth",
    "eliza": "elizabeth",
    "kate": "katherine",
    "katie": "katherine",
    "kathy": "katherine",
    "maggie": "margaret",
    "peggy": "margaret",
    "meg": "margaret",
    "sue": "susan",
    "susie": "susan",
    "jen": "jennifer",
    "jenny": "jennifer",
    "jess": "jessica",
    "jessie": "jessica",
    "abby": "abigail",
    "annie": "ann",
    "cathy": "catherine",
    "debbie": "deborah",
    "vicky": "victoria",

}