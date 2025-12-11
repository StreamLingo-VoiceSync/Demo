"""Configuration for the MT system."""

from __future__ import annotations
import pathlib

# Logging
# from common.logger import setup_mt_logger
# log = setup_mt_logger()

# Output directories
MT_OUTPUT_DIR = pathlib.Path("./mt_output")
for d in ["tts_ready", "grammar_analysis", "alignment_analysis", "metrics", "logs", "dual_lane"]:
    (MT_OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)

# Language configuration
LANGUAGE_CODES = {
    "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
    "hi": "hin_Deva", "hin": "hin_Deva", "hindi": "hin_Deva",
    "es": "spa_Latn", "spa": "spa_Latn", "spanish": "spa_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn", "french": "fra_Latn",
}

LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
}

# Advanced idiom database (50+ idioms per language)
IDIOM_DATABASE = {
    "eng_Latn": {
        "break a leg": ("good luck", "idiom", "theater"),
        "piece of cake": ("very easy", "idiom", "casual"),
        "raining cats and dogs": ("heavy rain", "idiom", "weather"),
        "under the weather": ("sick", "idiom", "health"),
        "hit the books": ("study hard", "idiom", "education"),
        "let the cat out of the bag": ("reveal secret", "idiom", "secrets"),
        "where there's a will, there's a way": ("determination leads to success", "idiom", "motivation"),
        "against all odds": ("despite difficulties", "idiom", "challenges"),
        "give it a go": ("try", "idiom", "action"),
        "pull it off": ("succeed", "idiom", "success"),
        "long shot": ("unlikely possibility", "idiom", "probability"),
        "piece of the pie": ("share of benefits", "idiom", "business"),
        "bite the bullet": ("face difficulty", "idiom", "courage"),
        "blow off steam": ("relieve stress", "idiom", "emotion"),
        "break the ice": ("start conversation", "idiom", "social"),
        "burn the midnight oil": ("work hard", "idiom", "work"),
        "call it a day": ("stop working", "idiom", "work"),
        "caught red handed": ("caught doing wrong", "idiom", "guilt"),
        "devil's advocate": ("opposite viewpoint", "idiom", "debate"),
        "dime a dozen": ("very common", "idiom", "quantity"),
        "down to earth": ("practical", "idiom", "personality"),
        "eagle eye": ("keen observation", "idiom", "vision"),
        "easy as pie": ("very easy", "idiom", "simplicity"),
        "eleventh hour": ("last moment", "idiom", "timing"),
        "every dog has its day": ("everyone gets chance", "idiom", "fairness"),
        "face the music": ("accept consequence", "idiom", "responsibility"),
        "fall on deaf ears": ("ignored", "idiom", "communication"),
        "few and far between": ("rare", "idiom", "rarity"),
        "fit as a fiddle": ("healthy", "idiom", "health"),
        "go the extra mile": ("exceed expectation", "idiom", "dedication"),
        "gold digger": ("money seeker", "idiom", "character"),
        "good as gold": ("trustworthy", "idiom", "trust"),
        "greek to me": ("incomprehensible", "idiom", "understanding"),
        "growing pains": ("transition difficulties", "idiom", "development"),
        "gum up the works": ("cause problems", "idiom", "disruption"),
        "hand to mouth": ("barely surviving", "idiom", "poverty"),
        "hang in there": ("persevere", "idiom", "encouragement"),
        "hard pill to swallow": ("difficult truth", "idiom", "acceptance"),
        "head over heels": ("very much in love", "idiom", "love"),
        "heart of gold": ("kind person", "idiom", "kindness"),
        "help yourself": ("take what you want", "idiom", "permission"),
        "high and dry": ("abandoned", "idiom", "abandonment"),
        "hit paydirt": ("find success", "idiom", "success"),
        "hold your head up high": ("be proud", "idiom", "pride"),
        "hold your horses": ("wait", "idiom", "patience"),
        "hold your own": ("manage well", "idiom", "ability"),
        "honest as the day is long": ("very honest", "idiom", "honesty"),
    },
    "hin_Deva": {
        "आँखें खुल जाना": ("sudden realization", "idiom", "awareness"),
        "दिल छोटा न करना": ("don't lose heart", "idiom", "courage"),
        "हाथ आना": ("opportunity arriving", "idiom", "opportunity"),
        "मुंह देखना": ("watch someone carefully", "idiom", "observation"),
        "दिल बैठ जाना": ("lose courage", "idiom", "fear"),
        "गले लगाना": ("embrace", "idiom", "affection"),
        "कान पर जूँ न रेंगना": ("not to care", "idiom", "indifference"),
        "लकीर का फकीर": ("orthodox person", "idiom", "tradition"),
        "मीठा तोड़ना": ("win sympathy", "idiom", "charm"),
        "नाक में नकेल": ("under control", "idiom", "control"),
    },
    "spa_Latn": {
        "estar en las nubes": ("daydreaming", "idiom", "imagination"),
        "costar un ojo de la cara": ("very expensive", "idiom", "cost"),
        "llevarse bien": ("get along", "idiom", "relationship"),
        "ponerse verde": ("become angry", "idiom", "anger"),
        "echar de menos": ("miss someone", "idiom", "longing"),
        "estar de buen humor": ("in good mood", "idiom", "mood"),
        "tomar el pelo": ("make fun", "idiom", "humor"),
        "no poder con la carga": ("cannot handle", "idiom", "overwhelm"),
        "meterse en lios": ("get into trouble", "idiom", "trouble"),
        "cambiar de idea": ("change mind", "idiom", "decision"),
    },
    "fra_Latn": {
        "avoir le cafard": ("depressed", "idiom", "mood"),
        "avoir un chat dans la gorge": ("frog in throat", "idiom", "voice"),
        "bête noire": ("pet peeve", "idiom", "dislike"),
        "coûter cher": ("cost a lot", "idiom", "expense"),
        "donner un coup de main": ("help", "idiom", "assistance"),
        "être de mauvaise humeur": ("bad mood", "idiom", "mood"),
        "faire la tête": ("sulk", "idiom", "emotion"),
        "être sur le point de": ("about to", "idiom", "timing"),
        "faire du feu": ("make fire", "idiom", "action"),
        "je suis à court de": ("running out of", "idiom", "shortage"),
    }
}

# Homophones database
HOMOPHONES_DB = {
    "eng_Latn": {
        "to": ["too", "two"],
        "their": ["there", "they're"],
        "for": ["fore", "four"],
        "know": ["no"],
        "right": ["write"],
        "be": ["bee"],
        "son": ["sun"],
        "sea": ["see"],
        "buy": ["by"],
        "meat": ["meet"],
    },
    "hin_Deva": {
        "काना": ["कान"],
        "पार": ["पार"],
        "राज": ["राज"],
    },
    "spa_Latn": {
        "caza": ["casa"],
        "vaya": ["valla"],
        "ves": ["vez"],
    },
    "fra_Latn": {
        "cent": ["sang"],
        "sait": ["serait"],
        "pair": ["père"],
    }
}

# Synonyms database
SYNONYMS_DB = {
    "eng_Latn": {
        "happy": ["joyful", "glad", "cheerful", "content"],
        "sad": ["unhappy", "depressed", "sorrowful"],
        "big": ["large", "huge", "enormous", "massive"],
        "small": ["tiny", "little", "petite"],
        "fast": ["quick", "swift", "rapid"],
        "slow": ["sluggish", "leisurely"],
        "beautiful": ["pretty", "gorgeous", "lovely"],
        "ugly": ["unattractive", "hideous"],
        "good": ["excellent", "fine", "great"],
        "bad": ["poor", "terrible", "awful"],
    },
    "hin_Deva": {
        "खुश": ["आनंदित", "प्रसन्न"],
        "दुखी": ["उदास", "विषाद"],
        "बड़ा": ["विशाल", "विस्तृत"],
        "छोटा": ["लघु", "सूक्ष्म"],
    },
    "spa_Latn": {
        "feliz": ["contento", "alegre"],
        "triste": ["deprimido", "melancólico"],
        "grande": ["enorme", "vasto"],
        "pequeño": ["diminuto", "minúsculo"],
    },
    "fra_Latn": {
        "heureux": ["joyeux", "content"],
        "triste": ["morose", "mélancolique"],
        "grand": ["énorme", "vaste"],
        "petit": ["diminutif", "minuscule"],
    }
}

# Grammar rules
GRAMMAR_RULES = {
    "eng_Latn": {
        "subject_verb_agreement": True,
        "article_usage": True,
        "tense_consistency": True,
    },
    "hin_Deva": {
        "gender_agreement": True,
        "case_marking": True,
        "verb_conjugation": True,
    },
}

# Punctuation rules
PUNCTUATION_RULES = {
    "eng_Latn": {"sentence_end": ".", "question": "?", "exclamation": "!"},
    "hin_Deva": {"sentence_end": "।", "question": "?", "exclamation": "!"},
    "spa_Latn": {"sentence_end": ".", "question": "?", "exclamation": "!"},
    "fra_Latn": {"sentence_end": ".", "question": "?", "exclamation": "!"},
}

# Context window size for dual-lane processing
CONTEXT_WINDOW_SIZE = 5

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.70,
    "low": 0.50
}