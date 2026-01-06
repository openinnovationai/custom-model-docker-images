"""
Language mappings for ASR systems.
"""

import logging

from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

logger = logging.getLogger(__name__)

OMNILINGUAL_ASR_LANGUAGES: list[str] = set(supported_langs)

# Whisper supported languages (2-letter code, full name)
# Mapping from Whisper/OpenAI format to Omnilingual-ASR format
# See: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
# See: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py
WHISPER_TO_OMNILINGUAL: dict[str, str] = {
    # English
    "en": "eng_Latn",
    "english": "eng_Latn",
    # Chinese (defaulting to Mandarin, many other variants available)
    "zh": "cmn_Hans",
    "chinese": "cmn_Hans",
    "mandarin": "cmn_Hans",
    # Spanish
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "castilian": "spa_Latn",
    # French
    "fr": "fra_Latn",
    "french": "fra_Latn",
    # German
    "de": "deu_Latn",
    "german": "deu_Latn",
    # Italian
    "it": "ita_Latn",
    "italian": "ita_Latn",
    # Portuguese
    "pt": "por_Latn",
    "portuguese": "por_Latn",
    # Russian
    "ru": "rus_Cyrl",
    "russian": "rus_Cyrl",
    # Japanese
    "ja": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    # Korean
    "ko": "kor_Hang",
    "korean": "kor_Hang",
    # Arabic
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    # Hindi
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    # Dutch
    "nl": "nld_Latn",
    "dutch": "nld_Latn",
    "flemish": "nld_Latn",
    # Polish
    "pl": "pol_Latn",
    "polish": "pol_Latn",
    # Turkish
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
    # Vietnamese
    "vi": "vie_Latn",
    "vietnamese": "vie_Latn",
    # Thai
    "th": "tha_Thai",
    "thai": "tha_Thai",
    # Indonesian
    "id": "ind_Latn",
    "indonesian": "ind_Latn",
    # Malay
    "ms": "zsm_Latn",
    "malay": "zsm_Latn",
    # Ukrainian
    "uk": "ukr_Cyrl",
    "ukrainian": "ukr_Cyrl",
    # Czech
    "cs": "ces_Latn",
    "czech": "ces_Latn",
    # Swedish
    "sv": "swe_Latn",
    "swedish": "swe_Latn",
    # Danish
    "da": "dan_Latn",
    "danish": "dan_Latn",
    # Finnish
    "fi": "fin_Latn",
    "finnish": "fin_Latn",
    # Norwegian
    "no": "nob_Latn",
    "norwegian": "nob_Latn",
    # Norwegian Nynorsk (not available in Omnilingual-ASR v0.2.0)
    # "nn": "nno_Latn",
    # "nynorsk": "nno_Latn",
    # Greek
    "el": "ell_Grek",
    "greek": "ell_Grek",
    # Hebrew
    "he": "heb_Hebr",
    "hebrew": "heb_Hebr",
    # Hungarian
    "hu": "hun_Latn",
    "hungarian": "hun_Latn",
    # Romanian
    "ro": "ron_Latn",
    "romanian": "ron_Latn",
    "moldavian": "ron_Latn",
    "moldovan": "ron_Latn",
    # Bulgarian
    "bg": "bul_Cyrl",
    "bulgarian": "bul_Cyrl",
    # Slovak
    "sk": "slk_Latn",
    "slovak": "slk_Latn",
    # Croatian
    "hr": "hrv_Latn",
    "croatian": "hrv_Latn",
    # Slovenian
    "sl": "slv_Latn",
    "slovenian": "slv_Latn",
    # Serbian
    "sr": "srp_Cyrl",
    "serbian": "srp_Cyrl",
    # Estonian (defaulting to ekk_Latn, other options: vro_Latn)
    "et": "ekk_Latn",
    "estonian": "ekk_Latn",
    # Latvian (defaulting to lav_Latn, other options: ltg_Latn)
    "lv": "lav_Latn",
    "latvian": "lav_Latn",
    # Lithuanian
    "lt": "lit_Latn",
    "lithuanian": "lit_Latn",
    # Catalan
    "ca": "cat_Latn",
    "catalan": "cat_Latn",
    "valencian": "cat_Latn",
    # Tamil
    "ta": "tam_Taml",
    "tamil": "tam_Taml",
    # Urdu
    "ur": "urd_Arab",
    "urdu": "urd_Arab",
    # Malayalam
    "ml": "mal_Mlym",
    "malayalam": "mal_Mlym",
    # Welsh
    "cy": "cym_Latn",
    "welsh": "cym_Latn",
    # Telugu
    "te": "tel_Telu",
    "telugu": "tel_Telu",
    # Persian
    "fa": "fas_Arab",
    "persian": "fas_Arab",
    # Bengali
    "bn": "ben_Beng",
    "bengali": "ben_Beng",
    # Azerbaijani (defaulting to aze_Latn, other options: aze_Arab, aze_Cyrl)
    "az": "aze_Latn",
    "azerbaijani": "aze_Latn",
    # Kannada
    "kn": "kan_Knda",
    "kannada": "kan_Knda",
    # Macedonian
    "mk": "mkd_Cyrl",
    "macedonian": "mkd_Cyrl",
    # Basque
    "eu": "eus_Latn",
    "basque": "eus_Latn",
    # Icelandic
    "is": "isl_Latn",
    "icelandic": "isl_Latn",
    # Armenian
    "hy": "hye_Armn",
    "armenian": "hye_Armn",
    # Nepali
    "ne": "nep_Deva",
    "nepali": "nep_Deva",
    # Mongolian
    "mn": "khk_Cyrl",
    "mongolian": "khk_Cyrl",
    # Bosnian
    "bs": "bos_Latn",
    "bosnian": "bos_Latn",
    # Kazakh
    "kk": "kaz_Cyrl",
    "kazakh": "kaz_Cyrl",
    # Albanian
    "sq": "als_Latn",
    "albanian": "als_Latn",
    # Swahili
    "sw": "swh_Latn",
    "swahili": "swh_Latn",
    # Galician
    "gl": "glg_Latn",
    "galician": "glg_Latn",
    # Marathi
    "mr": "mar_Deva",
    "marathi": "mar_Deva",
    # Punjabi
    "pa": "pan_Guru",
    "punjabi": "pan_Guru",
    "panjabi": "pan_Guru",
    # Sinhala
    "si": "sin_Sinh",
    "sinhala": "sin_Sinh",
    "sinhalese": "sin_Sinh",
    # Khmer
    "km": "khm_Khmr",
    "khmer": "khm_Khmr",
    # Yoruba
    "yo": "yor_Latn",
    "yoruba": "yor_Latn",
    # Somali
    "so": "som_Latn",
    "somali": "som_Latn",
    # Afrikaans
    "af": "afr_Latn",
    "afrikaans": "afr_Latn",
    # Georgian
    "ka": "kat_Geor",
    "georgian": "kat_Geor",
    # Belarusian
    "be": "bel_Cyrl",
    "belarusian": "bel_Cyrl",
    # Tajik
    "tg": "tgk_Cyrl",
    "tajik": "tgk_Cyrl",
    # Sindhi
    "sd": "snd_Arab",
    "sindhi": "snd_Arab",
    # Gujarati
    "gu": "guj_Gujr",
    "gujarati": "guj_Gujr",
    # Amharic
    "am": "amh_Ethi",
    "amharic": "amh_Ethi",
    # Lao
    "lo": "lao_Laoo",
    "lao": "lao_Laoo",
    # Uzbek
    "uz": "uzn_Latn",
    "uzbek": "uzn_Latn",
    # Pashto
    "ps": "pbt_Arab",
    "pashto": "pbt_Arab",
    "pushto": "pbt_Arab",
    # Maltese
    "mt": "mlt_Latn",
    "maltese": "mlt_Latn",
    # Myanmar/Burmese
    "my": "mya_Mymr",
    "myanmar": "mya_Mymr",
    "burmese": "mya_Mymr",
    # Tagalog
    "tl": "tgl_Latn",
    "tagalog": "tgl_Latn",
    # Malagasy
    "mg": "plt_Latn",
    "malagasy": "plt_Latn",
    # Assamese
    "as": "asm_Beng",
    "assamese": "asm_Beng",
    # Lingala
    "ln": "lin_Latn",
    "lingala": "lin_Latn",
    # Hausa
    "ha": "hau_Latn",
    "hausa": "hau_Latn",
    # Javanese
    "jw": "jav_Latn",
    "javanese": "jav_Latn",
    # Sundanese
    "su": "sun_Latn",
    "sundanese": "sun_Latn",
    # Cantonese
    "yue": "yue_Hant",
    "cantonese": "yue_Hant",
    # Latin
    "la": "lat_Latn",
    "latin": "lat_Latn",
    # Maori
    "mi": "mri_Latn",
    "maori": "mri_Latn",
    # Breton
    "br": "bre_Latn",
    "breton": "bre_Latn",
    # Shona
    "sn": "sna_Latn",
    "shona": "sna_Latn",
    # Occitan
    "oc": "oci_Latn",
    "occitan": "oci_Latn",
    # Yiddish (only Eastern Yiddish present in Omnilingual-ASR v0.2.0)
    "yi": "ydd_Hebr",
    "yiddish": "ydd_Hebr",
    # Faroese
    "fo": "fao_Latn",
    "faroese": "fao_Latn",
    # Haitian Creole
    "ht": "hat_Latn",
    "haitian creole": "hat_Latn",
    # Turkmen (defaulting to Latin, other options: tuk_Arab)
    "tk": "tuk_Latn",
    "turkmen": "tuk_Latn",
    # Sanskrit (not available in Omnilingual-ASR v0.2.0)
    # "sa": "san_Deva",
    # "sanskrit": "san_Deva",
    # Luxembourgish
    "lb": "ltg_Latn",
    "luxembourgish": "ltg_Latn",
    # Tibetan
    "bo": "bod_Tibt",
    "tibetan": "bod_Tibt",
    # Tatar
    "tt": "tat_Cyrl",
    "tatar": "tat_Cyrl",
    # Hawaiian
    "haw": "haw_Latn",
    "hawaiian": "haw_Latn",
    # Bashkir
    "ba": "bak_Cyrl",
    "bashkir": "bak_Cyrl",
}


def map_whisper_to_omnilingual(language: str) -> str:
    """
    Map Whisper/OpenAI language codes to Omnilingual-ASR format.

    Args:
        language: Language code (e.g., "en", "english") or Omnilingual format (e.g., "eng_Latn")

    Returns:
        Omnilingual-ASR language code
    """
    lang_lower = language.lower()
    if lang_lower in WHISPER_TO_OMNILINGUAL:
        return WHISPER_TO_OMNILINGUAL[lang_lower]

    if language in OMNILINGUAL_ASR_LANGUAGES:
        return language

    logger.warning(f"Unknown language: {language}. Defaulting to 'eng_Latn'.")
    return "eng_Latn"
