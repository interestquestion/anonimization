import itertools
import logging
import re

from pymystem3 import Mystem

from address import CITIES, REGIONS

# import spacy

# SPACY_MODEL_LG = spacy.load("ru_core_news_lg")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MYSTEM = Mystem()


# def get_all_names_spacy(text) -> list[tuple[int, int]]:
#     doc = SPACY_MODEL_LG(text)
#     names_positions = []
#     for token in doc:
#         print(
#             f"{token.text} {token.pos_} {token.dep_} {token.head.text} {token.ent_type_}"
#         )
#         if token.ent_type_ == "PER":
#             names_positions.append((token.idx, token.idx + len(token.text) - 1))
#     return names_positions


# def get_all_addresses_spacy(text) -> list[tuple[int, int]]:
#     doc = SPACY_MODEL_LG(text)
#     addresses_positions = []
#     for token in doc:
#         print(
#             f"{token.text} {token.pos_} {token.dep_} {token.head.text} {token.ent_type_}"
#         )
#         if token.ent_type_ == "LOC":
#             addresses_positions.append((token.idx, token.idx + len(token.text) - 1))
#     return addresses_positions


def get_all_addresses_natasha(text) -> list[tuple[int, int]]:
    from natasha import (
        Doc,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        NewsSyntaxParser,
        Segmenter,
    )

    natasha_segmenter = Segmenter()
    natasha_morph_vocab = MorphVocab()
    natasha_news_embedding = NewsEmbedding()
    natasha_morph_tagger = NewsMorphTagger(natasha_news_embedding)
    natasha_syntax_parser = NewsSyntaxParser(natasha_news_embedding)
    natasha_ner_tagger = NewsNERTagger(natasha_news_embedding)

    doc = Doc(text)
    doc.segment(natasha_segmenter)
    doc.tag_morph(natasha_morph_tagger)
    doc.parse_syntax(natasha_syntax_parser)
    doc.tag_ner(natasha_ner_tagger)

    logging.info(doc.spans)

    addresses_positions = []
    for span in doc.spans:
        if span.type == "LOC":
            addresses_positions.append((span.start, span.stop))
    return addresses_positions


def get_all_names_mystem(text) -> list[tuple[int, int]]:
    analyze = MYSTEM.analyze(text)
    names_positions = []
    last_index = 0
    previous_words = []
    previous_word_positions = []
    previous_word_tags = []
    # add positions if consecutive word tags are фам, имя, отч or фам, имя or имя отч фам or имя фам
    for word in analyze:
        try:
            analysis = word["analysis"][0]
        except (KeyError, IndexError):
            continue
        index = text.find(word["text"], last_index)
        if "фам" in analysis["gr"]:
            previous_word_tags.append("фам")
        elif "имя" in analysis["gr"]:
            previous_word_tags.append("имя")
        elif "отч" in analysis["gr"]:
            previous_word_tags.append("отч")
        # if word is one letter, save the tag letter
        elif len(word["text"].strip(".")) == 1 and word["text"].strip(".").isalpha():
            previous_word_tags.append("letter")
        else:
            previous_word_tags.append("")
        previous_word_positions.append((index, index + len(word["text"]) - 2))
        previous_words.append(word["text"])
        if previous_word_tags[-3:] in [
            ["фам", "имя", "отч"],
            ["имя", "отч", "фам"],
            ["фам", "letter", "letter"],
            ["letter", "letter", "фам"],
        ]:
            detected_name = text[previous_word_positions[-3][0]:previous_word_positions[-1][1]+2]
            if re.search(r"[\d()]", detected_name):
                continue
            names_positions += previous_word_positions[-3:]
            logger.info(f"Name detected (3 words): {detected_name} (anonymized). Previous word tags: {previous_word_tags[-3:]}. Previous words: {previous_words[-3:]}")
        elif previous_word_tags[-2:] in [["фам", "имя"], ["имя", "фам"]]:
            detected_name = text[previous_word_positions[-2][0]:previous_word_positions[-1][1]+2]
            if re.search(r"[\d()]", detected_name):
                continue
            names_positions += previous_word_positions[-2:]
            logger.info(f"Name detected (2 words): {detected_name} (anonymized). Previous word tags: {previous_word_tags[-2:]}. Previous words: {previous_words[-2:]}")

        last_index = index + len(word["text"])
    return names_positions


def get_address_indices_algo(text):
    address_levels = [
        ["респ\.", "республика", "край", "область", "обл\.", "г\.ф\.з\.", "а\.окр\."],
        ["пос\.", "поселение", "р-н", "район", "с\/с", "сельсовет"],
        [
            "г\.",
            "город",
            "пгт\.",
            "рп\.",
            "кп\.",
            "гп\.",
            "п\.",
            "поселок",
            "аал",
            "арбан",
            "аул",
            "в-ки",
            "выселки",
            "г-к",
            "заимка",
            "з-ка",
            "починок",
            "п-к",
            "киш\.",
            "кишлак",
            "п\.ст\.",
            "ж\/д",
            "м-ко",
            "местечко",
            "деревня",
            "с\.",
            "село",
            "сл\.",
            "ст\.",
            "станция",
            "ст-ца",
            "станица",
            "у\.",
            "улус",
            "х\.",
            "хутор",
            "рзд\.",
            "разъезд",
            "зим\.",
            "зимовье",
            "д\.",
        ],
        [
            "ал\.",
            "аллея",
            "б-р",
            "бульвар",
            "взв\.",
            "взд\.",
            "въезд",
            "дор\.",
            "дорога",
            "ззд\.",
            "заезд",
            "километр",
            "к-цо",
            "кольцо",
            "лн\.",
            "линия",
            "мгстр\.",
            "магистраль",
            "наб\.",
            "набережная",
            "пер-д",
            "переезд",
            "пер\.",
            "переулок",
            "пл-ка",
            "площадка",
            "пл\.",
            "площадь",
            "пр-кт\.",
            "проспект",
            "проул\.",
            "проулок",
            "рзд\.",
            "разъезд",
            "ряд",
            "с-р",
            "сквер",
            "с-к",
            "спуск",
            "сзд\.",
            "съезд",
            "тракт",
            "туп\.",
            "тупик",
            "ул\.",
            "улица",
            "ш\.",
            "шоссе",
        ],
        [
            "влд\.",
            "владение",
            "г-ж",
            "гараж",
            "д\.",
            "дом",
            "двлд\.",
            "домовладение",
            "зд\.",
            "здание",
            "з\/у",
            "участок",
            "кв\.",
            "квартира",
            "ком\.",
            "комната",
            "подв\.",
            "подвал",
            "кот\.",
            "котельная",
            "п-б",
            "погреб",
            "к\.",
            "корпус",
            "офис",
            "пав\.",
            "павильон",
            "помещ\.",
            "помещение",
            "раб\.уч\.",
            "скл\.",
            "склад",
            "соор\.",
            "сооружение",
            "стр\.",
            "строение",
            "торг\.зал\.",
            "цех",
        ],
    ]

    address_regex = re.compile(
        r"(" + "|".join([r"|".join(level) for level in address_levels]) + r")",
        re.IGNORECASE,
    )

    matches = []
    for match in address_regex.finditer(text):
        matches.append((match.start(), match.end()))

    return matches


def find_addresses_algo(text):
    indices = get_address_indices_algo(text)
    current_address = ""
    start_index = -1
    end_index = -1
    addresses = []

    for start, end in indices:
        if start_index == -1:
            start_index = start

        part = text[start:end]
        current_address += part + " "
        end_index = end

        if re.search(r"[,.!?;]", text[end:]):
            addresses.append((current_address.strip(), start_index, end_index))
            current_address = ""
            start_index = -1
            end_index = -1

    if current_address:
        addresses.append((current_address.strip(), start_index, end_index))

    return [(start, end) for _, start, end in addresses]


def get_phone_numbers_positions(text) -> list[tuple[int, int]]:
    matches = re.finditer(
        r"(\+?[78][\s-]{,3})?(\(\d{1,4}\)[\s-]{,3})?" r"\d[\d\s-]{5,14}\d", text
    )
    positions = []
    for match in matches:
        match_text = re.sub(r"[\(\)\s+-]", "", match.group())
        if len(match_text) < 7:
            continue
        start = match.start()
        end = match.end() - 1
        positions.append((start, end))
        logger.info(f"Phone number detected: {match.group()} (anonymized)")
    return positions


def get_bd_positions(text, date_year_max: int = 2016) -> list[tuple[int, int]]:
    bd_positions = []
    text = text.lower()
    regex = r"(?:рожде[ни]{2}.|\bд.\s*р.)[\s:-]*([\d.,\s-]{4,14}\d)"
    regex_dates = r"(\d{2}[.,/-]\d{2}[.,/-]\s*\d{2,4})"
    matches = itertools.chain(re.finditer(regex, text), re.finditer(regex_dates, text))
    for match in matches:
        # if date is recent, it is not a birth date
        try:
            try:
                day, month, year = re.split(r"[./-]\s*", match.group(1))
                year = year.split()[0]
            except Exception:
                match_text = re.sub(r"[^\d]", "", match.group(1))
                if len(match_text) == 8:
                    day, month, year = match_text[:2], match_text[2:4], match_text[4:]
            year = int(year)
            if year > date_year_max or year / 100 < 1 and year > (date_year_max % 100):
                continue
            bd_positions.append((match.start(1), match.end(1) - 1))
            logger.info(f"Birth date detected: {match.group(1)} (anonymized)")
        except Exception as e:
            continue
    return bd_positions


def extract_complex_address_indices(text):
    regions = [region.lower() for region in CITIES + REGIONS]

    # Define the abbreviations and full words in a dictionary
    terms_word = {
        "респ": "ублика",
        "кр": "ай",
        "автономная обл": "асть",
        "автономный округ": "",
        "обл": "асть",
        "г": "ород",
        "пос": "елок",
        "ул": "ица",
        "ш": "оссе",
        "просп": "ект",
        "пл": "ощадь",
        "пер": "еулок",
        "б": "ульвар",
        "мкр": "орайон",
        "село": "",
        "ст": "анция",
        "пр": "омзона",
        "производственная зона": "",
    }
    terms_num = {
        "д": "ом",
        "кв": "артира",
        "корп": "ус",
        "оф": "ис",
        "пом": "ещение",
    }

    # Generate the regular expression pattern
    patterns_num = [
        rf"\b({abbrev}({full_word})?[\s.,])[\s\d-]+"
        for abbrev, full_word in terms_num.items()
    ]
    patterns_words = rf'(?P<big_city>{"|".join(regions)})'
    patterns_words += "|".join(
        [
            rf"\b({abbrev}({full_word})?[\s.,])\s*[а-яё-]+"
            for abbrev, full_word in terms_word.items()
        ]
    )
    patterns = [patterns_words] + patterns_num
    pattern = "|".join(patterns)

    # Compile the regular expression
    address_regex = re.compile(pattern, re.IGNORECASE)

    matches = address_regex.finditer(text)

    address_indices = [(match.start(), match.end() - 1) for match in matches]
    # if there is just one match, return []
    if len(address_indices) == 1:
        return []
    # merge indices if they are close, remove lone indices if they form a group of size 1
    merged_indices = []
    cur_count = 1
    for i in range(1, len(address_indices)):
        if address_indices[i][0] - address_indices[i - 1][1] < 10:
            cur_count += 1
        else:
            if cur_count > 1:
                start = address_indices[i - cur_count][0]
                end = address_indices[i - 1][1]
                address_text = text[start:end+1]
                
                # Validate the address before accepting it
                if is_valid_address(address_text):
                    merged_indices.append((start, end))
                    logger.info(f"Address detected: {address_text} (anonymized)")
            cur_count = 1
    if cur_count > 1:
        start = address_indices[-cur_count][0]
        end = address_indices[-1][1]
        address_text = text[start:end+1]
        
        # Validate the address before accepting it
        if is_valid_address(address_text):
            merged_indices.append((start, end))
            logger.info(f"Address detected: {address_text} (anonymized)")
    return merged_indices

def is_valid_address(text):
    """
    Validate an address to avoid false positives.
    """
    # Minimum reasonable length for an address
    if len(text) < 10:
        return False
    
    # Must contain at least one digit (for house number, etc.)
    if not re.search(r'\d', text):
        return False
    
    # Check for common address components
    address_components = ['ул', 'д', 'дом', 'кв', 'г', 'пер', 'обл', 'пос', 'р-н', 'район']
    has_component = False
    for component in address_components:
        if re.search(rf'\b{component}[\s\.,]', text, re.IGNORECASE):
            has_component = True
            break
    
    # Check for sequences of uppercase letters that look like abbreviations
    if re.search(r'[А-Я]{3,}', text) and not has_component:
        return False
    
    # Avoid very short phrases with commas that are likely not addresses
    comma_parts = text.split(',')
    if len(comma_parts) >= 2 and any(len(part.strip()) <= 2 for part in comma_parts):
        # If parts are very short, require more address-like patterns
        return has_component and re.search(r'\d+', text)
    
    return has_component or re.search(r'\b\d+\s*[\.,]?\s*\w+', text)


def find_16_digit_numbers(text):
    number_regex = re.compile(r"\b(?:\d\s*){16}\b")

    matches = number_regex.finditer(text)

    number_indices = [(match.start(), match.end()) for match in matches]
    return number_indices


def find_numeric_sequences(text):
    number_regex = re.compile(r"\b\+?(?:\d[\s\-]*){8,}\b")

    matches = number_regex.finditer(text)

    number_indices = []
    for match in matches:
        start = match.start()
        end = match.end() - 3
        number_indices.append((start, end))
        logger.info(f"Numeric sequence detected: {text[start:end]} (anonymized)")
    return number_indices


def get_specific_numbers(text):
    text = text.lower()
    number = r"(?:\d[\s-]*){4,16}"
    words = r"(?:снилс|\bинн|паспорта?|паспортные данные|полиса?|\bомс|тел\.?(?:ефона?)?|№|карты|\b[ин]\s*/\s*б|номер)"
    regex = f"{words}[\s:-]*({number})"
    matches = re.finditer(regex, text)
    positions = []
    for match in matches:
        start = match.start(1)
        end = match.end(1) - 3
        positions.append((start, end))
        logger.info(f"Specific number detected: {text[start:end]} (type: {match.group().split()[0]}, anonymized)")
    return positions
