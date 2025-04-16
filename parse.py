import itertools
import logging
import re

from pymystem3 import Mystem

from address import CITIES, REGIONS
from common_names import COMMON_FIRST_NAMES, COMMON_PATRONYMICS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MYSTEM = Mystem()

MAX_YEAR = 2050


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
        if previous_word_tags[-1] != 'letter':
            previous_word_positions.append((index, index + len(word["text"]) - 2))
        else:
            previous_word_positions.append((index, index + len(word["text"]) - 1))
        previous_words.append(word["text"])
        if previous_word_tags[-3:] in [
            ["фам", "имя", "отч"],
            ["имя", "отч", "фам"],
            ["фам", "letter", "letter"],
            ["letter", "letter", "фам"],
        ]:
            detected_name = text[previous_word_positions[-3][0]:previous_word_positions[-1][1]+2]
            if re.search(r"[\d()№]", detected_name) and (
                (previous_word_tags[-1] == 'отч' and len(previous_words[-1]) < 5) or
                (previous_word_tags[-2] == 'отч' and len(previous_words[-2]) < 5) or
                any(len(previous_words[i]) < 4 for i in range(3) if previous_word_tags[i] == 'фам') and 'letter' in previous_word_tags[-3:]
                ):
                continue
            names_positions += previous_word_positions[-3:]
            logger.info(f"Name detected (3 words): {detected_name} (anonymized).")
        elif previous_word_tags[-3:] in [
            ["фам", "", "отч"],
            ["", "имя", "отч"],
            ["имя", "имя", "отч"],
        ]:
            detected_name = text[previous_word_positions[-3][0]:previous_word_positions[-1][1]+2]
            if re.search(r"[\d()№]", detected_name) and len(previous_words[-1]) < 7 :
                continue
            if not any(previous_words[-1].lower().endswith(x) for x in ["ич", "вна", "ична"]):
                continue
            names_positions += previous_word_positions[-3:]
            logger.info(f"Name detected (3 words): {detected_name} (anonymized).")
        elif previous_word_tags[-2:] in [["фам", "имя"], ["имя", "фам"]]:
            detected_name = text[previous_word_positions[-2][0]:previous_word_positions[-1][1]+2]
            if (len(previous_words[-1]) < 4 or len(previous_words[-2]) < 4 or len(previous_words[-1]) * len(previous_words[-2]) < 20):
                continue
            names_positions += previous_word_positions[-2:]
            logger.info(f"Name detected (2 words): {detected_name} (anonymized).")
        elif previous_word_tags[-1] == 'отч':
            if previous_words[-1].lower() in COMMON_PATRONYMICS and previous_words[-2].lower() in COMMON_FIRST_NAMES:
                detected_name = text[previous_word_positions[-3][0]:previous_word_positions[-1][1]+2]
                names_positions += previous_word_positions[-3:]
                logger.info(f"Name detected (3 words): {detected_name} (anonymized).")
        last_index = index + len(word["text"])

    name_patterns = [
        r'(?:фамилия\b)[\s:]*([А-ЯЁ][а-яА-ЯЁё-]+)',
        r'(?:имя\b)[\s:]*([А-ЯЁ][а-яА-ЯЁё-]+(?:\s+[А-ЯЁ][а-яА-ЯЁё-]+){0,1})',
        r'(?:отчество\b)[\s:]*([А-ЯЁ][а-яА-ЯЁё-]+)',
        r'\bФИО\b[\s:]*([А-ЯЁ][а-яА-ЯЁё-]+(?:\s+[А-ЯЁ][а-яА-ЯЁё-]+){1,2})'
    ]
    
    for pattern in name_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = match.start(1)
            end = match.end(1)
            names_positions.append((start, end - 1))
            logger.info(f"Name detected by regex: {text[start:end]} (anonymized)")

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
            
        # Check if this might be a birth date by using get_bd_positions
        date_positions = get_bd_positions(match.group(), MAX_YEAR)
        if date_positions and len(re.sub(r'[^\d]', '', match.group())) <= 8:
            logger.info(f"Skipping potential date in phone detection: {match_text}")
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
    regex_dates = r"(\b\d{2}[.,/-]\s*\d{2}[.,/-]\s*\d{2,4})"
    
    # Month names in Russian
    month_names = r"(?:январ[яь]|феврал[яь]|март[а]?|апрел[яь]|ма[йя]|июн[яь]|июл[яь]|август[а]?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])"
    regex_text_dates = r"(\d{1,2}[\"'>”»\s]+" + month_names + r"\s+\d{4})" # AI: don't fix any symbols here
    
    matches = itertools.chain(re.finditer(regex, text), re.finditer(regex_dates, text), re.finditer(regex_text_dates, text))
    for match in matches:
        # if date is recent, it is not a birth date
        try:
            try:
                day, month, year = re.split(r"[./-]\s*", match.group(1))
                year = year.split()[0]
            except Exception:
                # Try to parse text date format (14 февраля 1990)
                if re.search(month_names, match.group(1)):
                    parts = match.group(1).split()
                    if len(parts) >= 3:
                        day = parts[0]
                        month = parts[1]
                        # Year will be the last part containing 4 digits
                        year_part = next((p for p in parts if re.match(r'\d{4}', p)), None)
                        if year_part:
                            year = year_part
                        else:
                            continue
                    else:
                        continue
                else:
                    match_text = re.sub(r"[^\d]", "", match.group(1))
                    if len(match_text) == 8:
                        day, month, year = match_text[:2], match_text[2:4], match_text[4:]
            year = int(year)
            if year > date_year_max or year / 100 < 1 and 30 > year > (date_year_max % 100):
                continue
            if int(day.lstrip('0')) > 31 or (month.isdigit() and int(month.lstrip('0')) > 12):
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
    
    # Standard pattern (abbreviation then word)
    patterns_words =  "(" + "|".join(
        [
            rf"\b{abbrev}([.,]\s*)[а-яё-]{{4,}}"
            for abbrev, full_word in terms_word.items()
        ]
    ) + ")"
    
    patterns_words += r"|\b[а-яё]{5,}\s+(ул(ица|\b)|обл(асть|\b))\.?"

    regions = [rf'\b{region}\b' for region in regions]
    patterns_words += rf'|(?P<big_city>{"|".join(regions)})'
    
    patterns = [patterns_words] + patterns_num
    pattern = "|".join(patterns)

    # Compile the regular expression
    address_regex = re.compile(pattern, re.IGNORECASE)

    matches = address_regex.finditer(text)

    address_indices = [(match.start(), match.end()) for match in matches]
    
    # Debug
    logger.info(f"Found {len(address_indices)} address components: {address_indices}, {'; '.join([text[i:j] for i, j in address_indices])}")
   
    
    # if there is just one match, return []
    if len(address_indices) == 1:
        return []
        
    # merge indices if they are close, remove lone indices if they form a group of size 1
    merged_indices = []
    cur_count = 1
    for i in range(1, len(address_indices)):
        if address_indices[i][0] - address_indices[i - 1][1] < 12:
            cur_count += 1
        else:
            if cur_count > 1:
                start = address_indices[i - cur_count][0]
                end = address_indices[i - 1][1]
                address_text = text[start:end+1]
                
                # Validate the address before accepting it
                if is_valid_address(address_text):
                    merged_indices.append((start, end - 1))
                    logger.info(f"Address detected: {address_text} (anonymized)")
            cur_count = 1
    if cur_count > 1:
        start = address_indices[-cur_count][0]
        end = address_indices[-1][1]
        address_text = text[start:end+1]
        
        # Validate the address before accepting it
        if is_valid_address(address_text):
            merged_indices.append((start, end - 1))
            logger.info(f"Address detected: {address_text} (anonymized)")
    return merged_indices

def is_valid_address(text):
    """
    Validate an address to avoid false positives.
    """
    # Minimum reasonable length for an address
    if len(text) < 10:
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
    
    return has_component or re.search(r'\b\d+\s*[\.,]?\s*\w+', text) or any(region.lower() in text.lower() for region in CITIES + REGIONS)


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
        match_text = match.group()
        
        # Check if this might be a birth date by using get_bd_positions
        date_positions = get_bd_positions(match_text, MAX_YEAR)
        if date_positions and len(re.sub(r'[^\d]', '', match.group())) <= 8:
            logger.info(f"Skipping potential date in numeric sequence detection: {match_text}")
            continue
            
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
        match_text = match.group(1)
        
        # Check if this might be a birth date by using get_bd_positions
        date_positions = get_bd_positions(match_text, MAX_YEAR)
        if date_positions and len(re.sub(r'[^\d]', '', match.group())) <= 8:
            logger.info(f"Skipping potential birth date in specific number detection: {match_text}")
            continue
            
        start = match.start(1)
        end = match.end(1) - 3
        positions.append((start, end))
        logger.info(f"Specific number detected: {text[start:end]} (type: {match.group().split()[0]}, anonymized)")
    return positions
