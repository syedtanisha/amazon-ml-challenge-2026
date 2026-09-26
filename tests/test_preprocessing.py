import pytest

from src.preprocessing.normalize import (
    normalize_text,
    normalize_business_name,
    normalize_legal_suffixes,
    normalize_business_name_full,
    normalize_address,
    normalize_country,
    normalize_tokens,
    normalize_business_record,
    is_missing,
    normalize_optional_text,
)


def test_normalize_text_basic():
    assert normalize_text("  Tata & Sons, Inc.  ") == "tata and sons inc"


def test_normalize_text_unicode():
    assert normalize_text("Café Élite") == "cafe elite"


def test_normalize_text_empty():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_business_name_normalization():
    assert normalize_business_name("  ABC Company  ") == "abc company"


def test_legal_suffix_normalization():
    assert normalize_legal_suffixes("ABC Pvt. Ltd.") == "abc private limited"
    assert normalize_legal_suffixes("Tata Sons Inc.") == "tata sons incorporated"
    assert normalize_legal_suffixes("Example Corp.") == "example corporation"
    assert normalize_legal_suffixes("My Business Co.") == "my business company"


def test_dotted_legal_suffixes():
    assert normalize_legal_suffixes("Demo L.L.C.") == "demo llc"
    assert normalize_legal_suffixes("Demo P.L.C.") == "demo plc"
    assert normalize_legal_suffixes("Demo L.L.P.") == "demo llp"
    assert normalize_legal_suffixes("Demo P.C.") == "demo pc"
    assert normalize_legal_suffixes("Demo PC") == "demo pc"


def test_address_normalization():
    assert normalize_address("123 Main St.") == "123 main street"
    assert normalize_address("45 Park Rd.") == "45 park road"
    assert (
        normalize_address("10 MG Road, Apt. 4B")
        == "10 mg road apartment 4b"
    )


def test_address_abbreviations():
    assert (
        normalize_address("500 Oak Blvd, Ste 20")
        == "500 oak boulevard suite 20"
    )
    assert (
        normalize_address("Building 5, Room 12")
        == "building 5 room 12"
    )


def test_country_normalization():
    assert normalize_country(" US ") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country(None) == ""


def test_token_normalization():
    assert normalize_tokens("Tata & Sons") == ["tata", "and", "sons"]
    assert normalize_tokens(None) == []


def test_missing_values():
    assert is_missing(None) is True
    assert is_missing("") is True
    assert is_missing("   ") is True
    assert is_missing("ABC") is False


def test_optional_text():
    assert normalize_optional_text(None) == ""
    assert normalize_optional_text("   ") == ""
    assert normalize_optional_text("  Hello World  ") == "hello world"


def test_complete_business_record():
    result = normalize_business_record(
        "ABC Pvt. Ltd.",
        "123 Main St.",
        "US",
    )

    assert result == {
        "business_name": "abc private limited",
        "business_address": "123 main street",
        "country": "us",
    }


def test_missing_business_record():
    result = normalize_business_record(None, None, None)

    assert result == {
        "business_name": "",
        "business_address": "",
        "country": "",
    }
def test_real_world_address_identifiers():
    assert normalize_address("H No. 16-2-751/A/70") == "h number 16 2 751 a 70"
    assert normalize_address("S No. 42/2/3, H No. B/5") == "s number 42 2 3 h number b 5"
    assert normalize_address("37B, Pushtikar Chs Ltd") == "37b pushtikar chs ltd"
    assert normalize_address("Sub Plot No.-L6/29") == "sub plot number l6 29"
    assert normalize_address("Plot No.-780") == "plot number 780"
    assert normalize_address("A-115, Neb Sarai") == "a 115 neb sarai"    