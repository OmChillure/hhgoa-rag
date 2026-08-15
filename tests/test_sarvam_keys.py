from voice_rag.config import Settings


def test_collects_numbered_and_csv_keys():
    s = Settings(
        _env_file=None,
        sarvam_api_key="aaa",
        sarvam_api_key_2="bbb",
        sarvam_api_key_3="ccc",
        sarvam_api_key_4="",
        sarvam_api_key_5="aaa",
        sarvam_api_keys="ddd, bbb ,eee",
    )
    assert s.sarvam_key_list() == ["aaa", "bbb", "ccc", "ddd", "eee"]


def test_empty_pool():
    s = Settings(
        _env_file=None,
        sarvam_api_key="",
        sarvam_api_key_2="",
        sarvam_api_key_3="",
        sarvam_api_key_4="",
        sarvam_api_key_5="",
        sarvam_api_keys="",
    )
    assert s.sarvam_key_list() == []
