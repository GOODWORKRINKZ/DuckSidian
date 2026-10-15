"""Подсказка личного режима: сценарий «пришли картинку X» и запрет описывать невиденное."""
from bot.agent.prompts import PERSONAL_SYSTEM


def test_personal_prompt_send_image_scenario() -> None:
    p = PERSONAL_SYSTEM
    i = p.index("Пришли / покажи / найди картинку")
    chain = p[i:i + 900]
    # web_search/fetch_url → fetch_image → describe_media → IMAGES, именно в таком порядке
    order = [chain.index(s) for s in ("web_search", "fetch_image(url)", "describe_media(path)", "IMAGES")]
    assert order == sorted(order)
    assert "ТОЛЬКО файл, который" in chain and "реально есть в vault" in chain
    assert "НИКОГДА не описывай содержимое изображения" in chain


def test_personal_prompt_images_block_requires_real_file() -> None:
    assert "файл реально есть в vault" in PERSONAL_SYSTEM
    assert "если ты посмотрел файл через describe_media и он уместен" not in PERSONAL_SYSTEM
