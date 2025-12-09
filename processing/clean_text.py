import re

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZČĆĐŠŽčćđšž0-9 ,.!?;:\-]", "", text)
    return text.strip()
