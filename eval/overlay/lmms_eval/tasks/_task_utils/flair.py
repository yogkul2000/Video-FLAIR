import re

FLAIR_POST = {
    "mcq": "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer>...</answer> tags.\nExample:\n<answer>A</answer>",
    "number": "Please provide only the numerical value within the <answer>...</answer> tags.\nExample:\n<answer>3.14</answer>",
    "math": "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\nExample:\n<answer>$$-\\dfrac{3}{2}$$</answer>",
    "open": "Please provide only your text answer within the <answer>...</answer> tags.\nExample:\n<answer>The capital of France is Paris.</answer>",
    "short": "Please provide only a single word or short phrase within the <answer>...</answer> tags.\nExample:\n<answer>red</answer>",
    "yesno": "Please answer only yes or no within the <answer>...</answer> tags.\nExample:\n<answer>yes</answer>",
}

MODES = ("DIRECT", "CONCISE", "DEEP")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_WRAP = re.compile(r"</?(?:DIRECT|CONCISE|DEEP|ADAPTIVE|think|answer)>")
_OBJ = re.compile(r"<obj>(.*?)</obj>", re.S)
_BOX = re.compile(r"<box>.*?</box>", re.S)
_T = re.compile(r"<t>.*?</t>", re.S)


def is_flair(kwargs):
    return bool(kwargs) and kwargs.get("flair") is True


def flair_post(kind):
    return FLAIR_POST[kind]


def strip_grounding(text):
    text = _OBJ.sub(r"\1", text)
    text = _BOX.sub("", text)
    text = _T.sub("", text)
    return text


def flair_answer(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        return text
    found = _ANSWER.findall(text)
    if found:
        out = found[-1]
    elif "<answer>" in text:
        out = text.rsplit("<answer>", 1)[1]
    elif "</think>" in text:
        out = text.rsplit("</think>", 1)[1]
    else:
        out = text
    out = _WRAP.sub("", strip_grounding(out))
    return out.strip()


def flair_mode(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"<(DIRECT|CONCISE|DEEP)>", text)
    return m.group(1) if m else None
