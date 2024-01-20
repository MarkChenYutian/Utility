import json
from typing import Literal

with open("color_darker.json", "r") as f: DARKER_COLOR = json.load(f)
with open("color_lighter.json", "r") as f: LIGHTER_COLOR = json.load(f)

def getColor(category: Literal["+", "-"], base_idx: int, variant: int):
    if category == "+": return LIGHTER_COLOR[base_idx][variant]
    elif category == "-": return DARKER_COLOR[base_idx][variant]
    raise ValueError(f"Receive unexpected category {category}")
