import re

def clean_menu(text):
    lines = text.split("\n")
    
    cleaned = []
    current_category = "Unknown"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # category detection
        if line.isupper() or "Specials" in line or "Curries" in line:
            current_category = line.strip()
            cleaned.append(f"\nCATEGORY: {current_category}\n")
            continue

        # split item + price
        match = re.match(r"(.+?)\$(\d+\.\d{2})(.*)", line)
        if match:
            item = match.group(1).strip()
            price = match.group(2)
            desc = match.group(3).strip()

            cleaned.append(f"ITEM: {item}")
            cleaned.append(f"PRICE: ${price}")
            cleaned.append(f"DESCRIPTION: {desc}\n")
        else:
            cleaned.append(line)

    return "\n".join(cleaned)