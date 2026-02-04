# chatbot.py — multilingual, concise tips
from typing import Literal

def chat_response(user_input: str, lang: Literal["en","hi","te","ta","kn"]="en") -> str:
    ui = (user_input or "").lower()
    tips = {
        "fertilizer": "Split doses: Urea (N), DAP (P), MOP (K). Avoid excess near flowering.",
        "pest": "Scout every 3 days. Use pheromone traps. Spray only when threshold is crossed.",
        "yield": "Keep soil pH near 6.5; irrigate at night; time fertilizer by stage.",
        "irrigation": "Irrigate at night, 18–24 mm in two cycles. Reduce if heavy rain is forecast.",
        "soil": "If acidic add lime; if alkaline add gypsum—per soil test."
    }
    base = tips["yield"]
    if "fertilizer" in ui: base = tips["fertilizer"]
    elif "pest" in ui: base = tips["pest"]
    elif "irrigation" in ui: base = tips["irrigation"]
    elif "soil" in ui: base = tips["soil"]

    local = {
        "en": base,
        "hi": "खाद को हिस्सों में दें। रात में सिंचाई (18–24 मिमी, 2 चक्र)। आवश्यकता पर ही छिड़काव करें.",
        "te": "ఎరువులు విడతలుగా ఇవ్వండి. రాత్రి నీరు (18–24 మిమీ, 2 సైకిళ్లు). అవసరమైతేనే స్ప్రే చేయండి.",
        "ta": "உரங்களை பிரித்து இடுங்கள். இரவில் பாசனம் (18–24 மிமீ, 2 சுற்று). தேவைப்பட்டால் மட்டுமே தெளிக்கவும்.",
        "kn": "ಗೊಬ್ಬರವನ್ನು ಹಂತವಾಗಿ ನೀಡಿ. ರಾತ್ರಿ ನೀರಾವರಿ (18–24 ಮಿಮೀ, 2 ಸೈಕಲ್). ಅಗತ್ಯವಿದ್ದಾಗ ಮಾತ್ರ ಸಿಂಪಡಿಸಿ."
    }
    return local.get(lang, base)
