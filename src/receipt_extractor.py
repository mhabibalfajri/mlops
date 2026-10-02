import json, os
from pathlib import Path
import lmstudio as lms

import sys
IMAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/nota-sample.png")
MODEL = os.environ["LM_STUDIO_MODEL"]
image = lms.prepare_image(str(IMAGE))
model = lms.llm(MODEL)
chat = lms.Chat()
chat.add_user_message(
  "Baca nota. Ekstrak merchant, tanggal, item, subtotal, pajak, dan total. "
  "Keluarkan JSON valid. Jika pajak tidak terlihat, isi 0. Jangan mengarang.",
  images=[image],
)
prediction = model.respond(chat)
text = prediction.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
result = json.loads(text)
Path(f"reports/{IMAGE.stem}.json").write_text(
  json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(result, indent=2, ensure_ascii=False))
