# AIllu · Let's work together

**Small AI for coffee cooperatives in the Andes.** A collection officer photographs 100 coffee beans; a 0.49 MB model running **offline in the phone's browser** counts the defects, estimates the SCA grade and a reference price, and sends the farmer an **SMS + audio in Quechua or Spanish**. Farmers report their harvest by SMS (in Quechua, Spanish or a mix), and the cooperative pools those notices into a full lot for a buyer, who reads everything in English.

> Built for the **World Bank × Hack-Nation "Small AI for Development" Hackathon 2026 — Agriculture challenge (Annex B)**.
> *Ayllu* is the Quechua word for the Andean community that works together. AIllu puts AI at the service of that community.

- **Live demo:** `<your Vercel URL>` (the `app/` folder, static, no build step)
- **Videos:** Demo (60 s) · Tech (60 s) · Team (60 s)
- **Languages:** the whole UI switches between Spanish, Quechua (Runasimi) and English (for the jury).

---

## The problem

Noor farms 2 ha of coffee in Cusco, Peru. Her yields are falling and she doesn't know why; the extension officer visits twice a year, and at harvest the middleman sets the price. In Peru only **3.8 % of farmers received technical assistance in 2022** (ENA 2022, INEI) and **223,000 families** live from coffee (MIDAGRI).

**Problem statement.** With AIllu, Noor will know the grade and reference price of her coffee, and the likely cause of her lower yield (coffee berry borer), *before* the middleman arrives.

## How it works (4 roles)

| Role | What they do | Where AI is |
|---|---|---|
| Collection officer | 4 guided steps: choose member → photo of 100 beans → review → send | Classical segmentation + a 123k-parameter CNN that classifies each bean into 11 defect classes |
| Member (basic phone) | Receives SMS + audio; replies "pusaq qq pergamino pa noviembre" | SMS reader (char n-gram logistic regression + rules for Quechua numerals); pre-recorded voice |
| Cooperative manager | Handles unclear SMS, sees the lot forecast, publishes when there is an 80 % chance of filling a container, replies to the buyer | Forecast is a simulation (no AI) |
| Buyer | Sees the lot in English with no personal data; messages arrive translated | Translated templates (no free-form translation) |

**Guardrails.** If the photo is not enough the tool says *"not sure — take your coffee to the cupper"*; Specialty grade is never confirmed from a photo; prices (World Bank Pink Sheet + cooperative differential) and advice are fixed, reviewable texts; a person confirms every send and every publication.

## Results

| Component | Result |
|---|---|
| Bean classifier (176 held-out beans) | 98.9 % sound vs. defect · 83 % exact class (11 classes) |
| End-to-end, 60 synthetic 100-bean photos | 90 % correct grade · 27/27 "reliable" results correct · mean error 0.7 defects/sample |
| SMS reader (305 new messages, different generator) | 90.5 % intent · 95.3 % quantity · 98.8 % unit · 3/172 silent errors |
| Speed | ~60 ms per bean in JavaScript (≈ 6 s per photo) |

## Data (and what it doesn't cover)

| Data | Source / licence |
|---|---|
| 582 individual green beans, 11 classes | [Green Coffee Beans Dataset](https://github.com/j4ckdev/greencoffeebeansdataset) (Colombia), CC BY-NC-SA 4.0 |
| 60 + 2 sample photos of ~100 beans | **Synthetic**: composites of held-out beans on paper backgrounds |
| 1,199 + 308 + 305 SMS | **Synthetic**, generated with Claude (Haiku 4.5 train / Sonnet 4.6 test) |
| 50-member registry, 34 harvest notices | **Synthetic** |
| Advice texts, Quechua translations | Drafts generated with Claude — **pending review by an agronomist and native speakers** |
| Quechua voice | `facebook/mms-tts-quz` (Meta MMS), CC BY-NC 4.0 |
| Spanish & English voice | ElevenLabs (eleven_v3) |
| Arabica price | World Bank Pink Sheet (Aug 2026): Jul 2026 = 7.91 US$/kg |

**Not covered yet:** no Peruvian beans and no real field photos; few examples of fungus/unripe beans; borderline samples can vary slightly between devices (e.g. 74–88 %), which the tool flags as uncertain; the Quechua voice is synthetic.

## Repository structure

```
app/                  ← the hosted demo (static site, no build)
  index.html          UI (Spanish / Quechua / English)
  engine.js           offline engine: segmentation, CNN forward pass, SCA grading, price, SMS reader, forecast
  granos.b64.txt      CNN weights (base64, 0.49 MB) + granos_manifest.json
  intent_model.json   SMS intent classifier
  ui.json             UI strings in es / qu / en
  data.json           config, advice, messages, synthetic registry
  img/  audio2/       sample photos, pre-recorded audio (es / qu / en)
training/             how the model and data were built (Python)
  vision/             crop.py → compose.py → train.py → evaluate.py
  sms/                gen_sms.py (Claude API) → train_intent.py, parser.py
  logic/              SCA table, probabilistic grading, price
  content/ coop/ audio/ i18n/   advice, synthetic registry, voice generation, UI translation
docs/                 technical sheet and video script (Spanish)
```

## Setup

### Run the demo locally (no install)
```bash
git clone https://github.com/<your-user>/aillu.git
cd aillu/app
python3 -m http.server 8000      # any static server works
# open http://localhost:8000
```
Everything runs in the browser; after the page loads, grading works without internet.

### Deploy on Vercel
1. Import the GitHub repo in Vercel → **Framework preset: Other**.
2. **Root Directory: `app`**, no build command, no output directory.
3. Deploy. (Also works on GitHub Pages, Netlify or Replit as a static site.)

### Reproduce the training (optional)
```bash
pip install torch timm opencv-python-headless scikit-learn numpy soundfile transformers
git clone https://github.com/j4ckdev/greencoffeebeansdataset data/GreenCoffeeBeansDataset
cd training/vision
python crop.py && python compose.py
QP_S=80 QP_W=16,32,48,64,96 QP_OUT=net_small.pt QP_E=50 python train.py
QP_S=80 QP_W=16,32,48,64,96 python evaluate.py
```
Paths in the scripts point to `/home/claude/hack/...` (the original build machine); adjust them to your folder. Scripts that call the Claude or ElevenLabs APIs read `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` and `ELEVENLABS_API_KEY` from a `.env` file that is **not** included.

## Tools
PyTorch (training, CPU) · own dependency-free JavaScript inference engine · OpenCV · scikit-learn · Claude API (synthetic SMS, advice drafts, translations) · Meta MMS-TTS (Quechua) · ElevenLabs (Spanish/English voice) · Playwright + ffmpeg (testing and videos).

## Team
Diego Guevara Orosco — Lima, Peru.

## License
Code: MIT. Data and models keep their original licences (see table above); the bean dataset and MMS-TTS are non-commercial.
