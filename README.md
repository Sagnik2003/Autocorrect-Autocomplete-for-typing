
# 🧠 Autocorrect + Autocomplete Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black.svg?logo=flask)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)

A sleek Flask-powered web application that provides **real-time autocorrect** and **autocomplete** suggestions — just like a smart text editor or mobile keyboard.  
Designed to feel natural, intelligent, and fast. ✨

---

## 🎬 Preview

![Demo GIF Placeholder](https://github.com/Sagnik2003/Autocorrect-Autocomplete-for-typing/blob/main/project%20demo.gif)  


---

## 🚀 Features

✅ **Autocorrect:** Fixes misspelled words instantly and displays blue suggestion buttons.  
✅ **Autocomplete:** Predicts the next possible word based on input context (green suggestion buttons).  
✅ **Smart Interaction:** Works even after applying corrections; keeps the cursor active for continuous typing.  
✅ **Responsive UI:** Built with lightweight vanilla JS for speed (~90 WPM friendly).  
✅ **Modular Design:** Separate Flask endpoints for autocorrect and autocomplete logic.

---

## 🧩 Project Structure

```

📂 project/
│
├── app.py               # Flask backend
├── templates/
│   └── index.html       # Main HTML UI
├── static/
│   ├── style.css        # CSS for UI styling
│   └── script.js        # JS handling real-time suggestion logic
└── README.md            # You're reading this!

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/autocorrect-autocomplete-app.git
cd autocorrect-autocomplete-app
````

### 2️⃣ Install Dependencies
use the command to install the dependencies
~pip install -r requirements.txt`


### 3️⃣ Run the App

```bash
python app.py
```

Then visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 How It Works

### **Frontend (JavaScript)**

* Captures input events.
* Sends requests to Flask endpoints using `fetch()` after a short debounce (~1s).

### **Backend (Flask)**

* `/autocorrect` → Suggests spelling corrections.
* `/autocomplete` → Predicts likely next words.

### **Response Rendering**

* **Blue buttons** → autocorrect suggestions
* **Green buttons** → autocomplete predictions
* Clicking a suggestion inserts it instantly without breaking typing flow.

---

## 💡 Customization

You can easily modify:

* **Models:** Swap in your own ML models (e.g., spaCy, transformer-based)
* **Timing:** Adjust debounce delay in `script.js` (default = 1000ms)
* **UI Theme:** Change colors, glow effects, or button animations in `style.css`

---

## 🧪 Example Endpoints

### `/autocorrect`

**Request**

```json
{ "text": "thier" }
```

**Response**

```json
{ "suggestions": ["their", "there", "tier"] }
```

### `/autocomplete`

**Request**

```json
{ "text": "I am going" }
```

**Response**

```json
{ "suggestions": ["to", "home", "out"] }
```

---

## 🎨 Styling Highlights

| Element  | Color       | Purpose                  |
| -------- | ----------- | ------------------------ |
| 🔵 Blue  | `#00aaff`   | Autocorrect suggestions  |
| 🟢 Green | `#00ff99`   | Autocomplete predictions |
| ✨ Hover  | Subtle glow | Modern feedback effect   |

---

## 👨‍💻 Author

**Sagnik Kayal**
[![GitHub](https://img.shields.io/badge/GitHub-SagnikKayal-black?logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourlinkedin)

---

## 📜 License

This project is licensed under the **MIT License** © 2025 — Free to use, modify, and share with attribution.

---

## 🌟 Future Enhancements

* Transformer-based semantic completion
* Multilingual autocorrect support
* Grammar correction (BERT/T5-based)
* Integration with text editors or notebooks

---

## 🧩 Quick Demo Workflow

1. Start typing a sentence.
2. Blue buttons appear for correction for misspelled words.
3. Green buttons predict your next words.
4. Click one — it inserts automatically, and typing continues seamlessly.
5. Suggestions keep updating in real time.

---

> ✍️ *This app bridges machine intelligence with natural typing — blending predictive language models and classic correction logic into one smooth, interactive experience.*

---

