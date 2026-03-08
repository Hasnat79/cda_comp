# 🚀 Quick Start Guide - Feed Bunk Dashboard

## 30 Seconds to Running

### Step 1: Install Flask
```bash
pip install Flask==2.3.2 Werkzeug==2.3.6
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
python app.py
```

### Step 3: Open Browser
Go to: **http://localhost:5000**

That's it! You now have a fully functional Feed Bunk Dashboard.

---

## Project Files

```
📦 feed-bunk-dashboard/
├── 📄 app.py                 ← Main Python application (Flask backend)
├── 📄 requirements.txt        ← Python dependencies
├── 📄 README.md             ← Full documentation
├── 📄 QUICKSTART.md         ← This file
└── 📁 templates/
    └── 📄 dashboard.html    ← Frontend (HTML/CSS/JavaScript)
```

---

## What's Included

✅ **Complete Dashboard** with 7 sections
✅ **REST API** with 6 endpoints
✅ **Real-time Charts** using Chart.js
✅ **Responsive Design** for all devices
✅ **Mock Data** pre-loaded for immediate use
✅ **Dark Theme** with modern glassmorphism UI

---

## Dashboard Features

### 📊 Bunk Overview
- Score-based system (0-4 scale)
- Color-coded recommendations
- Quick reference table

### 📷 Feed Bunk Monitoring
- Live image gallery (6 bunks)
- Status indicators (Low/Moderate/High)
- Real-time action recommendations

### 📈 Score Trend
- 24-hour historical data
- Interactive Chart.js visualization
- Time-series analysis

### 📋 Daily Summary
- Bunks needing more feed
- Bunks to reduce
- Recommended actions

### ⚡ Quick Look
- Latest scores at a glance
- Color-coded actions
- Fast reference cards

---

## Customizing the Dashboard

### Add More Bunks
Edit `app.py`, find `bunks_data` and add:
```python
8: {"name": "Bunk 8", "score": 1.5, "action": "Maintain", "status": "Low", "feed_level": 50},
```

### Change Colors
Edit CSS in `dashboard.html`:
- Find `.score-0`, `.score-1`, `.score-2`, `.score-3` classes
- Modify the `background` and `color` values

### Modify Recommendations
Edit the `get_recommendations()` function in `app.py`

### Connect Real Data
Replace `bunks_data` dictionary with database queries

---

## API Endpoints (for integration)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/bunks` | GET | All bunks data |
| `/api/bunks/1` | GET | Specific bunk |
| `/api/bunks/1` | PUT | Update bunk |
| `/api/summary` | GET | Daily stats |
| `/api/recommend` | GET | Recommendations |
| `/api/trend` | GET | 24-hour trend |

---

## Troubleshooting

**Port 5000 already in use?**
```bash
python app.py
# Then edit app.py last line: port=8080
```

**Module errors?**
```bash
pip install -r requirements.txt
```

**Nothing showing?**
1. Check http://localhost:5000 (not https)
2. Check browser console (F12)
3. Restart the Flask server

---

## Next Steps

1. ✅ Run the application
2. ✅ Explore the dashboard
3. ✅ Customize with your data
4. ✅ Deploy to production (see README.md)

---

## Production Deployment

Ready for real farm use? See README.md for:
- Gunicorn setup
- Docker containerization
- Cloud deployment options
- Security best practices

---

**Happy farming! 🐄**
