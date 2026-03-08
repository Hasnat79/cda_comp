from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import json
import random
import os
import pandas
# add path of the .. folder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image
from torchvision import transforms
from model import FeedBunkClassifier
import cv2
import numpy as np
from data import train_dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.unique_scores)
model = FeedBunkClassifier(num_classes)
app = Flask(__name__)
# Safe Inference
def infer(model,model_path,img_path):
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)  # safe: handles CPU/GPU saved models
    
    # Load state dict
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    img = Image.open(img_path).convert('RGB')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    input_tensor = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()
        conf = torch.max(probs).item()
    print(f"Inference - Predicted: {pred}, Confidence: {conf:.2f}")
    return pred

example_img = '/data/hma18/CDA_hackathon/FBSI/dataset/Score 1/score-1_0.jpg'  # Adjust
model_save_path = "/data/hma18/CDA_hackathon/cda_comp/outputs/models/advanced_vit_best.pth"
# infer(model, model_save_path, example_img)



# Sample data structure for bunks

def get_status(score):
    if score <= 1:
        return "low" #low
    if score <= 2:
        return "moderate"
    return "high" #high

def get_action(score):
    if score <= 1:
        return "Increase"
    if score == 2:
        return "Maintain"
    return "Reduce"

def get_name(bunk_id):
    return f"Bunk {bunk_id + 1}"
def get_adjustment(score):
    if score < 1:
        return "increase"
    if score >1:
        return "decrease"
    return "maintenance"

def get_score(bunk_id):
    # return random.choice([0, 0.5, 1, 2, 3, 4])
    img_paths=[
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-0_1.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-0.5_4.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-1_4.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-2_55.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-3_3.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-4_0.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-0_26.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-0_82.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-0.5_43.jpg",
        "/data/hma18/CDA_hackathon/cda_comp/src/User_Dashboard_Interface_Farmer/static/score-2_44.jpg"
    ]
    data = {
        0:infer(model, model_save_path, img_paths[0]),
        1:infer(model, model_save_path, img_paths[1]),
        2:infer(model, model_save_path, img_paths[2]),
        3:infer(model, model_save_path, img_paths[3]),
        4:infer(model, model_save_path, img_paths[4]),
        5:infer(model, model_save_path, img_paths[5]),
        6:infer(model, model_save_path, img_paths[6]),
        7:infer(model, model_save_path, img_paths[7]),
        8:infer(model, model_save_path, img_paths[8]),
        9:infer(model, model_save_path, img_paths[9])
    }
    return data.get(bunk_id, 2)  # default to 2 if bunk_id is out of range
# def get_score(bunk_id):
#     # return random.choice([0, 0.5, 1, 2, 3, 4])
#     data = {
#         0:0,
#         1:0.5,
#         2:1,
#         3:2,
#         4:3,
#         5:4,
#         6:0.5,
#         7:3,
#         8:2,
#         9:1
#     }
#     return data.get(bunk_id, 2)  # default to 2 if bunk_id is out of range


def update_bunk_fields(bunk_data):
    for bunk in [0,1,2,3,4,5,6,7,8]:
        bunk_data[bunk] = {}
    for bunk_id, data in bunk_data.items():
        print(f"Processing Bunk {bunk_id}...")
        data['score'] = get_score(bunk_id)
        print(f"Bunk {bunk_id} score: {data['score']}")
        data['status'] = get_status(data['score'])
        data['action'] = get_action(data['score'])
        data['name']   = get_name(bunk_id)
        data['adjustment'] = get_adjustment(data['score'])
bunks_data = {}
update_bunk_fields(bunks_data)
# print(bunks_data)




# Generate trend data
# def generate_trend_data():
#     trend = []
#     now = datetime.now()
#     for i in range(24, -1, -1):
#         time = now - timedelta(hours=i)
#         # Create realistic trend with some variation
#         base_score = 2 - (i / 24) * 2
#         score = base_score + random.uniform(-0.3, 0.3)
#         trend.append({
#             "time": time.strftime("%I %p").lstrip("0"),
#             "score": round(max(0, min(4, score)), 2),
#             "timestamp": time.isoformat()
#         })
#     return trend
def generate_trend_data():
    trend = []
    now = datetime.now()
    
    # We'll generate the last 25 hours so we always cover a full cycle
    for i in range(24, -1, -1):
        dt = now - timedelta(hours=i)
        hour = dt.hour
        
        # Core pattern: 
        #   6:00   → score ≈ 4.0
        #   15:00  → score ≈ 1.0–1.2
        #   6:00   → score ≈ 4.0 again
        
        # Distance from 6 AM in hours (shortest in the cycle)
        hours_from_6am = (hour - 6) % 24
        
        # First half: morning → afternoon decline (6am to 3pm = 9 hours)
        if hours_from_6am <= 9:
            # linear drop from 4.0 to ~1.1 over 9 hours
            base_score = 4.0 - (hours_from_6am / 9.0) * 2.9
        else:
            # afternoon → next morning rise (3pm to 6am next = 15 hours)
            remaining_hours = hours_from_6am - 9
            base_score = 1.1 + (remaining_hours / 15.0) * 2.9
        
        # Add small realistic noise
        score = base_score + random.uniform(-0.25, 0.25)
        
        # Keep in realistic bounds (e.g. 0.5–4.5)
        score = max(0.5, min(4.5, score))
        
        trend.append({
            "time": dt.strftime("%I %p").lstrip("0"),     # e.g. "6 AM", "3 PM"
            "score": round(score, 2),
            "timestamp": dt.isoformat()
        })
    
    # Optional: sort oldest → newest (if your chart expects chronological order)
    trend.sort(key=lambda x: x["timestamp"])
    
    return trend

trend_data = generate_trend_data()
# print(trend_data)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/bunks')
def get_bunks():
    return jsonify(bunks_data)

@app.route('/api/bunks/<int:bunk_id>')
def get_bunk(bunk_id):
    if bunk_id in bunks_data:
        return jsonify(bunks_data[bunk_id])
    return jsonify({"error": "Bunk not found"}), 404


@app.route('/api/bunks/<int:bunk_id>', methods=['PUT'])
def update_bunk(bunk_id):
    if bunk_id not in bunks_data:
        return jsonify({"error": "Bunk not found"}), 404
    
    data = request.json
    bunks_data[bunk_id].update(data)

    update_bunk_fields(bunks_data)
    return jsonify(bunks_data[bunk_id])

@app.route('/api/trend')
def get_trend():
    return jsonify(trend_data)

@app.route('/api/summary')
def get_summary():
    increase_count = sum(1 for b in bunks_data.values() if b['action'] == 'Increase')
    maintain_count = sum(1 for b in bunks_data.values() if b['action'] == 'Maintain')
    reduce_count = sum(1 for b in bunks_data.values() if b['action'] == 'Reduce')
    
    return jsonify({
        "bunks_needing_more": {"increase": increase_count, "moderate": maintain_count},
        "bunks_to_reduce": reduce_count,
        "total_bunks": len(bunks_data),
        "system_status": "OK",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/recommend')
def get_recommendations():
    increase_bunks = [bid for bid, b in bunks_data.items() if b['action'] == 'Increase']
    reduce_bunks = [bid for bid, b in bunks_data.items() if b['action'] == 'Reduce']
    
    return jsonify({
        "increase_feed": increase_bunks,
        "reduce_feed": reduce_bunks,
        "recommendations": [
            f"Increase Feed for Bunk {', '.join(map(str, increase_bunks))}" if increase_bunks else None,
            f"Reduce Feed for Bunk {', '.join(map(str, reduce_bunks))}" if reduce_bunks else None
        ]
    })

@app.route('/api/bunk-images')
def get_bunk_images():
    
    # image_dir = os.path.join(app.static_folder, 'bunk_images')
    image_dir = "/data/hma18/CDA_hackathon/FBSI/dataset/Score 4"
    
    if not os.path.exists(image_dir):
        return jsonify({"error": "Image directory not found"}), 404
    
    # Get all image files (you can add more extensions if needed)
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
        and os.path.splitext(f)[1].lower() in allowed_extensions
    ]
    
    return jsonify({
        "images": images,                    # just filenames
        "base_url": "/static/bunk_images/"   # prefix to make full URLs
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
