#  AI Gym & Fitness Assistant

A comprehensive AI-powered fitness platform built in a single Google Colab notebook, combining computer vision, machine learning, NLP, and IoT simulation across **7 specialized modules**.

---

##  Quick Start

1. Open in [GoogleColab](https://docs.google.com/document/d/1eJx-rOO4UXB34648eZbbxeTyYYBKbwePFV63_fij-DQ/edit?usp=sharing)
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run **Step 1** to install dependencies
4. Run **Step 2** to import libraries
5. Execute any module cell(s) you want to use

---

## Installation

```bash
pip install ultralytics scikit-learn pandas numpy matplotlib seaborn fastapi uvicorn nest_asyncio
```

---

## Modules

### Module 1 — AI Gym Trainer (YOLOv8 Pose Detection)
Detects exercise repetitions and provides real-time form feedback using the YOLOv8 pose model.

- Loads `yolov8n-pose.pt` (~6 MB) to detect **17 COCO body keypoints** per frame
- Computes joint angles from keypoint triplets (e.g. shoulder–elbow–wrist)
- Counts reps and flags form issues using angle thresholds
- Simulates realistic keypoint trajectories for 6 exercises in the notebook environment

**Supported exercises:** `bicep_curl`, `squat`, `pushup`, `shoulder_press`, `deadlift`, `lunge`

```python
run_yolo_workout('bicep_curl', total_reps=10)
```

**Output:** Joint angle chart, rep count graph, form quality pie chart, skeleton visualisation, keypoint confidence bars.

---

### Module 2 — AI Dietician & Calorie Coach
Generates personalised meal plans based on body metrics and fitness goals.

- Calculates **BMR** (Mifflin-St Jeor), **TDEE**, and **BMI**
- Adjusts daily calorie target: −500 kcal (weight loss), +300 kcal (muscle gain), or maintenance
- Pulls from a 14-food nutrition database across 5 meal slots

```python
user = {
    'name': 'Alex', 'weight_kg': 75, 'height_cm': 175,
    'age': 28, 'gender': 'male',
    'goal': 'muscle_gain',        # weight_loss | muscle_gain | maintenance
    'activity_level': 'moderate'  # sedentary | light | moderate | active | very_active
}
generate_meal_plan(user)
```

**Output:** Full meal plan with macros, macronutrient pie chart, calorie-per-meal bar chart.

---

### Module 3 — Smart Gym Assistant (IoT Simulation)
Simulates real-time gym equipment sensor data and provides adaptive coaching.

- Generates heart rate, speed, and calorie data over a 30-minute session
- Detects whether the user is in the optimal fat-burning HR zone (110–150 bpm)
- Issues recommendations to increase or reduce intensity

```python
session = IoTEquipment().generate_session(30)
analyze_session(session)
```

**Output:** Heart rate timeline, speed curve, per-minute calorie bar chart, text recommendations.

---

### Module 4 — AI Fitness Habit Tracker (Behavioural AI)
Predicts the likelihood of a user skipping a workout using a Random Forest classifier.

- Trained on a synthetic dataset of 300 users × 60 days
- Features: sleep hours, stress level, work hours, days since last workout, motivation score, weekend flag, previous week's workouts, current streak
- Outputs a **30-day skip-risk calendar** with colour-coded nudge alerts

**Model accuracy:** ~91%

```python
# Nudge days are printed automatically after training
```

**Output:** Feature importance chart, 30-day skip probability bar chart (green/amber/red).

---

### Module 5 — Virtual Gym Buddy (AI Chat Companion)
A rule-based NLP chatbot that responds to fitness questions and provides emotional support.

- Detects 5 intents: `motivation`, `tired`, `diet`, `progress`, `injury`
- Performs simple sentiment analysis (positive / negative / neutral)
- Adjusts response tone based on sentiment (e.g. prepends "I hear you!" for negative sentiment)

```python
response, intent, sentiment = gym_buddy_chat("I feel tired today")
```

**Output:** Console conversation demo with intent/sentiment labels and a daily motivational quote.

---

### Module 6 — Pose-to-Performance Analyzer
Tracks and scores workout performance across an 8-week training block.

Composite score formula:

| Component    | Weight |
|--------------|--------|
| Form %       | 40%    |
| Consistency  | 25%    |
| Endurance    | 20%    |
| Effort (HR)  | 15%    |

```python
weekly_report('Alex', weeks=8)
```

**Output:** Weekly score trend, stacked bar breakdown, radar chart for the latest week, reps-per-week bar chart.

---

### Module 7 — Gym Recommender & Planner
Recommends gyms and weekly workout programs using content-based filtering.

- Scores 8 gyms across distance, monthly fee, rating, and requested amenities
- Returns the top 3 matches with a full feature breakdown
- Pairs the recommendation with a structured weekly program (8–12 weeks)

```python
prefs = {'goal': 'muscle_gain', 'budget': 3000, 'features': ['weights', 'pt']}
top_gyms = recommend_gyms(prefs)
```

**Available goals:** `weight_loss`, `muscle_gain`, `endurance`

**Output:** Match score chart, price-vs-rating scatter plot, full weekly schedule.

---

##  FastAPI REST API

A FastAPI backend with 6 endpoints is defined in the notebook and can be tested locally.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/diet/meal-plan` | Generate meal plan |
| POST | `/api/buddy/chat` | Chat with Gym Buddy |
| POST | `/api/gyms/recommend` | Recommend gyms |
| POST | `/api/workout/simulate` | Simulate workout reps |
| GET | `/api/habit/risk` | Get skip-risk score |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Pose Detection | YOLOv8 (`ultralytics`) |
| ML / Classification | scikit-learn (Random Forest) |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| API Framework | FastAPI + Uvicorn |
| Environment | Google Colab (T4 GPU recommended) |

**Suggested production stack:** React / Next.js frontend · Python + FastAPI backend · MongoDB or PostgreSQL · MQTT / Node-RED for IoT · AWS S3 or Firebase for storage.

---

##  Configuration

Edit the user profile in **Module 2** and preferences in **Module 7** to personalise outputs:

```python
# Module 2 — user profile
user = {
    'name': 'Your Name',
    'weight_kg': 70,
    'height_cm': 170,
    'age': 25,
    'gender': 'female',
    'goal': 'weight_loss',
    'activity_level': 'active'
}

# Module 7 — gym preferences
prefs = {
    'goal': 'weight_loss',
    'budget': 2000,
    'features': ['classes', 'pool']
}
```

---

##  Notes

- Pose detection in the notebook **simulates** YOLO keypoints. To use a real video feed, replace the simulation call with:
  ```python
  results = yolo_model("your_video.mp4", stream=True)
  for r in results:
      kps = r.keypoints.data[0].cpu().numpy()  # shape (17, 3)
      cnt, stage, angle = counter.process_keypoints(kps)
  ```
- The habit tracker and dietician modules use synthetic data; swap in real user data for production use.
- The FastAPI server requires `nest_asyncio` to run inside a Colab notebook environment.
