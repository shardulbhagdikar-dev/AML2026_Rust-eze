# Desktop CatBoost Training Package (GTX 1650 / Multi-Core CPU)

Train an orthogonal CatBoost model on your second machine to create an architectural ensemble with our LightGBM and XGBoost models.

---

### Step 1: Pull Latest Code
On your desktop machine:
```bash
git pull origin main
```

---

### Step 2: Install Requirements
```bash
pip install -r amazon_er/models/desktop_trainer/requirements_desktop.txt
```

---

### Step 3: Run Training
```bash
# If your GTX 1650 has CUDA enabled:
python amazon_er/models/desktop_trainer/train_desktop_catboost.py --task-type GPU

# If running on CPU:
python amazon_er/models/desktop_trainer/train_desktop_catboost.py --task-type CPU
```

---

### Step 4: Push Model Artifact to GitHub
Once training finishes, it saves `catboost_model.cbm` (~5-8 MB) in this directory.
Push it to GitHub:
```bash
git add amazon_er/models/desktop_trainer/catboost_model.cbm
git commit -m "feat(models): add trained CatBoost desktop model"
git push origin main
```
On your laptop, you can then `git pull` and blend CatBoost directly into `generate_submission_v2.py`!
