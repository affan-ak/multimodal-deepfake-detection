# Complete Step-by-Step Instructions

This guide assumes zero experience with GitHub or SageMaker. Follow every step in order.

---

## PART 1: Set Up GitHub (One-time, on your laptop)

### Step 1.1: Create a GitHub Account

1. Go to https://github.com
2. Click "Sign Up"
3. Use your personal email (not work email)
4. Choose a username (e.g., `affankazim` or `affan-ahmed-kazim`)
5. Complete the verification

### Step 1.2: Install Git on Your Mac

Open Terminal (press Cmd+Space, type "Terminal", hit Enter) and run:

```bash
git --version
```

If it prompts you to install Xcode Command Line Tools, click "Install" and wait.

### Step 1.3: Configure Git with Your Identity

In Terminal, run (replace with your info):

```bash
git config --global user.name "Affan Ahmed Kazim"
git config --global user.email "affan.kazim@gmail.com"
```

### Step 1.4: Create a GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name it: "SageMaker access"
4. Check the box for "repo" (full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN NOW** — you won't see it again. Save it somewhere safe (e.g., Apple Notes)

### Step 1.5: Create the Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `multimodal-deepfake-detection`
3. Description: "Multimodal deepfake detection framework with reproducible evaluation harness and explainability artifacts"
4. Choose **Public** (you want this visible for your petition)
5. Check "Add a README file"
6. Under "Add .gitignore" select "Python"
7. Under "Choose a license" select "Apache License 2.0"
8. Click "Create repository"

### Step 1.6: Clone the Repository to Your Mac

In Terminal:

```bash
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/multimodal-deepfake-detection.git
```

It will ask for your username and password. For the password, **use the token from Step 1.4** (not your actual GitHub password).

Now copy all the project files into this folder:

```bash
cp -r "/Users/affankaz/NIW Claude/deepfake-detection-project/"* ~/Desktop/multimodal-deepfake-detection/
```

---

## PART 2: Get the DFDC Dataset from Kaggle

### Step 2.1: Create a Kaggle Account

1. Go to https://www.kaggle.com
2. Sign up with your Google account or email

### Step 2.2: Accept the DFDC Competition Rules

1. Go to https://www.kaggle.com/competitions/deepfake-detection-challenge
2. Click "Late Submission" or "Join Competition"
3. Accept the rules (required to download the data)

### Step 2.3: Get Your Kaggle API Key

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads a file called `kaggle.json` — keep it safe, you'll need it in SageMaker

---

## PART 3: Set Up the SageMaker Notebook Instance

### Step 3.1: Open SageMaker in AWS Console

1. Sign into the AWS Console
2. Search for "SageMaker" in the top search bar
3. Click on "Amazon SageMaker"
4. In the left sidebar, click "Notebooks" → "Notebook instances"

### Step 3.2: Create a Notebook Instance

1. Click "Create notebook instance"
2. Configure as follows:
   - **Notebook instance name**: `deepfake-detection`
   - **Instance type**: `ml.g5.xlarge` (for GPU training)
   - **Volume size**: `200` GB (datasets are large)
   - **IAM role**: 
     - If you have an existing role with S3 access, select it
     - Otherwise, select "Create a new role"
     - In the popup, select "Any S3 bucket" and click "Create role"
3. Under "Git repositories" — skip this for now (we'll handle git manually)
4. Click "Create notebook instance"
5. Wait 3-5 minutes for status to change to "InService"

### Step 3.3: Open the Notebook

1. Once status is "InService", click "Open JupyterLab"
2. You now have a GPU-powered Jupyter environment

---

## PART 4: Set Up the Environment Inside SageMaker

### Step 4.1: Open a Terminal in JupyterLab

1. In JupyterLab, click File → New → Terminal

### Step 4.2: Upload and Run the Setup Script

Option A (recommended): In the JupyterLab terminal, run:

```bash
# Clone your repo (replace YOUR_USERNAME)
cd /home/ec2-user/SageMaker
git clone https://github.com/YOUR_USERNAME/multimodal-deepfake-detection.git
cd multimodal-deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

Option B: If you haven't pushed code to GitHub yet, upload the project files:
1. In JupyterLab, click the upload button (up arrow icon) in the file browser
2. Upload the entire `deepfake-detection-project` folder contents
3. Or drag-and-drop files into the file browser

### Step 4.3: Configure Kaggle API Access

In the JupyterLab terminal:

```bash
mkdir -p ~/.kaggle
```

Now upload your `kaggle.json` file (from Part 2, Step 2.3):
1. In JupyterLab file browser, navigate to the terminal's home
2. Upload `kaggle.json` to the notebook instance
3. Then in terminal:

```bash
mv /home/ec2-user/SageMaker/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4.4: Download the DFDC Dataset

In the terminal:

```bash
cd /home/ec2-user/SageMaker/multimodal-deepfake-detection

# Download DFDC sample data (approximately 10GB compressed)
# This downloads 4 parts - we only need parts 0-3 for a strong prototype
kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_00.zip
kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_01.zip
kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_02.zip
kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_03.zip

# Create data directory and unzip
mkdir -p data/raw
unzip dfdc_train_part_00.zip -d data/raw/part_00
unzip dfdc_train_part_01.zip -d data/raw/part_01
unzip dfdc_train_part_02.zip -d data/raw/part_02
unzip dfdc_train_part_03.zip -d data/raw/part_03

# Clean up zip files to save space
rm -f dfdc_train_part_*.zip
```

**Note**: Each part is ~10GB. If disk space is a concern, start with just parts 00 and 01.

---

## PART 5: Run the Pipeline (Use the Notebook)

### Step 5.1: Open the Main Notebook

1. In JupyterLab file browser, navigate to `notebooks/`
2. Double-click `01_full_pipeline.ipynb`
3. Select kernel: "conda_pytorch_p310" (or whichever PyTorch kernel is available)

### Step 5.2: Run Each Cell in Order

The notebook is organized into sections. Run cells one at a time (Shift+Enter):

1. **Setup & Imports** — verifies GPU is available
2. **Face Extraction** — extracts faces from videos (takes 2-4 hours for 4 parts)
3. **Train XceptionNet** — trains the deepfake detector (~2-3 hours)
4. **Evaluate** — generates metrics (AUC, F1, EER, confusion matrix)
5. **GradCAM Heatmaps** — generates explainability visualizations
6. **Generate Reports** — produces the evaluation report and model card

Each section has explanatory text and progress indicators.

### Step 5.3: Monitor Progress

- Face extraction is the longest step. You'll see a progress bar.
- Training shows loss/accuracy per epoch. Expect ~15-20 epochs.
- If the notebook disconnects, the instance keeps running. Just reconnect and check the `results/` folder.

---

## PART 6: Upload Everything to GitHub

### Step 6.1: Check Your Results

After the pipeline completes, verify you have:

```
results/
├── metrics/
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── heatmaps/
│   ├── sample_001_real.png
│   ├── sample_002_fake.png
│   └── ... (20-50 sample heatmaps)
└── checkpoints/
    └── xceptionnet_best.pth  (this may be too large for GitHub)
```

### Step 6.2: Push Code and Results to GitHub

In the JupyterLab terminal:

```bash
cd /home/ec2-user/SageMaker/multimodal-deepfake-detection

# Configure git identity
git config user.name "Affan Ahmed Kazim"
git config user.email "affan.kazim@gmail.com"

# Stage all new files (but NOT the dataset or large model files)
git add src/
git add scripts/
git add notebooks/
git add results/metrics/
git add results/heatmaps/
git add requirements.txt
git add MODEL_CARD.md
git add README.md
git add .gitignore

# Check what you're about to commit
git status

# Commit
git commit -m "Initial release: deepfake detection framework with evaluation harness and explainability artifacts"

# Push to GitHub
git push origin main
```

It will ask for credentials:
- Username: your GitHub username
- Password: your **Personal Access Token** from Part 1, Step 1.4

### Step 6.3: Upload the Model Checkpoint (Large File)

The model checkpoint (.pth file) is too large for regular GitHub (>100MB). Use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git add .gitattributes
git add results/checkpoints/xceptionnet_best.pth

git commit -m "Add trained model checkpoint via LFS"
git push origin main
```

**Alternative**: If Git LFS is complicated, upload the checkpoint to HuggingFace Hub instead (see Part 7).

### Step 6.4: Verify on GitHub

1. Go to https://github.com/YOUR_USERNAME/multimodal-deepfake-detection
2. You should see all your files, the README rendering nicely, results visible
3. Check the "results/heatmaps" folder — images should display inline

---

## PART 7 (Optional): Deploy a Demo on HuggingFace Spaces

This gives you a live URL where anyone (including USCIS officers) can test the model:

1. Go to https://huggingface.co and create an account
2. Click your profile → "New Space"
3. Name: `deepfake-detector`
4. Select "Gradio" as the SDK
5. Upload the `src/demo_app.py` file and model checkpoint
6. It auto-deploys in ~5 minutes
7. You get a URL like: `https://huggingface.co/spaces/your-username/deepfake-detector`

---

## PART 8: Shut Down SageMaker (IMPORTANT — Saves Money!)

When you're done for the day:

1. Go back to AWS Console → SageMaker → Notebook instances
2. Select your instance
3. Click "Stop"
4. **This stops billing.** The instance is ~$1.41/hour when running.
5. Your data is preserved on the volume — when you start it again, everything is still there.

When you're completely done with the project:
- You can "Delete" the instance to stop all charges
- But first make sure everything is pushed to GitHub

---

## Time Estimates

| Step | Time |
|------|------|
| GitHub + Kaggle setup (Parts 1-2) | 30 minutes |
| SageMaker instance creation (Part 3) | 10 minutes |
| Environment setup + dataset download (Part 4) | 30-60 minutes |
| Face extraction | 2-4 hours (can leave running) |
| Model training | 2-3 hours (can leave running) |
| Evaluation + heatmaps | 30-60 minutes |
| Push to GitHub (Part 6) | 15 minutes |
| **Total active time (you at keyboard)** | **~2 hours** |
| **Total wall-clock time** | **~8-12 hours** |

---

## Cost Estimate

| Item | Cost |
|------|------|
| SageMaker ml.g5.xlarge (~10 hours) | ~$14 |
| SageMaker storage (200GB) | ~$0.70/day while stopped |
| GitHub | Free |
| Kaggle | Free |
| HuggingFace | Free |
| **Total** | **~$15-20** |

---

## Troubleshooting

**"CUDA out of memory"**: Reduce batch size in `src/config.py` from 32 to 16.

**"No such file or directory" for kaggle**: Make sure `~/.kaggle/kaggle.json` exists and has correct permissions (chmod 600).

**Git push rejected**: Make sure you're using the Personal Access Token as password, not your GitHub password.

**Notebook kernel dies**: The instance may be running out of RAM during face extraction. Process fewer videos at a time (edit `config.py` to reduce `MAX_VIDEOS_PER_PART`).

**Instance stuck on "Pending"**: Your AWS account may not have quota for g5 instances. Try `ml.g4dn.xlarge` instead (slightly slower but usually available).
