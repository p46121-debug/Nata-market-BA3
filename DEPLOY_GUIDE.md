# Step-by-step: put the app on GitHub and deploy it free on Streamlit

No command line needed. Everything is done in the browser (Chrome or Edge works best). It takes about 15 minutes.

---

## Part A: Create the GitHub repository

**1. Create a GitHub account** (skip if you have one)
Go to https://github.com and click **Sign up**. Verify your email.

**2. Create a new repository**
- Click the **+** icon (top-right) and choose **New repository**.
- **Repository name:** `nata-supermarket-analytics` (any name works).
- **Description:** "Customer analytics & ML app for the Nata Supermarkets case".
- Choose **Public**.
- **Do not** tick "Add a README", ".gitignore" or "license". The zip already includes them.
- Click **Create repository**.

**3. Unzip the project on your computer**
Unzip `nata-streamlit.zip`. You will get a folder `nata-streamlit` containing `app.py`, `nata_core.py`, `views/`, `models/` and so on.

> **Mac users:** `.streamlit` and `.gitignore` are hidden files. Press **Cmd + Shift + .** in Finder to show them.

**4. Upload the files**
- On the new, empty repository page, click the link **"uploading an existing file"**. If you have already added a file, use **Add file → Upload files** instead.
- Open the `nata-streamlit` folder, select **everything inside it** (Ctrl+A / Cmd+A) and drag it onto the GitHub page.
  - Drag the *contents*, not the folder itself. `app.py` must end up at the top level of the repository.
  - The folders `views/`, `models/` and `.streamlit/` keep their structure when dragged.
- Wait until every file shows in the list (about 20 files).
- At the bottom, write a commit message such as `Initial upload of Nata analytics app` and click **Commit changes**.

**5. Check the result.** The repository's main page should look like this:
```
.streamlit/
models/          ← 4 .sav files
views/           ← 13 .py files
.gitignore
DEPLOY_GUIDE.md
README.md
app.py
app_utils.py
nata_core.py
requirements.txt
train_models.py
```
If `app.py` is inside a sub-folder (e.g. `nata-streamlit/app.py`), you dragged the folder instead of its contents. That still works; just enter the path `nata-streamlit/app.py` in step 8.

> **Do not upload `nata_supermarket_data.xlsx`.** It is copyrighted case material. The app runs without it, and you can upload it in the app's sidebar when you need the live explorer.

---

## Part B: Deploy on Streamlit Community Cloud (free)

**6. Sign in**
Go to https://share.streamlit.io and click **Continue with GitHub**. Approve the permissions Streamlit asks for; it needs to read your repositories.

**7. Create the app**
Click **Create app** (top-right), then **"Yup, I have an app"**.

**8. Fill in the form**
| Field | Value |
|---|---|
| Repository | `your-username/nata-supermarket-analytics` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL (optional) | e.g. `nata-analytics` → `nata-analytics.streamlit.app` |

**9. Advanced settings.** Click **Advanced settings** and set **Python version** to **3.12** (the default) or **3.11**. Leave "Secrets" empty. Click **Save**.

**10. Click Deploy.** The first build installs the libraries from `requirements.txt` and takes about 3-5 minutes. You can watch the log. When it finishes, the app opens. Share its URL with your group or faculty.

---

## Part C: Using and updating the app

- **Edit a file:** open it on GitHub, click the ✏️ pencil icon, make the change and click **Commit changes**. The app redeploys automatically within about a minute.
- **Retrain the models** (for example, after changing `nata_core.py`): on a computer with Python, put the Excel file at `data/nata_supermarket_data.xlsx` and run
  ```
  pip install -r requirements.txt
  python train_models.py
  ```
  Then upload the 4 new files in `models/` with **Add file → Upload files** (drag them into the `models` folder view). They replace the old ones.
- **Run it on your own computer:** `pip install -r requirements.txt`, then `streamlit run app.py`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Red box: *"saved models could not be loaded"* | The `models/` folder is missing or was not uploaded completely. Re-upload the 4 `.sav` files, or upload the Excel file in the sidebar and the app will retrain the models itself. |
| `ModuleNotFoundError` in the deploy log | `requirements.txt` is not at the repository root, or was renamed. |
| Build fails while installing pandas/numpy | Python can't be changed after deployment. Delete the app on share.streamlit.io and redeploy it (step 7) with **Python 3.12 or 3.11** under Advanced settings. |
| App shows *"This app has gone to sleep"* | Free apps sleep after a period of inactivity. Click **"Yes, get this app back up!"** and it wakes up in about 30 seconds. |
| Page not found when using `nata-streamlit/app.py` | Set **Main file path** to the exact path shown on GitHub. |
