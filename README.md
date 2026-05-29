\# AutoPrep-Uplift



LLM-augmented pipeline for uplift modeling in marketing A/B tests.



\## Quick start



```powershell

git clone https://github.com/allisksks/autoprep-uplift.git

cd autoprep-uplift

py -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

py -m pip install -r requirements.txt

copy .env.example .env

jupyter notebook experiments/00\_magnit\_baseline.ipynb

```



\## Branch strategy



\- `main` — stable releases only

\- `dev` — integration branch

\- `feature/\*` — one branch per task, deleted after merge



\## Structure

uplift/          # core pipeline

experiments/     # notebooks per dataset

docs/            # GitHub Pages site



\## Status



Work in progress. Paper coming soon.

