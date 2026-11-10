# Run VQC Depth on Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/EduardoTBuss/vqc-depth/blob/main/notebooks/ERAMIA_launcher.ipynb).
Choose a CPU runtime, then run the cells in order. Authorize Google Drive when
prompted. The notebook clones the public GitHub repository over HTTPS.

1. **Drive.** Mounts Drive and prepares `MyDrive/ERAMIA-VQC/` for the main grid.
2. **Repository.** Clones a fresh copy and prints the commit used for the session.
3. **Dependencies.** Installs the Python package and its dependencies. If Colab
   has a stale library loaded after pip changes its version, restart the runtime
   and run the notebook again. Completed runs stay on Drive.
4. **Tests.** Runs the test suite with UCI downloads enabled. Resolve a failing
   test before continuing.
5. **Pilot.** Runs two five-epoch Banknote/VQC-4 jobs in a fresh directory
   under `MyDrive/ERAMIA-VQC/pilots/` and estimates the full grid's runtime.
6. **Main grid.** Runs all 120 configurations and logs progress to Drive.
   After a disconnect, rerun the setup and training cells to resume.
7. **Analysis.** Rebuilds CSVs and writes tables and figures. The analysis
   also works with a partially completed grid.

Each completed run has a JSON record in `runs/`. Retrying skips completed
records and restarts an interrupted run from its first epoch. For parallel
training, give each runtime a disjoint dataset or seed subset and run the
analysis in one runtime once training has finished.

See the [README](README.md) for local commands and the [paper directory](paper/)
for results from the original study.
