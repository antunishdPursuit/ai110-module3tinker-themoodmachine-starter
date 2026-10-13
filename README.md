# The Mood Machine

A Week 5 AI110 Tinker comparing a transparent rule-based sentiment classifier with a small logistic-regression model. The dataset includes slang, emojis, sarcasm, and mixed emotions; the [model card](model_card.md) explains the observed results and limitations.

## Files

- `dataset.py`: the 14-example baseline, three Part 4 additions, word lists, and final 17-example dataset.
- `mood_analyzer.py`: preprocessing, occurrence-based scoring, label prediction, and explanations.
- `main.py`: rule-based evaluation, batch predictions, and an interactive loop.
- `ml_experiments.py`: the reproducible 14-versus-17 data experiment, ML training evaluation, and an interactive loop.
- `model_card.md`: data provenance, results, failures, language-style coverage, and AI-review reflection.
- `requirements.txt`: the existing dependencies.

## Run on Windows

From this folder, create an environment if one does not already exist:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the existing scripts with UTF-8 enabled so the emoji examples print correctly:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B main.py
.\.venv\Scripts\python.exe -X utf8 -B ml_experiments.py
```

Type `quit` or press Enter on an empty line to exit either interactive loop. Directly invoking the environment's Python does not require changing PowerShell's activation policy.

## Part 4: What Adding Data Changes

The ML script first trains on the baseline 14 labeled posts and predicts the three additions. It then retrains on all 17 and compares predictions on those same three posts. The rules and word lists stay fixed.

- Rules: 6/14 on the baseline; 7/17 on the expanded dataset.
- ML: 1/3 correct on the three additions before training on them; 3/3 afterward, when they are now training examples.
- Expanded ML training accuracy: 17/17.

The two music-praise posts change from neutral to positive in ML after expansion. Fixed-rule predictions do not change. The 3/3 and 17/17 results reuse training examples and do not establish performance on new text. There is no target accuracy requirement to chase.

Both scripts and dataset alignment were verified. No new tests were added, and the repository contains no existing test files. See the model card for specific misclassifications and the limits of this small experiment.

[Current Week 5 Tinker instructions](https://courses.codepath.org/courses/ai110/unit/5#!tinker)
