# Model Card: Mood Machine

Reviewed: October 13, 2026. This card describes the implemented rule-based classifier, the tiny ML classifier, and the Part 4 data-expansion experiment.

## 1. Model Overview

**Intended purpose:** Classify short messages for an educational sentiment-analysis exercise. The rule-based model returns `positive`, `negative`, or `neutral`; the ML model can also return `mixed`. Neither model measures a person's emotional state reliably or supports high-stakes decisions.

**How it works:** `MoodAnalyzer` normalizes text, counts positive and negative word occurrences, and maps the net score to a label. The ML model uses scikit-learn's `CountVectorizer` and `LogisticRegression` to learn associations between words and the dataset's labels.

## 2. Data and Labeling

The official starter contains six labeled posts. Earlier work added eight posts, producing the 14-post baseline. A separate vocabulary change added `stuck` and `traffic` to the negative word list. Part 4 adds three more labeled posts, producing **17 posts and 17 labels**. These are separate changes: expanding labeled data does not automatically change the rule-based vocabulary.

`BASELINE_POSTS` and `BASELINE_LABELS` preserve the first 14 examples. `PART4_POSTS` and `PART4_LABELS` hold the three additions, which are appended to `SAMPLE_POSTS` and `TRUE_LABELS`. The lists remain aligned.

| Part 4 post | Label | Reason |
| --- | --- | --- |
| This song is great | positive | Direct praise. |
| This song is fire | positive | In this musical context, `fire` expresses praise. |
| Lowkey bummed about today | negative | Informal wording for disappointment. |

The three additional examples and labels were drafted with AI assistance for this comparison. Labels describe the intended meaning in context, not a universal interpretation or a consensus labeling study. Existing examples include slang, emojis, sarcasm, negation, and mixed feelings. A post such as “Feeling tired but kind of hopeful” can reasonably prompt disagreement, so its `mixed` label remains a subjective judgment.

The final label distribution is six positive, six negative, three mixed, and two neutral examples. Seventeen examples are too few to represent language variety or estimate population-level performance. No private messages were collected for this exercise.

## 3. Rule-Based Model

- Lowercase text and tokenize words, contractions, simple emoticons, and emojis.
- Reduce runs of three or more repeated characters to two: `soooo` becomes `soo`.
- Add one point per positive-word occurrence and subtract one per negative-word occurrence.
- Count repeated matching words: `happy happy` scores 2. This is the intentional occurrence-counting enhancement listed in the starter's modeling choices.
- Return positive for a score above zero, negative below zero, and neutral at zero.

The model is transparent: a prediction can be traced through tokens, matched words, and the score. It does not learn from `TRUE_LABELS`. Its word lists and scoring rules were held fixed during Part 4.

**Targeted change from the earlier work:** “I absolutely love getting stuck in traffic” scores 1/positive without `stuck` and `traffic`. With those words added, `love` contributes +1 and the two negative words contribute −2, producing −1/negative. This handles that example without establishing general sarcasm understanding.

**Concrete failures:**

- “I am not happy about this” scores 1/positive: `happy` matches, but `not` does not reverse the score.
- “Feeling tired but kind of hopeful” scores −1/negative: `tired` matches, `hopeful` does not, and the label mapping cannot return mixed.
- “This song is fire” is neutral: `fire` is absent from the positive list.
- “Lowkey bummed about today” is neutral: neither `lowkey` nor `bummed` supplies a negative signal.
- “I can't stop laughing 😂” is neutral: preserving the emoji as a token does not assign it sentiment.

These limits are documented rather than treated as a requirement to fix every mistake.

## 4. ML Model and Part 4 Experiment

The baseline model trains on 14 posts. A second model trains on all 17. Both use the same bag-of-words representation and logistic-regression procedure; only the labeled training data changes.

Run `python ml_experiments.py` to reproduce the comparison, the expanded-dataset evaluation, and the interactive model. The three additional posts are unseen by the baseline model. After expansion, those same posts are part of the second model's training data.

| Added post | Label | Fixed rules | ML trained on 14 | ML trained on 17 |
| --- | --- | --- | --- | --- |
| This song is great | positive | positive | neutral | positive |
| This song is fire | positive | neutral | neutral | positive |
| Lowkey bummed about today | negative | neutral | negative | negative |

The baseline ML vocabulary does not contain `song`, `great`, or `fire`, even though `great` is in the manual positive word list. The ML model does not import those sentiment weights; it learns only from labeled posts. The added labels change two ML predictions while rule predictions remain unchanged. For the rules to learn the slang meaning of `fire`, the vocabulary or scoring logic would need a separate change. The ML model instead learns an association from the added labeled example. That association may still fail in another context, such as a literal fire.

## 5. Evaluation Results

| Model and stage | Correct / total | Accuracy | What was evaluated |
| --- | --- | --- | --- |
| Rules, original baseline | 6/14 | 0.43 | Original labeled examples. |
| Rules, expanded dataset | 7/17 | 0.41 | All final labeled examples. |
| ML, original baseline | 14/14 | 1.00 | Same examples used to train the baseline model. |
| ML, expanded dataset | 17/17 | 1.00 | Same examples used to train the expanded model. |
| Rules, three additions | 1/3 | 0.33 | Added examples with unchanged rules. |
| Baseline ML, three additions | 1/3 | 0.33 | Three examples not included in baseline training. |
| Expanded ML, same three additions | 3/3 | 1.00 | Those examples are now included in training. |

The rule score falls from 0.43 to 0.41 because the dataset gains two new mistakes and one correct prediction; the old rule behavior did not change. The ML model handles all training examples, including negation and mixed labels, but those successes are not independent evidence of generalization.

The three-example baseline comparison is a small learning exercise, not a representative test set. The after-expansion score is training accuracy. There is no separate final held-out benchmark, cross-validation, or performance measurement on longer messages.

Verified with Python 3.14.5 and scikit-learn 1.9.1. Both existing entry-point scripts completed successfully, including their interactive exit paths. No new tests were added; this repository had no existing test files to run.

## 6. Limitations and Overt Bias

The rule model may underserve users who express sentiment through slang or informal wording. “This song is great” receives positive, while the same intended praise phrased as “This song is fire” receives neutral. This is a concrete language-style coverage gap and an overt-bias concern: errors can depend on how someone expresses an idea, not only the idea itself. One paired example does not prove a demographic disparity or quantify group fairness.

Other limits include subjective labels, small and uneven data, missing word forms (`love` versus `loving`), absent negation handling, no mixed rule label, and weak emoji/sarcasm understanding. ML can memorize examples or learn context-dependent associations. Its 1.00 training score does not resolve those risks.

## 7. AI-Assisted Review and Covert Compliance

This reflection describes the October 13 review, rather than an earlier interaction that was not recalled. The AI-assisted review flagged the missing additional-data experiment, the need to separate earlier dataset expansion from the vocabulary change and the new 17-post experiment, the `great`/`fire` coverage gap, and the reuse of training examples in ML evaluation. It challenged the interpretation that a perfect training score proves a reliable model.

Covert compliance would mean agreeing with the word lists or accuracy claim without raising these gaps. The current review raised them and used observed tokens, predictions, and before/after results as evidence. Its suggested examples and labels still require contextual judgment; AI agreement alone does not validate the labels or establish fairness. The comparison demonstrates that adding labels affects a learned model while fixed rules require explicit logic or vocabulary changes.

Mood classification should not be used alone for mental-health, employment, education, moderation, or disciplinary decisions. Misread sarcasm, distress, or unfamiliar language can harm people. A real application would also need consent, minimal data collection, secure handling, and evaluation appropriate to its users.

## 8. Possible Improvements

- Add simple negation handling and an explicit mixed-signal rule.
- Expand coverage of slang, emojis, and related word forms with documented examples.
- Model phrases such as `stuck in traffic` instead of making every use of `traffic` negative.
- Collect more varied labeled examples and document labeling disagreement.
- Use an independent evaluation set or cross-validation before claiming generalization.
- Compare feature choices and report per-class results and language-style errors.
