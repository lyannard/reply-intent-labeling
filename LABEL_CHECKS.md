# How the labels were checked

`labels.csv` has one row for each of the 840 replies in the Q3 2026 Outbound Email Reply Export, with exactly five labels.

## From ten labels to five

| Five-label scheme | Folded from | Replies |
|---|---|---|
| `ready_to_buy` | Ready to buy | 66 |
| `evaluating` | Evaluating | 198 |
| `not_now` | Interested later, No need (free is enough), Auto-reply | 271 |
| `not_interested` | Not interested, Opt-out, Lost to competitor | 174 |
| `wrong_contact` | Wrong contact, Student/individual | 131 |

Three groups were judgment calls:

- **Auto-replies are `not_now`.** They show no intent either way, so "try again later" fits best. The rationale starts with "Auto-reply" so they can be filtered out.
- **Opt-outs are `not_interested`.** The rationale says "opt-out; suppress", so the suppression signal survives the fold.
- **Students are `wrong_contact`.** They are individuals asking about personal pricing, not team buyers. All 19 come from university or K-12 workspaces.

## Checks run on `labels.csv`

- 840 rows, unique `reply_id`, same IDs and order as the export, and `account_id` matches the export on every row.
- Only the five allowed labels, no blanks, confidence between 0 and 1.
- The 840 replies are 38 templated messages with only numbers changed. Each template gets one label, and no template matched two rules.
- Every label equals the earlier ten-label file after folding (0 mismatches).
- All 69 replies with an auto-reply subject line are `not_now`, and all 19 student rows and 89 opt-outs are flagged in the rationale.

## What was not checked

- **The script has not been run against OpenRouter.** The environment that built this repo could not reach openrouter.ai. The labels in this file come from the template mapping above, with `confidence` set by hand per template (0.95+ for clear cases, 0.80 to 0.90 for the judgment calls). The script itself was tested with a mocked API that returned rate-limit errors, non-JSON, an invalid label and out-of-range confidence, and it recovered each time.
- **Do this on the first real run:** `python label_replies.py` backs up this file as `labels_reference.csv`, writes the model's labels to `labels.csv`, and prints agreement and every disagreement. Review those before trusting the model's labels. A model's self-reported confidence is not calibrated, so don't use it as a threshold without checking it.

## Running it

Put the export next to the script and a `.env` file in this folder containing one line, `OPENROUTER_API_KEY=<your key>`. `.env` is listed in `.gitignore` and must never be committed. The free tier allows 20 requests a minute and 50 a day, so the script makes one call per distinct text (38 calls), not 840. For repeatable labels, pin one specific `:free` model with `--model`, because the default free router picks a model at random.
