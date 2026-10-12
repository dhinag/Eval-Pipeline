# 6. Measuring review quality with a single reviewer

Status: accepted

## Context
Inter-rater agreement needs two people. This project has one reviewer, and the review step is the main cost of building a golden set, so it still has to be measured.

## Decision
- Measure agreement between the reviewer and the LLM labels (accept versus correct, and label match).
- Queue a seeded 10% of reviewed records a second time. The re-review is blind: the page never shows an earlier decision. Report the reviewer's consistency with themselves.
- Log seconds spent per record, giving throughput, and report cost per golden example from an hourly rate.
- Review priority favours records the model could not label or was unsure about, suspected failures, rare shapes and representatives of large clusters, so scarce reviewer time goes where it changes the golden set most.

## Consequences
- The numbers are honest about their limits: one reviewer's consistency is not a population estimate.
- The same decisions later calibrate the LLM judge in project 2.
- Scaling to several reviewers needs a different store and a disagreement-resolution step, which is out of scope here.
