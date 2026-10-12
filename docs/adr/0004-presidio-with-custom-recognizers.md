# 4. Presidio with custom recognizers, protected identifiers and longest-first overlap resolution

Status: accepted

## Context
Spiking Presidio's defaults on realistic text showed gaps that matter for a scrub stage:
- Street addresses and "born on" dates are not detected.
- Account ids, order numbers and ticket ids are labelled as people or places.
- A bare phone number scores 0.4, below the 0.5 default threshold, and is only lifted by words such as "phone" or "number".
- Overlapping detections of different types can leave an address half redacted.

## Decision
Keep Presidio as the only engine. Add custom recognizers for card last four digits, account ids, obfuscated emails, dates of birth and street addresses. Discard detections that overlap protected identifiers (order numbers, tracking numbers, SKUs, ticket ids and existing placeholders). Resolve overlaps longest first. Replace each distinct value with a typed, numbered placeholder. After redaction, analyse the output again and quarantine any trace with a remaining detection.

## Consequences
- Recall and precision are measured per type against planted spans, and a span counts as caught only when detections cover at least 80% of it, because a half-redacted address is still a leak.
- Known weaknesses, all visible in the report: phone numbers without a context word, product names that look like people's names, and PII phrasings the recognizers do not cover.
- The detection threshold is the lever for the phone gap. The full evaluation at 0.5 caught 64% of planted phone numbers; at 0.4 it caught all of them, phone precision moved from 0.997 to 0.992, and overall recall rose from 0.905 to 0.979. The default is therefore 0.4. Both evaluations are committed so the trade-off is checkable.
- Not built: a catalogue allow-list for product names. It would remove the largest source of false positives but needs a real product catalogue.
- Alternative detectors were considered out of scope: the point is a measured baseline with Presidio.
