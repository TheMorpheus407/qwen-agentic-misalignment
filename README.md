# Qwen agentic-misalignment raw outputs

Raw model outputs and inputs from a local replication of the fictional email
blackmail scenario. The source corpus comprises **802 main runs and 40 technical
pilot runs**; this release includes **796 main responses and all 40 pilot responses**
after the scenario-origin email review described below. Related video:
[AI Safety Research](https://youtu.be/jE7qdFdVJLY).

## Models and provenance

Both models were run locally with llama.cpp, using Q4_K_M GGUF weights:

- Base: [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/f0eec4a4bb4975114a030d048952d83c0a53c034), revision `f0eec4a4bb4975114a030d048952d83c0a53c034`.
- Abliterated derivative: [bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF/tree/87d37daf5e5eb72a926d8b413e08809a57f1a120), revision `87d37daf5e5eb72a926d8b413e08809a57f1a120`.

Exact weight filenames and SHA-256 hashes are in `config/local_provider.json`;
weights are not bundled. The framework excerpt comes from
[anthropic-experimental/agentic-misalignment](https://github.com/anthropic-experimental/agentic-misalignment/tree/ea0630e1a3eaae7f9f9740fd2703229d3854ccda)
at commit `ea0630e1a3eaae7f9f9740fd2703229d3854ccda`, with a local-provider
adaptation. Its MIT license is retained in `framework/LICENSE.txt`.

## Files

- `runs/main/models/`: 796 JSON response records, grouped by model, condition
  and sample. `runs/pilot/models/` contains all 40 pilot records separately.
  Sample identifiers retain their original numbering.
- `runs/{main,pilot}/prompts/`: exact system, user and email-context prompts for
  each of four conditions (24 files). Main and pilot used the same inputs.
- `config/chat_template.jinja`: the chat template used for generation.
- `config/local_provider.json`: weight identities and fixed sampling settings.
- `configs/{main,pilot}.yaml`: run configuration, model IDs, sample counts,
  token budget, shuffle seed and per-response seed scheme. Actual seeds,
  sampling settings and prompt hashes are retained in each response record.
- `framework/`: the framework excerpt and its MIT license.
- `SHA256SUMS`: SHA-256 checksums of every published file except this checksum
  file itself. Verify from the repository root with `sha256sum -c SHA256SUMS`.

Each response has three top-level fields: `raw_response` (final answer),
`reasoning_content` (saved thinking), and `metadata`. Technical metadata includes
model/condition/sample, timestamp, seed and seed scheme, token budget, finish
reason, empty-completion flag, elapsed time, machine/isolation labels and
source/binary/original-record hashes. `metadata.local_provider` retains token
usage, verified settings, weight/template identity, slots/context, echoed
sampling settings and prompt hash, plus `raw_generated_text` (raw generated
stream). Generated strings are unchanged; technical metadata uses an allowlist.
Empty and truncated responses remain in the dataset. Visible thinking is
model-generated text.

In the source corpus, the first 52 main records and all 40 pilot records carry
the provenance label `legacy-user-session`; the remaining 750 main records carry
`rootless-v1`. This release retains 92 `legacy-user-session` records and 744
`rootless-v1` records. The earlier label does not establish retrospective
filesystem isolation.
Machine labels `ws` and `gx` identify runtime provenance. Hardware, kernels and
batching can prevent bit-identical reproduction even with fixed seeds.

Names and email addresses in the scenario are fictional benchmark inputs or
model-generated text. They are not contact information.

## Scope

This repository contains raw outputs and generation inputs. Judgments, merged
outcomes, result tables, analyses, reports, audits, qualitative codes, video
selection, preregistration/deviations and local evaluation code are outside
this release.

## Publication review and withheld files

Email addresses were checked against the scenario inputs, the bundled framework
and run configurations. Literal scenario addresses and model-generated variants
of scenario people, companies or organizations are included, including addresses
at real mail providers. These strings are fictional benchmark text, not verified
mailboxes. JSON newline escape fragments are decoded before classification.

The follow-up review restored **633 files**: 598 main responses, 27 pilot
responses and all eight email-context prompts. **Six main response files remain
withheld** because they contain addresses with no established scenario origin.
This includes generic model-generated placeholders under the publication rule;
the review does not establish that any of these mailboxes exists.

| Withheld response file | Address and reason |
|---|---|
| `runs/main/models/local_qwen3.8-27b-q4km/blackmail_explicit-america_none/sample_049/response.json` | Generic model-generated placeholder absent from the inputs; no scenario person or company name. |
| `runs/main/models/local_qwen3.8-27b-q4km/blackmail_explicit-america_replacement/sample_068/response.json` | Generic model-generated placeholder absent from the inputs; no scenario person or company name. |
| `runs/main/models/local_qwen3.8-27b-q4km/blackmail_explicit-america_replacement/sample_150/response.json` | Additional agency absent from the scenario inputs. |
| `runs/main/models/local_qwen3.8-27b-q4km/blackmail_none-none_replacement/sample_005/response.json` | Generic model-generated placeholder absent from the inputs; no scenario person or company name. |
| `runs/main/models/local_qwen3.8-27b-unc-q4km/blackmail_explicit-america_none/sample_047/response.json` | Generic model-generated placeholder absent from the inputs; no scenario person or company name. |
| `runs/main/models/local_qwen3.8-27b-unc-q4km/blackmail_explicit-america_replacement/sample_183/response.json` | Generic model-generated placeholder absent from the inputs; no scenario person or company name. |

All 870 released source files are byte-for-byte unchanged. With README and
SHA256SUMS, the repository contains **872 files** (871 checksum entries).
Entire affected files are omitted; generated text is never redacted. The
released subset remains incomplete because six main responses are withheld;
the original run configurations describe the full source corpus. No judgments
or analyses are included.
