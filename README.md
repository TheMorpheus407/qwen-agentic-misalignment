# Qwen agentic-misalignment raw outputs

Raw model outputs and inputs from a local replication of the fictional email
blackmail scenario. The source corpus comprises **802 main runs and 40 technical
pilot runs**; this release includes **198 main responses and 13 pilot responses**
after the publication email-domain filter described below. Related video:
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

- `runs/main/models/`: 198 JSON response records, grouped by model, condition
  and sample. `runs/pilot/models/` contains 13 pilot records separately.
  Sample identifiers retain their original numbering.
- `runs/{main,pilot}/prompts/`: exact system and user prompts for each of four
  conditions (16 files). The eight email-context files are withheld by the
  publication filter. Main and pilot originally used the same inputs.
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
`rootless-v1`. This release retains 17 `legacy-user-session` records and 194
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

## Bewusst zurückgehalten

**639 source files are withheld**: 604 main responses, 27 pilot responses and
eight email-context prompt files. These files contain email addresses at real
mail providers or other domains outside the fictional-company and reserved
example-domain exceptions. A fictional person name at a real provider is still
withheld under this release decision. JSON newline escape fragments are
recognized as scanner artifacts; the underlying decoded address is also checked.

Retained source files are byte-for-byte unchanged. Entire affected files are
omitted; generated text is never redacted. The released subset is incomplete
and cannot by itself reproduce the original experiment. The original run
configurations describe the full source corpus. No judgments or analyses are
included.
