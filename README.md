# GridTalk

Talk to your spreadsheet without data ever leaving your PC.

GridTalk is a lightweight, local-first evolution of [ExcelGRAG](https://github.com/sjvrensburg/ExcelGRAG), built on the [Gridcraft](https://github.com/storytold/gridcraft) spreadsheet engine. Open a CSV/TSV/XLSX/ODS workbook, ask it questions in plain language, and get answers grounded in specific cells — with inference running on your own machine by default.

## Privacy model

GridTalk has exactly two trust postures. There is no redaction layer in GridTalk itself:

- **Local backend (default):** a GGUF model runs on your PC. Nothing leaves the machine.
- **External backend (opt-in):** GridTalk sends the rendered passage, including cell values, verbatim to the OpenAI-compatible endpoint you configure. If that is unacceptable, stay local, or point GridTalk at an external redacting proxy (a separate, future project) — GridTalk needs no changes to work through such a proxy.

## Status

Working local-first prototype: CSV/TSV/XLSX/ODS ingestion (via the
vendored Gridcraft engine), a lifted property graph with cited
answers, hybrid lexical+embedding search, an investigate tool
harness, a dual-view desktop UI, and real local GGUF generation on
candle for llama-family models (feature `candle-gen`). With no
model configured, chat uses a deterministic echo backend so the
pipeline is demoable offline — it does not generate prose. See
`docs/plan.md` for the phased plan (with honest checkboxes) and
`docs/decisions/` for architecture records.

## Licence

MIT.
