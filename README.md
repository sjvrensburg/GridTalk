# GridTalk

Talk to your spreadsheet without data ever leaving your PC.

GridTalk is a lightweight, local-first evolution of [ExcelGRAG](https://github.com/sjvrensburg/ExcelGRAG), built on the [Gridcraft](https://github.com/storytold/gridcraft) spreadsheet engine. Open a CSV/TSV/XLSX/ODS workbook, ask it questions in plain language, and get answers grounded in specific cells — with inference running on your own machine by default.

## Privacy model

GridTalk has exactly two trust postures. There is no redaction layer in GridTalk itself:

- **Local backend (default):** a GGUF model runs on your PC. Nothing leaves the machine.
- **External backend (opt-in):** GridTalk sends the rendered passage, including cell values, verbatim to the OpenAI-compatible endpoint you configure. If that is unacceptable, stay local, or point GridTalk at an external redacting proxy (a separate, future project) — GridTalk needs no changes to work through such a proxy.

## Status

Phase 1 scaffolding. See `docs/plan.md` for the phased implementation plan and `docs/decisions/` for architecture decision records.

## Licence

MIT.
