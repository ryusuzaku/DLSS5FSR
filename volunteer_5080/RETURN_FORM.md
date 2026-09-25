# Return note (copy, fill, and send with the results ZIP)

- GPU model: RTX 5080 / other:
- Windows version (optional):
- Is `nvidia-smi` working? Yes / No
- Did the kit produce 4/4 ViT maps? Yes / No (error text is in the ZIP)
- Did the DLL and optional full package come from the same download? Yes / No / Unsure
- Do you have the full package ZIP or `native-game-tiled-assets` folder? Yes / No
- Could you run one follow-up original C256/C32 kernel probe if we provide an exact kit? Yes / No
- Available time before the PC is taken apart:
- Anything else that failed or looked odd:

## Why a follow-up may be needed

The project's remaining original-GPU evidence, in priority order, is:

1. Controlled C256 outputs at block15 and block49, with the exact input,
   weight, cubin, and output hashes plus kernel launch dimensions.
2. Controlled C32 outputs at block67, block69, and post70 under the same
   provenance checks.
3. Original 16-token ViT attention/reduction, native preblock input, and
   final head outputs if a reliable harness can be prepared.

These are **not** requested as ad-hoc runs. This kit only performs the ViT
repack probe and asset inventory. We will supply exact controlled inputs and
a separate, reviewed run recipe before asking you to run items 1–3. Please
tell us now whether a follow-up is possible so we can prioritize your time.

Please return only the generated results ZIP and this note. No DLL, game,
model package, cubin, or screenshot is needed. Results can be sent to the
maintainer privately; post them publicly only if you are comfortable with
the report's filenames, hashes, and system metadata.
