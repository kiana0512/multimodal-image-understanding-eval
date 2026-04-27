# Local Game AIGC Asset Template

Use this folder as a schema reference for connecting outputs from ComfyUI, Stable Diffusion, or a separate `game-aigc-asset-workflow` repository.

Recommended workflow:

```bash
python scripts/prepare_local_game_assets.py --image-root D:/your_comfyui_outputs --output data/processed/game_assets/game_asset_manifest.csv
```

If you have prompt metadata, provide `--prompts-csv prompts.csv`. The CSV can contain `filename` or `image_path`, plus `prompt`, `negative_prompt`, `seed`, `model_name`, `style_tag`, and `asset_type`.

Do not commit real generated asset images unless you own the rights and intentionally want to publish them.
