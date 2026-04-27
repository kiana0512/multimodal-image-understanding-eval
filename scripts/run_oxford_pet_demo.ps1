$ErrorActionPreference = "Stop"
Write-Host "Preparing Oxford-IIIT Pet small demo..."
python scripts/prepare_oxford_pet.py --max-samples 100 --make-pseudo-masks
python scripts/run_pipeline.py --task oxford_pet_segmentation
Write-Host "Oxford Pet demo finished. Check outputs/segmentation_eval and outputs/reports."
