# models/

## `license_plate_detector.pt`

YOLOv8n fine-tuned to detect a single class, `license_plate`. Used by
`src/anpr.py` — it runs on each vehicle crop the main detector finds, then
EasyOCR reads the characters.

- **Source:** https://github.com/anindya-mukhopadhyay/ANPR (`model/license_plate_detector.pt`)
- **Licence:** MIT © 2023 BAPPY AHMED — redistributed here under the same terms.
- ~6 MB. Committed to the repo (unlike the other `*.pt`, which Ultralytics
  downloads on demand) because it isn't available from any model registry.

To disable ANPR entirely, set `anpr.enabled: false` in `config.yaml` — this
file is then unused.
