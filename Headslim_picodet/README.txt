picodet_update.zip
Extract ALL files into the same folder as picodet_letterbox.py (overwrite when asked):
  Windows:   right-click the zip > Extract All
  Linux/WSL: unzip -o picodet_update.zip -d ~/Picodet_LIBREYOLO
Then, from that folder:  python verify_update.py   (should end with ALL GOOD)
Defaults are now: activation = leakyrelu, gate = sigmoid (no hard-swish / hard-sigmoid).
Your ch_act_ scripts are NOT in this zip; keep passing --act leakyrelu --gate sigmoid there.
