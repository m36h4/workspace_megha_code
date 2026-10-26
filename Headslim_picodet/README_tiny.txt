Tiny PicoDet (strides 8/16, scaled ESNet, fewer stride-16 blocks, narrow neck, kernel 3)
LeakyReLU + sigmoid gate. Input 320x480. From scratch (no warm start possible).

Presets (ONNX GFLOPs incl. head, vs current small-stride neck64 = 0.56):
  t0 scale .5   s16=4 neck48 : ~0.29   (-49%)
  t1 scale .375 s16=3 neck48 : ~0.22-0.25 (-60%)   <- recommended first try
  t2 scale .375 s16=2 neck32 : ~0.17-0.19 (-69%)
  t3 scale .25  s16=2 neck32 : ~0.13   (-76%)

Train:   python picodet_esnet_tiny.py --data ds.yaml --preset t1 --epochs 300 --batch 64 --lr0 0.1 --device 0
Export:  python export_onnx.py --weights best.pt --arch esnet_tiny --tiny-preset t1 --act leakyrelu --gate sigmoid --imgsz 320 480 --head-style raw --out tiny_t1_raw.onnx
Eval:    picodet_eval_onnx.py decodes the raw 2-output file automatically.
