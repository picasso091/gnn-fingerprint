Fingerprinting Graph Neural Networks

Steps

1. Create virtual env. Activate it.
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Create folders

   ```bash
   mkdir models
   mkdir fingerprints
   mkdir plots
   mkdir lp
   mkdir gc
   ```

### For node classification task on Cora dataset on GCN arch

```bash
python train.py --arch gcn --epochs 200 --seed 0
```

```bash
python fine_tune_pirate.py --target_path models/target_model.pt --meta_path models/target_meta.json --num_variants 100
```

```bash
python distill_students.py --archs gat,sage --count_per_arch 50 --seed 0
```

```bash
python train_unrelated.py --count 200 --archs gcn,sage,mlp --seed 0
```

```bash
python fingerprint_generator.py
```

```bash
python generate_univerifier_dataset.py --fingerprints_path fingerprints/fingerprints.pt --target_path models/target_model.pt --target_meta models/target_meta.json --positives_glob "models/positives/ftpr_*.pt,models/positives/distill_*.pt" --negatives_glob "models/negatives/negative_*.pt" --out fingerprints/univerifier_dataset.pt
```

```bash
python train_univerifier.py --dataset fingerprints/univerifier_dataset.pt --epochs 200 --lr 1e-3 --val_split 0.2 --fingerprints_path fingerprints/fingerprints.pt --out fingerprints/univerifier.pt
```

```bash
python eval_verifier.py --fingerprints_path fingerprints/fingerprints.pt --verifier_path fingerprints/univerifier.pt --target_path models/target_model.pt --target_meta models/target_meta.json --positives_glob "models/positives/ftpr_*.pt,models/positives/distill_*.pt" --negatives_glob "models/negatives/negative_*.pt" --out_plot plots/cora_gcn_aruc.png --save_csv plots/cora_gcn_curves.csv
```

### For link prediction task on Citeseer dataset on GCN arch

```bash
python lp/train_lp_citeseer.py
   --model gcn
   --out fingerprints/lp/lp_citeseer.pt
```

```bash
python lp/make_lp_pos_neg.py
   --mode finetune
   --target-ckpt fingerprints/lp/lp_citeseer.pt
   --arch gcn
   --num-models 50
   --epochs 10
   --seed 10
   --out models/positives/lp

python lp/make_lp_pos_neg.py
  --mode finetune
  --target-ckpt fingerprints/lp/lp_citeseer.pt
  --arch gcn
  --num-models 50
  --epochs 10
  --seed 70
  --out models/positives/lp

python lp/make_lp_pos_neg.py
  --mode partial
  --target-ckpt fingerprints/lp/lp_citeseer.pt
  --arch gcn
  --num-models 50
  --epochs 10
  --seed 70
  --out models/positives/lp

python lp/make_lp_pos_neg.py
  --mode distill
  --target-ckpt fingerprints/lp/lp_citeseer.pt
  --student-arch sage
  --num-models 50
  --epochs 10
  --seed 130
  --out models/positives/lp


python lp/make_lp_pos_neg.py
  --mode scratch_neg
  --arch gcn
  --num-models 100
  --epochs 50
  --seed 10
  --out models/negatives/lp

python lp/make_lp_pos_neg.py
  --mode scratch_neg
  --arch sage
  --num-models 100
  --epochs 50
  --seed 110
  --out models/negatives/lp

```

```bash
python lp/save_target_encoder.py
```

```bash
python lp/fingerprint_generator_lp.py \
  --target-module train_lp_citeseer \
  --target-class GCNEncoder \
  --target-ckpt fingerprints/lp/enc_citeseer.pt \
  --target-kwargs '{"in_dim": 3703, "hid": 64, "out": 64, "num_layers": 3}' \
  --pos-glob "models/positives/lp/*.pt" \
  --neg-glob "models/negatives/lp/*.pt" \
  --n-fps 64 --n-nodes 32 --feat-dim 3703 --iters 100 --lr 1e-3 \
  --m-pairs 64 \
  --out fingerprints/lp/fingerprints_lp_citeseer.npz \
  --meta fingerprints/lp/fingerprints_lp_citeseer_meta.json
```

```bash
python lp/generate_univerifier_dataset_lp.py \
--fingerprints fingerprints/lp/fingerprints_lp_citeseer.npz \
--pos-glob "models/positives/lp/*.pt" \
--neg-glob "models/negatives/lp/*.pt" \
--target-ckpt fingerprints/lp/enc_citeseer.pt \
--out fingerprints/lp/univerifier_lp_citeseer.pt
```

```bash
python lp/train_univerifier_lp.py \
--data fingerprints/lp/univerifier_lp_citeseer.pt \
--epochs 100 \
--lr 1e-3 \
--val-split 0.5 \
--seed 0 \
--out fingerprints/lp/univerifier_lp.pt
```

```bash
python lp/eval_verifier_lp.py \
  --data fingerprints/lp/univerifier_lp_citeseer.pt \
  --verifier fingerprints/lp/univerifier_lp.pt \
  --test-size 0.5 \
  --seed 0 \
  --out-png plots/citeseer_gcn_aruc.png \
  --out-csv plots/citeseer_gcn_aruc.csv
```

### For graph classification task on ENZYMES dataset on GraphSage arch

```bash
python gc/train_gc.py --ckpt_dir fingerprints/gc --ckpt_name target_graphsage.pt
```

```bash
python gc/fine_tune_pirate_gc.py \
  --target_ckpt fingerprints/gc/target_graphsage.pt \
  --num_models 100
```

```bash
python gc/distill_students_gc.py  --target_ckpt fingerprints/gc/target_graphsage.pt --num_models 100
```

```bash
 python gc/train_unrelated_gc.py --num_models 200
```

```bash
python gc/fingerprint_generator_gc.py  --target_ckpt fingerprints/gc/target_graphsage.pt   --pos_dir models/positives/gc   --neg_dir models/negatives/gc   --save_dir fingerprints/gc  --save_name fingerprints_gc_enzyme.pt
```

```bash
python gc/fingerprint_generator_gc.py  --target_ckpt fingerprints/gc/target_graphsage.pt
```

```bash
python gc/train_univerifier_gc.py \
  --fingerprints_path fingerprints/gc/fingerprints_gc_enzyme.pt \
  --save_dataset
```

```bash
python gc/eval_verifier_gc.py \
  --fingerprints_path fingerprints/gc/fingerprints_gc_enzyme.pt \
  --verifier_ckpt fingerprints/gc/univerifier_gc_enzyme.pt
```
