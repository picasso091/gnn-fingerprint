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
   ```

4. ```bash
   python train.py --arch gcn --epochs 200 --seed 0
   ```

5. ```bash
   python fine_tune_pirate.py --target_path models/target_model.pt --meta_path models/target_meta.json --num_variants 100
   ```

6. ```bash
   python distill_students.py --archs gat,sage --count_per_arch 50 --seed 0
   ```

7. ```bash
   python train_unrelated.py --count 200 --archs gcn,sage,mlp --seed 0
   ```

8. ```bash
   python fingerprint_generator.py
   ```

9. ```bash
   python generate_univerifier_dataset.py --fingerprints_path fingerprints/fingerprints.pt --target_path models/target_model.pt --target_meta models/target_meta.json --positives_glob "models/positives/ftpr_*.pt,models/positives/distill_*.pt" --negatives_glob "models/negatives/negative_*.pt" --out fingerprints/univerifier_dataset.pt
   ```

10. ```bash
    python train_univerifier.py --dataset fingerprints/univerifier_dataset.pt --epochs 200 --lr 1e-3 --val_split 0.2 --fingerprints_path fingerprints/fingerprints.pt --out fingerprints/univerifier.pt
    ```

11. ```bash
    python eval_verifier.py --fingerprints_path fingerprints/fingerprints.pt --verifier_path fingerprints/univerifier.pt --target_path models/target_model.pt --target_meta models/target_meta.json --positives_glob "models/positives/ftpr_*.pt,models/positives/distill_*.pt" --negatives_glob "models/negatives/negative_*.pt" --out_plot plots/cora_gcn_aruc.png --save_csv plots/cora_gcn_curves.csv
    ```