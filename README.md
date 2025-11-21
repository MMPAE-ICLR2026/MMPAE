# MMPAE
This is the official implementation code for MMPAE
----

To reproduce the result, please following the procedure.

#### Environment configuration.
- download the polyone dataset from [here]([https://zenodo.org/records/11246593](https://zenodo.org/records/7766806)) and place the files in proper location.
- download the polybert and transpolymer checkpoint from [here](https://zenodo.org/records/17665048) and place the ckeckpoint in proper location
- 'PolyBert.pt, Property_Transformer.pt' locate in '/ckpt' and extract them
- run polyone_token_extract.py to preprocess the polyOne dataset
- create conda env using MMPAE.yaml

#### Train MMPAE
- python train_MMPAE.py --model_size large --loss_type CwA --temp 0.2 --beta 1000 --alpha 100 ---epochs 500 --batch_size 512 --interval 10 --dec_layers 12 --exp_name test

#### Evaluate Property preidction task
- python eval_MM_property.py

#### Evaluatte Inverse Design
- python eval_MM_drop.py
- python Property_prediction.py --target_folder [PSMILES generated folder]

#### Train FixMatch + DWD-UT using tranformed data
- python eval_MM_Retrieval.py

<br>




