rm -rf ~/wusuowei/data/single_cell/panacea
mkdir -p ~/wusuowei/data/single_cell/panacea
rm -rf ~/.cache/huggingface/datasets
python3 preprocess_pretrain.py
python3 process.py --data pancreas --adata /data/tiankang/SCALEX/pancreas/adata.h5ad
python3 process.py --data mouse_atlas --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_atlas/adata/adata.h5ad
python3 process.py --data mouse_lung_gr --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_lung_gr/adata/adata.h5ad
python3 process.py --data mouse_kidney_gr --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_kidney_gr/adata/adata.h5ad
python3 process.py --data mouse_brain_spatial --adata /data/tiankang/SCALEX/mouse_brain_spatial/adata.h5ad
python3 process.py --data human_atlas_adult --adata /data/tiankang/SCALEX/human_atlas_adult/adata.h5ad
