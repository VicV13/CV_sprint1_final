# CV_sprint1_final
touch hist
history
touch notebook.ipynb
cd ../../
mv test/annotations.json ./annotations/test/
mv valid/annotations.json ./annotations/valid/
mkdir annotations/valid
ll
mv -f val/annotations.json ./annotations/val/annotations.json
mv -f val/annotations.json ./annotations/val/
mv -f val/annotations.json ./annotations/val
mv -f train/annotations.json ./annotations/train/
mkdir annotations/val
mkdir annotations/test
mkdir annotations/train
mv --help
mv -r train/annotations.json ./annotations/train/
mv train/annotations.json ./annotations/train/
mkdir annotations
cd ../
cat annotations.json
ll | grep ann
cd train/
cd datasets/minecraft/
rm -rf datasets/
rm datasets/
mv dataset datasets
mkdir dataset
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
# 1. Скачиваем репозиторий
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# 2. Устанавливаем сам mmdet в режиме разработчика
pip install -v -e . --no-build-isolation

# 3. Даунгрейдим NumPy до стабильной версии для этой сборки
pip install "numpy==1.26.4"
# 1. Обновляем pip и ставим утилиту mim
python -m pip install --upgrade pip
pip install openmim==0.3.9

# 2. Ставим движок обучения строго заданной версии
pip install "mmengine==0.10.7"

# 3. Устанавливаем базовые свёртки mmcv нужной версии под наш CPU-торч
mim install "mmcv==2.1.0"
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 -c pytorch -y
python -m pip install pip==21.2.3
conda activate mmdet_env
conda create --name mmdet_env python=3.10 -y
cd CV_sprint1_final/
mkdir CV_sprint1_final
git push -u origin main
git commit -m "first commit. lesson1"
git add .