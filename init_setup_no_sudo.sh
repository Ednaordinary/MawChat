echo "Warning: Assuming you have git-lfs and espeak-ng already"
python3 -m venv venv
source venv/bin/activate
git clone https://github.com/yl4579/StyleTTS2
pip install -r requirements.txt
cd StyleTTS2
pip install -r requirements.txt
git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech
mv StyleTTS2-LJSpeech/Models .
mv Models ..
mv Modules ..
mv Utils ..
mv models.py ..
mv text_utils.py ..
mv utils.py ..
cd ..
rm -rf ./StyleTTS2
