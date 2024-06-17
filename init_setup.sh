echo "This next part will install espeak-ng and git-lfs. If you already have these installed or don't want to run as a super user, use init_setup_no_sudo.sh"
sudo apt install espeak-ng git-lfs
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
