# Các thư viện được sử dụng:
pip install annoy (version 1.17.0)
pip install librosa (version 0.9.2)
pip install soundfile (version 0.10.3)
pip install tqdm (version 4.64.0)
pip install numpy (version 1.21.5)
pip install python_speech_features(version 0.6)
pip install argparse (version 1.4.0)
pip install flask (version 1.1.2)
pip install tensorflow (version 2.8)
pip install keras (version 2.11.0)
pip install matplotlib (version 3.5.1)
pip install pandas (version 1.4.2)
pip install path (version 16.5.2)


# Câu lệnh được sử dung:
python cv-run_save_vgg16.py -t train -r test -e 500 -c 100 -k 0 --mod cnn001

python cv-run_save_vgg16.py -t train -r test -e 500 -c 100 -k 0 --mod fc

#Giới thiệu các file

#2D
file cutvideo.py: Các tệp âm thanh gốc được chia thành các đoạn âm thanh có độ dài khác nhau
     randomcut.py: cắt ngẫu nhiên các file âm thanh gốc thành các clip 
     preprocessing.py: chuyển đổi âm thanh thành ảnh phổ
     cv-run_save_vgg16.py: xây dựng kiến trúc mô hình CNN/FC dùng đánh giá hiệu suất phân loại