# Cài đặt
**Đảm bảo rằng đang sử dụng python3.7**

Chạy các lệnh sau:
```shell
# Clone repo này
git clone https://github.com/phuocvtran/toxiclassifier.git --branch model
# Chuyển vào folder toxiclassifier
cd toxiclassifier
# Clone repo của VnCoreNLP
git clone https://github.com/vncorenlp/VnCoreNLP.git
# Cài đặt các thư viện cần thiết 
pip install -r requirements.txt
```
# Sử dụng
Predict sẽ hơi chậm (mất vài giây ngay cả khi chỉ có 1 bình luận) là do phải khởi tạo bộ tokenizer của VnCoreNLP.
Nếu không có vấn đề gì có thể sử dụng các câu lệnh dưới đây để dự đoán:
```shell
# Nhận dạng một bình luận bằng model hồi quy logistic
python predict.py -m lg -t 'bình luận này không độc hại'
# Nhận dạng một bình luận bằng model svm
python predict.py -m svm -t 'bình luận này không độc hại'
# Đọc bình luận từ file in.txt và lưu kết quả vào file out.txt 
python predict.py -m lg -f in.txt >> out.txt
```
File input có dạng:
```text
bình luận 1
bình luận 2
bình luận 3
```
File output có dạng:
```text
bình luận 1 
[Nhãn] 0
bình luận 2
[Nhãn] 0
bình luận 3
[Nhãn] 1
```
