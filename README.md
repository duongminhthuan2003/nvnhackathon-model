## Cài đặt:
### Clone repo và cài các libraries cần thiết:
```bash
git clone https://github.com/photienanh/Vietnamese-Sign-Language-Recognition
cd Vietnamese-Sign-Language-Recognition
pip install -r requirements.txt
```
## Training:
### Tạo data set:
1. Tải dataset từ link: https://drive.google.com/drive/folders/1wSo8qsBwAOVg-JHWVK3WNENl0se3dqiC?usp=sharing
2. Quay video training và đặt tên theo format v[xxxx]_[y].mp4
3. Chỉnh sửa lại file Text trong dataset cho đúng với từng video để notebook nhận được video mới
### Tạo data:
```bash
python create_data_augment.py
```
### Training
1. Tạo tài khoản Kaggle, sau đó xác minh để có thể dùng GPU train nhanh hơn **rất** nhiều.
2. Nén folder Data (gồm các file .npz) tạo được ở phần "Tạo data" và upload lên Kaggle.
3. Upload file label_map.json lên Kaggle.
4. Mở note book Kaggle để train: https://www.kaggle.com/code/duongminhthuan/trainning
5. Ở phần input bên phải, nhấn Add input và chọn folder Data + file label_map.json vừa tải lên.
6. Ở cell thứ 2 của notebook, sửa lại:
```bash
DATA_PATH        = 'đường dẫn chỉ đến folder Data'
LABEL_MAP_PATH   = 'đường dẫn chỉ đến file label_map.json'
```
7. Chọn vào Settings của notebook -> chọn Accelerator -> GPU T4 x2
8. Nhấn Run all
9. Sau khi train xong, file final_model.keras sẽ nằm ở trong folder output ở bên tay phải.
10. Có thể test model trước với Streamlit.
