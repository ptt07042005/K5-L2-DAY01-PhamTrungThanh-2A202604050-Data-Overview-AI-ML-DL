# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Vì tác vụ `image_classification` (single-label) của checkpoint `yolo11n-cls.pt` gán **một lớp chung cho toàn ảnh** (`640x428`), không tạo tọa độ box hay tách object riêng lẻ trong JSON.
3. Class list **đi kèm taxonomy/checkpoint** (`ImageNet-1K` đóng gói cùng `yolo11n-cls.pt`), không phải model tự nghĩ ra tên lớp.
4. Cần cả ba vì: `class_id` cố định để máy xử lý; `class_name` dễ đọc cho người (annotator/reviewer); `taxonomy_name` cho biết đang chiếu theo bộ từ điển lớp nào (tránh nhầm lẫn khi đổi dataset hoặc checkpoint).
5. Guideline cần quy định rõ tiêu chí ưu tiên khi ảnh có nhiều vật thể như cảnh giao thông (ví dụ: chọn phương tiện chiếm diện tích lớn nhất, ở trung tâm khung hình, hoặc chọn chủ đề chính của cảnh), để mọi annotator chọn thống nhất **một** lớp.
6. Vì `score` (`0.510915`) chỉ là độ tự tin của model trên prediction; nó chưa được con người kiểm chứng theo guideline, và thực tế top-1 `cab` của ImageNet đang đoán lệch bối cảnh chung của bức ảnh có nhiều xe buýt (`minibus`), ô tô và người đi bộ (prediction ≠ ground truth).

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.91 ; `bbox_xyxy`=[385.5, 60.1, 498.2, 348.5] ; `bbox_width`=112.7 ; `bbox_height`=288.4
2. Lớp `person`, score `0.91`; box từ góc trên-trái khoảng pixel `(385.5, 60.1)` đến góc dưới-phải `(498.2, 348.5)`; rộng khoảng `112.7` px, cao khoảng `288.4` px.
3. So sánh các ngưỡng trên ảnh `kitchen`: `threshold=0.20` → nhiều prediction hơn (`17` vật thể, xuất hiện thêm `spoon`, `potted plant`, `dining table`, `bottle`); `threshold=0.35` → `11` vật thể; `threshold=0.60` → ít hơn (`6` vật thể: chỉ còn `person`, `bowl`, `oven`).
4. Khi hạ ngưỡng xuống thấp, `reviewer` phải kiểm tra thêm nhiều hộp hơn, bao gồm cả các hộp nhiễu/dự đoán thừa có `score` thấp, làm khối lượng kiểm tra (QC) tăng lên.
5. Quy tắc box: Box phải ôm sát phần nhìn thấy của vật thể (`tight box`), không cắt mất phần nhìn thấy của object và không vẽ một box chung gộp hai object tách rời.
6. Trên ảnh `kitchen`, các vật bị cắt ở mép trái hoặc che khuất chồng lấn góc dưới bàn (như cánh tay `person` `0.61`, khay `bowl` `0.38` và `bowl` `0.72`): tôi làm theo **guideline** nếu đã có quy tắc rõ ràng (ví dụ: còn nhìn thấy ≥ 50% diện tích object trong khung thì vẫn gán); nếu bị che khuất nặng không chắc chắn lớp của vật thể thì thực hiện **escalation** hỏi Lab Coach, không tự bịa nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.90 ; số điểm polygon=`polygon_point_count`=138 ; 3–5 điểm đầu của `polygon_xy`=[[438.0, 62.0], [435.0, 65.0], [431.0, 65.0], [428.0, 68.0], [424.0, 72.0]]
2. Polygon (`polygon_xy`) bám sát theo đường viền cong thực tế của object (như dáng người đầu bếp `person` hay cạnh bàn `dining table`), còn box (`bbox_xyxy`) chỉ là hình chữ nhật bao ngoài nên bị lẫn nhiều vùng nền và mất chi tiết hình dạng.
3. `instance_id` dùng để phân biệt từng cá thể riêng lẻ trong cùng một ảnh output (vd. phân biệt 4 chiếc `bowl` khác nhau là `kitchen-002`, `kitchen-003`...). Nó không phải mã lớp (`class_id` của COCO-80) và không phải mã theo dõi đối tượng qua thời gian (`tracking_id`).
4. Quy tắc biên mask: Mask phải đi sát biên nhìn thấy thực tế của từng instance; không tràn sang object bên cạnh hoặc nền; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác).
5. Trên ảnh `kitchen`, chỗ cụm khay/bát (`bowl` `0.74`, `bowl` `0.55`, `bowl` `0.70`) chạm nhau và đè lên mặt bàn (`dining table` `0.58`) phủ bột làm biên mờ: nếu **guideline** chưa nói rõ cách tách biên thì **escalation** hỏi Lab Coach; nếu **guideline** đã quy định “tách theo khe nhìn thấy và khoét trừ phần mặt bàn bị che” thì làm theo guideline. 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp (`class_name`) cho cả bức ảnh theo taxonomy `ImageNet-1K` | Ảnh `traffic` nhiều xe và người nhưng top-1 đoán `cab` (`score`=0.510915), chưa phản ánh hết bối cảnh (`prediction` ≠ `ground truth`) | Đọc `guideline` để chọn thống nhất 1 lớp chính cho toàn ảnh | Đúng `taxonomy`, đúng tiêu chí ưu tiên của `guideline`, không chép `score` thành nhãn |
| Phát hiện vật thể | Mỗi object: `class_name` + hộp `bbox_xyxy` | Ảnh `kitchen` hạ `threshold=0.20` sinh ra `17` box (nhiều hộp nhiễu); góc dưới trái có `person` `0.61` cắt mép và các `bowl` chồng lấn | Vẽ `tight box` ôm sát từng object theo rule; `escalation` nếu che khuất quá nặng | Box có ôm sát không, thiếu/thừa object không, xử lý đúng các ca cắt mép/che khuất chưa |
| Instance segmentation | Mỗi instance: `class_name` + đa giác `polygon_xy` | Trên bàn `kitchen`, các `bowl` (`0.74`, `0.55`, `0.70`) chạm nhau và đè lên mask `dining table` (`0.58`) phủ bột gây mờ biên | Vẽ mask/polygon bám sát biên nhìn thấy của từng `instance` riêng biệt | Mask có tràn/lem sang object khác không, đã tách đủ từng `instance` cùng lớp (`bowl`) chưa |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   Chỉ sử dụng bộ ảnh mẫu `COCO` công khai có sẵn trong notebook; tuyệt đối không tải ảnh cá nhân, khuôn mặt, biển số xe, thông tin định danh (họ tên, MSSV) hay dữ liệu nội bộ lên `Colab` và `GitHub`.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   Dừng thao tác ngay lập tức và thực hiện `escalation` báo cho `Lab Coach` (hoặc giảng viên phụ trách lab) để xử lý, không tự ý gán nhãn hay chia sẻ dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
