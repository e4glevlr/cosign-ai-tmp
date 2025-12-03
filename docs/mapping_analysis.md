# Phân Tích & Mapping: Apple Vision Framework sang Uni-Sign Model

Tài liệu này so sánh sự khác biệt giữa output của `PoseExtractor.swift` (Apple Vision) và input yêu cầu của Uni-Sign (dựa trên `datasets.py`), từ đó đề xuất chiến lược mapping dữ liệu.

## 1. Tổng quan về Uni-Sign Input
Dựa trên phân tích code `datasets.py`, mô hình Uni-Sign sử dụng 69 điểm keypoints, được trích xuất từ chuẩn **COCO-WholeBody (133 điểm)**:

*   **Cấu trúc ID của COCO-WholeBody (133 điểm):**
    *   0-16: Body (Thân, Tay, Chân, Mặt sơ bộ)
    *   17-22: Feet (Chân - Uni-Sign bỏ qua)
    *   23-90: Face (68 điểm landmarks khuôn mặt)
    *   91-111: Left Hand (Tay trái)
    *   112-132: Right Hand (Tay phải)

*   **Input thực tế của Uni-Sign (69 điểm):**
    1.  **Body (9 điểm):** Mũi, Tai (2), Vai (2), Khuỷu (2), Cổ tay (2).
    2.  **Left Hand (21 điểm):** Toàn bộ bàn tay trái.
    3.  **Right Hand (21 điểm):** Toàn bộ bàn tay phải.
    4.  **Face (18 điểm):**
        *   **Jawline (Viền hàm):** 9 điểm (lấy mẫu thưa từ vùng cằm/hàm).
        *   **Inner Lips (Môi trong):** 8 điểm (quan trọng cho khẩu hình).
        *   **Nose Tip (Chóp mũi):** 1 điểm.

---

## 2. Chi tiết Mapping

### A. Body (Thân trên) - 9 Điểm
Apple Vision (`VNHumanBodyPoseObservation`) cung cấp các tên khớp (Joint Name) tương ứng trực tiếp.

| Uni-Sign Index | Bộ phận | Vision Joint Name (`.jointName`) | Ghi chú |
| :--- | :--- | :--- | :--- |
| 0 | Nose (Mũi) | `.nose` | |
| 1 | Left Ear | `.leftEar` | |
| 2 | Right Ear | `.rightEar` | |
| 3 | Left Shoulder | `.leftShoulder` | |
| 4 | Right Shoulder | `.rightShoulder` | |
| 5 | Left Elbow | `.leftElbow` | |
| 6 | Right Elbow | `.rightElbow` | |
| 7 | Left Wrist | `.leftWrist` | |
| 8 | Right Wrist | `.rightWrist` | |

**Chiến lược:** Direct Mapping (1-1).

### B. Hands (Bàn tay) - 21 Điểm/Tay
Apple Vision (`VNHumanHandPoseObservation`) trả về dữ liệu khớp rất chi tiết. Chúng ta cần sắp xếp lại theo thứ tự chuẩn của COCO.

**Thứ tự chuẩn COCO (0-20):**
0: Wrist
1-4: Thumb (CMC, MCP, IP, Tip)
5-8: Index (MCP, PIP, DIP, Tip)
9-12: Middle ...
13-16: Ring ...
17-20: Little ...

| Index (0-20) | Vision Joint Name |
| :--- | :--- |
| 0 | `.wrist` |
| 1 | `.thumbCMC` |
| 2 | `.thumbMP` |
| 3 | `.thumbIP` |
| 4 | `.thumbTip` |
| 5 | `.indexMCP` |
| 6 | `.indexPIP` |
| ... | (Tương tự cho Middle, Ring, Little) |
| 20 | `.littleTip` |

**Chiến lược:** Loop qua các joint name định sẵn và gán vào array.

### C. Face (Khuôn mặt) - 18 Điểm (Phức tạp nhất)
Đây là phần khó nhất vì `VNFaceObservation` trả về **đường bao (contours)** gồm mảng các điểm (ví dụ: `faceContour` có thể có ~50 điểm tùy độ phân giải), trong khi Uni-Sign cần **index cụ thể** của chuẩn 68-landmarks.

**Phân tích Logic của Uni-Sign:**
*   `range(23, 40, 2)` (9 điểm): Tương ứng với vùng **Jawline (Viền hàm)** trong chuẩn 68 điểm (Local index 0-16).
*   `range(83, 91)` (8 điểm): Tương ứng với vùng **Inner Lips (Môi trong)** trong chuẩn 68 điểm (Local index 60-67).
*   `Index 53` (1 điểm): Tương ứng với **Nose Tip (Chóp mũi)** trong chuẩn 68 điểm (Local index 30).

**Chiến lược Mapping từ Apple Vision:**

1.  **Jawline (Viền hàm):**
    *   *Nguồn Vision:* `landmarks.faceContour` (Mảng n điểm).
    *   *Xử lý:* Lấy mẫu (sample) 9 điểm cách đều nhau trên đường `faceContour`. Do Vision trả về đường bao từ tai này sang tai kia, ta có thể chia mảng thành 9 phần và lấy điểm mốc.

2.  **Inner Lips (Môi trong):**
    *   *Nguồn Vision:* `landmarks.innerLips` (Mảng n điểm).
    *   *Xử lý:* Lấy mẫu 8 điểm đặc trưng (thường là các điểm cực đại, cực tiểu và trung gian). Hoặc đơn giản là chia đều mảng `innerLips` lấy 8 điểm.

3.  **Nose Tip (Chóp mũi):**
    *   *Nguồn Vision:* `landmarks.nose` hoặc `landmarks.noseCrest`.
    *   *Xử lý:* Lấy điểm cuối cùng của mảng `noseCrest` (sống mũi) hoặc điểm trung tâm của `nose`.

---

## 3. Định dạng dữ liệu đầu ra (JSON/PKL)
Để chạy được inference python script, Swift cần xuất ra file JSON hoặc .pkl có cấu trúc tương tự file training.

**Cấu trúc đề xuất (JSON):**
```json
{
  "video_name": "sample_video",
  "width": 1920,
  "height": 1080,
  "frames": [
    {
      "frame_index": 0,
      "keypoints": [
         // Mảng chứa [x, y, confidence]
         // Body (0-8)
         [0.5, 0.5, 0.9], ...
         // Face (9-26) - 18 điểm
         [0.4, 0.4, 0.8], ...
         // Left Hand (27-47) - 21 điểm
         [0.1, 0.2, 0.9], ...
         // Right Hand (48-68) - 21 điểm
         [0.8, 0.2, 0.9], ...
      ]
    },
    ...
  ]
}
```

**Lưu ý quan trọng về Tọa độ:**
1.  **Hệ tọa độ:** Vision dùng hệ tọa độ chuẩn hóa (0.0 - 1.0) với gốc (0,0) ở **dưới cùng bên trái**.
2.  **Yêu cầu Model:** Hầu hết model CV (bao gồm COCO) dùng gốc (0,0) ở **trên cùng bên trái**.
3.  **Hành động:** Trong Swift, khi xuất ra JSON, cần lật trục Y: `y_out = 1.0 - y_vision`.

## 4. Kết luận
Chúng ta hoàn toàn có thể dùng `PoseExtractor.swift` để tạo dữ liệu cho Uni-Sign. Tuy nhiên, cần viết thêm một lớp "Converter" (có thể bằng Python sau khi xuất JSON từ Swift) để:
1.  Gộp các nguồn (Body, Hand, Face) vào đúng thứ tự mảng.
2.  Thực hiện sampling (lấy mẫu) thông minh cho các điểm Face (từ Contour sang 18 điểm cố định).
3.  Đảm bảo chuẩn hóa tọa độ và lật trục Y.
