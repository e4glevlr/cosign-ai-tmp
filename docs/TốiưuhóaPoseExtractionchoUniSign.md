# **Báo cáo Nghiên cứu Kỹ thuật: Tối ưu hóa Mapping và Hợp nhất Pose Extraction từ Apple Vision Framework sang Chuẩn RTMPose (COCO-WholeBody) cho UniSign trên Phần cứng Đầu cuối**

## **1\. Tổng quan Điều hành và Phạm vi Nghiên cứu**

Trong bối cảnh trí tuệ nhân tạo biên (Edge AI) đang phát triển mạnh mẽ, việc triển khai các mô hình nhận diện ngôn ngữ ký hiệu (Sign Language Recognition \- SLR) phức tạp như **UniSign** lên các thiết bị di động của Apple (iOS/macOS) đặt ra những thách thức đáng kể về hiệu năng và tính tương thích. UniSign, một framework thống nhất cho việc hiểu ngôn ngữ ký hiệu, thường được huấn luyện dựa trên các chuẩn dữ liệu mở như **COCO-WholeBody** với 133 điểm đặc trưng (keypoints).1 Ngược lại, hệ sinh thái phần cứng của Apple cung cấp các API Vision Framework được tối ưu hóa sâu sắc cho kiến trúc Apple Neural Engine (ANE), nhưng lại trả về các định dạng dữ liệu rời rạc và không đồng nhất với chuẩn COCO.3  
Báo cáo này tập trung giải quyết bài toán "bắc cầu" kỹ thuật: làm thế nào để chuyển đổi (mapping) đầu ra từ các thuật toán VNDetectHumanBodyPoseRequest, VNDetectHumanHandPoseRequest, và VNDetectFaceLandmarksRequest của Apple thành véc-tơ đặc trưng 133 điểm chuẩn của RTMPose/COCO-WholeBody một cách hoàn hảo nhất. Mục tiêu cốt lõi là tận dụng tốc độ xử lý vượt trội của Vision Framework (có thể đạt độ trễ dưới 10ms trên ANE) để thay thế cho quá trình suy luận nặng nề của các mô hình RTMPose gốc khi chạy trên thiết bị di động, từ đó tăng cường hiệu năng tổng thể cho ứng dụng UniSign mà không hy sinh độ chính xác ngữ nghĩa.5  
Nghiên cứu này sẽ phân tích sâu về cấu trúc giải phẫu học của hai hệ quy chiếu, đề xuất các thuật toán nội suy và hợp nhất (fusion algorithms) để xử lý sự sai lệch về số lượng điểm ảnh, và xây dựng một kiến trúc đường ống (pipeline) xử lý dữ liệu tối ưu nhằm giảm thiểu độ trễ từ camera đến mô hình UniSign.

## ---

**2\. Cơ sở Lý thuyết và Phân tích Sự Khác biệt Kiến trúc**

Để thực hiện việc mapping chính xác, trước hết cần phải giải cấu trúc (deconstruct) hai hệ thống định dạng dữ liệu đang được xem xét. Sự khác biệt không chỉ nằm ở số lượng điểm mà còn ở định nghĩa ngữ nghĩa (semantic definition) của từng khớp xương và vùng đặc trưng trên khuôn mặt.

### **2.1 Chuẩn hóa Dữ liệu COCO-WholeBody (Mục tiêu của RTMPose)**

Mô hình RTMPose, được sử dụng rộng rãi trong các nghiên cứu hiện đại như UniSign, dựa trên bộ dữ liệu COCO-WholeBody. Đây là một phần mở rộng của bộ dữ liệu COCO gốc, bổ sung thêm các điểm đặc trưng cho tay, chân và mặt để tạo thành một đồ thị toàn diện gồm 133 điểm.7 Sự hiểu biết tường tận về thứ tự và ý nghĩa của 133 điểm này là điều kiện tiên quyết cho bất kỳ thuật toán mapping nào.  
Cấu trúc topo của COCO-WholeBody được chia thành bốn nhóm chính:

1. **Cơ thể (Body \- 17 điểm):** Kế thừa từ chuẩn COCO Keypoints 2017\. Các điểm này bao gồm mũi, hai mắt, hai tai, hai vai, hai khuỷu tay, hai cổ tay, hai hông, hai đầu gối và hai mắt cá chân.9 Điểm đáng lưu ý là chuẩn này *không* bao gồm điểm "Cổ" (Neck) hay điểm "Gốc" (Root/Pelvis) trung tâm, mà thường suy luận dựa trên trung điểm của hai vai hoặc hai hông.  
2. **Bàn chân (Feet \- 6 điểm):** Bao gồm ngón cái, ngón út và gót chân cho mỗi bàn chân. Đây là các điểm mở rộng giúp định vị hướng đứng chính xác hơn trong không gian 3D, tuy nhiên trong bài toán SLR, vai trò của chúng thường thứ yếu.8  
3. **Khuôn mặt (Face \- 68 điểm):** Tuân theo chuẩn iBUG 300-W (Intelligent Behaviour Understanding Group). Đây là hệ thống đánh dấu khuôn mặt cực kỳ chi tiết, bao gồm đường viền hàm, lông mày, sống mũi, cánh mũi, mắt và chi tiết môi.11 Đối với UniSign, các điểm này mang thông tin quan trọng về biểu cảm khuôn mặt (non-manual markers), đóng vai trò ngữ pháp trong ngôn ngữ ký hiệu (ví dụ: nhướn mày để hỏi, mím môi để phủ định).  
4. **Bàn tay (Hands \- 42 điểm):** Mỗi bàn tay có 21 điểm, bao gồm cổ tay và 4 điểm cho mỗi ngón tay (khớp CMC/MCP, PIP, DIP và đầu ngón tay).8 Độ chính xác của nhóm điểm này là yếu tố sống còn đối với UniSign.

### **2.2 Kiến trúc Trích xuất của Apple Vision Framework**

Apple Vision Framework không hoạt động như một mô hình monolithic (nguyên khối) trả về toàn bộ 133 điểm cùng lúc. Thay vào đó, nó là tập hợp của các yêu cầu (Requests) riêng biệt, hoạt động độc lập và có thể chạy song song hoặc tuần tự.3

1. **VNDetectHumanBodyPoseRequest:** Trả về một quan sát VNHumanBodyPoseObservation. Cấu trúc xương của Apple bao gồm 19 điểm, khác biệt so với 17 điểm của COCO. Điểm khác biệt lớn nhất là Apple cung cấp điểm .neck (cổ) và .root (xương cụt/hông), nhưng lại thường thiếu nhất quán trong việc cung cấp các điểm trên khuôn mặt (mắt, tai) trong các phiên bản iOS cũ, mặc dù các phiên bản mới đã bổ sung.3  
2. **VNDetectHumanHandPoseRequest:** Trả về VNHumanHandPoseObservation. Cấu trúc này tương đồng cao với chuẩn COCO về mặt giải phẫu học (21 điểm mỗi tay), tuy nhiên hệ quy chiếu tọa độ lại hoàn toàn cục bộ đối với vùng chứa bàn tay (ROI) hoặc toàn ảnh tùy thuộc vào cấu hình.16  
3. **VNDetectFaceLandmarksRequest:** Trả về VNFaceObservation chứa VNFaceLandmarks2D. Đây là điểm phức tạp nhất. Apple không trả về một danh sách cố định 68 điểm. Thay vào đó, họ trả về các "vùng" (regions) như faceContour, medianLine, outerLips, v.v. Số lượng điểm trong mỗi vùng là *động* (dynamic) và phụ thuộc vào thiết bị cũng như phiên bản iOS (ví dụ: faceContour có thể có 13 điểm trên iPhone X nhưng 15 điểm trên iPhone 14 Pro).18

### **2.3 Thách thức về Không gian Tọa độ (Coordinate Space)**

Một rào cản kỹ thuật lớn khi tích hợp Vision vào UniSign là sự khác biệt về không gian tọa độ:

* **Vision:** Sử dụng hệ tọa độ chuẩn hóa (Normalized Coordinates) từ , với gốc tọa độ (0,0) nằm ở **Góc Dưới Bên Trái** (Bottom-Left).17  
* **RTMPose/COCO:** Thường sử dụng hệ tọa độ Pixel (Pixel Coordinates) từ và \[0, H\], với gốc tọa độ (0,0) nằm ở **Góc Trên Bên Trái** (Top-Left).8

Việc chuyển đổi không chỉ đơn thuần là nhân với kích thước ảnh mà còn phải thực hiện phép lật trục Y (Y-flip) và xử lý tỉ lệ khung hình (Aspect Ratio) nếu ảnh đầu vào bị cắt (crop) hoặc co giãn (scale) trước khi đưa vào mô hình.22 Sai sót trong bước này sẽ dẫn đến việc UniSign nhận diện sai hướng chuyển động của tay (ví dụ: cử chỉ "đi lên" bị hiểu nhầm thành "đi xuống").

## ---

**3\. Chiến lược Mapping Chi tiết và Tối ưu hóa**

Để đạt được sự "hoàn hảo" như yêu cầu, chúng ta không thể chỉ mapping 1-1 đơn giản. Cần áp dụng các thuật toán nội suy (interpolation) và tái lấy mẫu (resampling) để đảm bảo tính nhất quán của dữ liệu đầu vào cho UniSign.

### **3.1 Mapping Cơ thể (Body): Từ Vision 19 điểm sang COCO 17 điểm**

UniSign yêu cầu 17 điểm cơ thể chuẩn. Vision cung cấp 19 điểm. Nhiệm vụ là loại bỏ các điểm thừa và map chính xác các điểm tương đồng, đồng thời xử lý các điểm thiếu hụt tiềm năng.  
**Bảng Mapping Chi Tiết:**

| Chỉ số COCO | Tên COCO | Tương ứng Apple Vision (JointName) | Chiến lược Xử lý & Tối ưu hóa |
| :---- | :---- | :---- | :---- |
| 0 | Nose | .nose | Lấy trực tiếp. Nếu Vision không trả về (confidence \= 0), sử dụng trung điểm của .leftEye và .rightEye hoặc ngoại suy từ .neck và trung điểm vai. |
| 1 | Left Eye | .leftEye | Lấy trực tiếp. Lưu ý: Vision định nghĩa Left/Right theo *ngôi thứ nhất* (của chủ thể), cần kiểm tra UniSign có yêu cầu *ngôi thứ ba* (người nhìn) hay không để hoán đổi (swap). |
| 2 | Right Eye | .rightEye | Tương tự Left Eye. |
| 3 | Left Ear | .leftEar | Lấy trực tiếp. |
| 4 | Right Ear | .rightEar | Lấy trực tiếp. |
| 5 | Left Shoulder | .leftShoulder | Lấy trực tiếp. |
| 6 | Right Shoulder | .rightShoulder | Lấy trực tiếp. |
| 7 | Left Elbow | .leftElbow | Lấy trực tiếp. |
| 8 | Right Elbow | .rightElbow | Lấy trực tiếp. |
| 9 | Left Wrist | .leftWrist | **CỰC KỲ QUAN TRỌNG:** Điểm này dùng làm "mỏ neo" (anchor) để gắn kết bàn tay chi tiết vào cơ thể. |
| 10 | Right Wrist | .rightWrist | Tương tự Left Wrist. |
| 11 | Left Hip | .leftHip | Lấy trực tiếp. |
| 12 | Right Hip | .rightHip | Lấy trực tiếp. |
| 13 | Left Knee | .leftKnee | Lấy trực tiếp. |
| 14 | Right Knee | .rightKnee | Lấy trực tiếp. |
| 15 | Left Ankle | .leftAnkle | Lấy trực tiếp. |
| 16 | Right Ankle | .rightAnkle | Lấy trực tiếp. |

**Xử lý điểm dư thừa:** Các điểm .neck và .root của Apple không được đưa vào véc-tơ 133 điểm cuối cùng. Tuy nhiên, chúng có giá trị lớn trong việc **kiểm chứng độ tin cậy** (sanity check). Ví dụ: vector từ .root đến .neck xác định trục dọc của cơ thể, giúp UniSign định hướng (normalize orientation) nếu người dùng nghiêng người.15

### **3.2 Mapping Khuôn mặt (Face): Tái cấu trúc iBUG 68 từ VNFaceLandmarks2D**

Đây là phần phức tạp nhất. UniSign, nếu được huấn luyện trên COCO-WholeBody, sẽ mong đợi đúng 68 điểm theo chuẩn iBUG. Apple Vision trả về các đường cong (contours) với số lượng điểm không cố định. Việc mapping trực tiếp (ví dụ: lấy điểm thứ 3 của Apple làm điểm thứ 3 của COCO) sẽ dẫn đến sai lệch vị trí nghiêm trọng.  
**Giải thuật Tái lấy mẫu Đường cong (Contour Resampling Algorithm):**  
Để tạo ra 68 điểm chuẩn, ta cần coi các điểm Vision trả về là các điểm kiểm soát (control points) và thực hiện nội suy.

1. **Đường viền hàm (Jawline \- COCO indices 23-39, 17 điểm):**  
   * *Dữ liệu Apple:* faceContour. Thường chứa khoảng 9-15 điểm.  
   * *Thuật toán:* Xây dựng một đường Spline (ví dụ: Catmull-Rom Spline) đi qua tất cả các điểm faceContour của Apple. Sau đó, tính toán độ dài đường cong và lấy mẫu 17 điểm cách đều nhau trên đường cong đó. Điều này đảm bảo mật độ điểm tương đồng với chuẩn iBUG bất kể Apple trả về bao nhiêu điểm.18  
2. **Lông mày (Eyebrows \- COCO indices 40-49):**  
   * *Dữ liệu Apple:* leftEyebrow, rightEyebrow.  
   * *Mục tiêu:* 5 điểm mỗi bên.  
   * *Thuật toán:* Tương tự hàm, tái lấy mẫu 5 điểm cách đều trên đường cong lông mày của Apple.  
3. **Sống mũi và Cánh mũi (Nose \- COCO indices 50-58):**  
   * *Dữ liệu Apple:* noseCrest (sống mũi), nose (cánh mũi/đáy mũi).  
   * *Mapping:*  
     * COCO 50-53 (Sống mũi): Tái lấy mẫu 4 điểm từ noseCrest. Lưu ý Vision có thể trả về noseCrest ngắn hơn thực tế, cần ngoại suy điểm đầu (giữa hai mắt) nếu cần.  
     * COCO 54-58 (Cánh mũi): Tái lấy mẫu 5 điểm từ nose.  
4. **Mắt (Eyes \- COCO indices 59-70):**  
   * *Dữ liệu Apple:* leftEye, rightEye, leftPupil, rightPupil.  
   * *Mapping:* COCO sử dụng 6 điểm định hình mí mắt (không bao gồm đồng tử).  
   * *Lưu ý:* Cần kiểm tra thứ tự điểm (Winding Order). Vision thường trả về các điểm theo chiều kim đồng hồ hoặc ngược lại. Cần đảm bảo thứ tự khớp với iBUG (thường bắt đầu từ góc mắt trong hoặc ngoài). Đồng tử (pupil) của Apple rất chính xác, có thể dùng để tinh chỉnh vị trí trung tâm của mắt nếu UniSign hỗ trợ tính năng theo dõi hướng nhìn (gaze tracking).11  
5. **Môi (Mouth \- COCO indices 71-90):**  
   * *Dữ liệu Apple:* outerLips, innerLips.  
   * *Mapping:*  
     * COCO 71-82 (Môi ngoài): 12 điểm. Tái lấy mẫu từ outerLips.  
     * COCO 83-90 (Môi trong): 8 điểm. Tái lấy mẫu từ innerLips.

### **3.3 Mapping Bàn tay (Hands): Hợp nhất và Căn chỉnh Cổ tay**

Trong ngôn ngữ ký hiệu, vị trí bàn tay so với cơ thể là yếu tố quyết định ngữ nghĩa. Tuy nhiên, VNDetectHumanHandPoseRequest hoạt động độc lập và có thể có độ lệch tọa độ so với VNDetectHumanBodyPoseRequest.17  
Vấn đề "Gãy cổ tay" (Wrist Dislocation):  
Tọa độ cổ tay (wrist) từ Body Request (toàn thân, độ phân giải thấp hơn trên vùng tay) và Hand Request (tập trung, độ phân giải cao) thường không trùng khớp. Nếu đưa trực tiếp vào UniSign, cánh tay có thể bị đứt gãy tại cổ tay.  
**Giải pháp "Mỏ neo" (Anchor Stitching Strategy):**

1. Sử dụng tọa độ Cổ tay từ **Body Pose** (COCO index 9 & 10\) làm "chân lý" (Ground Truth) cho vị trí gốc.  
2. Lấy tọa độ Cổ tay từ **Hand Pose**.  
3. Tính véc-tơ dịch chuyển (Offset Vector): $\\Delta \= \\text{BodyWrist} \- \\text{HandWrist}$.  
4. Cộng véc-tơ $\\Delta$ vào tất cả 20 điểm còn lại của bàn tay (ngón cái đến ngón út).  
5. Điều này "dán" bàn tay chi tiết vào đúng vị trí cổ tay trên cơ thể, đảm bảo tính liên tục của bộ xương (skeleton continuity).25

Thứ tự Mapping Bàn tay (COCO indices 91-132):  
Thứ tự các khớp ngón tay của Apple (CMC \-\> MCP \-\> IP \-\> Tip cho ngón cái, và MCP \-\> PIP \-\> DIP \-\> Tip cho các ngón khác) tương thích hoàn toàn 1:1 với chuẩn COCO. Không cần tái lấy mẫu, chỉ cần sắp xếp đúng thứ tự mảng.13

### **3.4 Mapping Bàn chân (Feet \- COCO indices 17-22)**

Vision không cung cấp thông tin ngón chân hay gót chân.3 Đối với UniSign và SLR nói chung, thông tin này thường là nhiễu hoặc không cần thiết.

* **Giải pháp:** Gán giá trị 0 (Zero-padding) hoặc sao chép tọa độ mắt cá chân (ankle) vào các vị trí ngón chân/gót chân. Điều này giữ cho véc-tơ đầu vào đủ 133 điểm mà không gây ra tín hiệu sai lệch cho mô hình.

## ---

**4\. Kiến trúc Hệ thống và Tối ưu hóa Hiệu năng trên Apple Silicon**

Để đạt được "hiệu năng cho phần cứng đầu cuối tốt hơn", việc chỉ mapping đúng là chưa đủ. Cần phải thiết kế luồng xử lý (pipeline) sao cho tận dụng tối đa phần cứng.

### **4.1 So sánh Hiệu năng: Vision vs. RTMPose CoreML**

Sử dụng trực tiếp RTMPose (thông qua CoreML) trên iOS gặp phải các vấn đề:

* **Bộ nhớ:** RTMPose-l/m yêu cầu tải trọng mô hình lớn, chiếm dụng băng thông bộ nhớ.  
* **Nhiệt độ:** Chạy liên tục mô hình nặng làm nóng máy, dẫn đến giảm xung nhịp (throttling) sau vài phút, làm giảm FPS.  
* **Neural Engine (ANE):** Một số lớp (layers) của RTMPose (như SimCC post-processing) có thể không tương thích hoàn toàn với ANE, buộc phải chạy trên GPU hoặc CPU, gây tốn pin.26

Ngược lại, Vision Framework sử dụng các mô hình tích hợp sẵn trong OS, được chia sẻ bộ nhớ và tối ưu hóa cấp thấp (low-level optimization) cho ANE. Việc chạy song song 3 requests (Body, Hand, Face) qua Vision thường tiêu tốn ít năng lượng hơn và đạt FPS cao hơn so với một mô hình RTMPose đơn lẻ.28

### **4.2 Thiết kế Pipeline Đa luồng (Multithreaded Pipeline)**

Để đạt 30-60 FPS cho UniSign, kiến trúc pipeline cần tuân thủ mô hình **Producer-Consumer**:

1. **Thu nhận ảnh (Camera Acquisition):**  
   * Sử dụng AVCaptureSession xuất ra CMSampleBuffer.  
   * Định dạng pixel: kCVPixelFormatType\_420YpCbCr8BiPlanarFullRange (YUV420) là định dạng tự nhiên của camera và Vision hỗ trợ trực tiếp, tránh tốn chi phí chuyển đổi sang RGB.30  
2. **Xử lý Vision (Vision Processing):**  
   * Khởi tạo **một** VNImageRequestHandler duy nhất cho mỗi khung hình.  
   * Thực hiện phương thức .perform() trên một hàng đợi nền (background dispatch queue) với QoS là .userInteractive.  
   * Việc gom nhóm (batching) các requests vào một perform cho phép ANE tái sử dụng các tính toán đặc trưng ảnh gốc (backbone features), giảm lặp lại tính toán.14  
3. **Lớp Mapping & Hợp nhất (Fusion Layer):**  
   * Thực hiện ngay trong block completionHandler hoặc một Serial Queue riêng biệt.  
   * Tính toán mapping (như mô tả ở Mục 3\) là các phép toán đại số tuyến tính nhẹ, tốn không đáng kể thời gian CPU (\< 1ms).  
   * Áp dụng **Bộ lọc 1 Euro (1€ Filter)** để khử nhiễu (jitter). Vision có độ nhạy cao nên các điểm thường bị rung nhẹ. Bộ lọc này giúp làm mượt chuyển động mà không gây độ trễ lớn như Moving Average, rất quan trọng để UniSign bắt trọn các chuyển động nhanh của tay.31  
4. **Suy luận UniSign (Inference):**  
   * Đưa véc-tơ 133 điểm (đã được làm phẳng \- flattened) vào mô hình UniSign CoreML.  
   * Mô hình UniSign lúc này chỉ cần là một mạng nơ-ron thuần túy (ví dụ: LSTM, Transformer hoặc GCN) mà không cần phần xử lý ảnh (CNN Backbone), giảm kích thước mô hình từ hàng trăm MB xuống còn vài MB.

### **4.3 Xử lý Dữ liệu Thiếu (Occlusion Handling)**

Trong SLR, việc tay che mặt hoặc tay này che tay kia là thường xuyên.

* **Vision:** Sẽ trả về confidence \= 0 hoặc không trả về kết quả cho vùng bị che.  
* **Chiến lược UniSign:** Không được để giá trị NaN hoặc NULL. Cần thiết lập quy tắc:  
  * Nếu mất dấu trong \< 5 khung hình: Sử dụng thuật toán **Optical Flow** hoặc **Kalman Filter** để dự đoán vị trí tiếp theo dựa trên vận tốc cũ.  
  * Nếu mất dấu lâu hơn: Gán về giá trị mặc định (ví dụ: 0\) và kèm theo một "mask" (mặt nạ) để mô hình UniSign biết bỏ qua các điểm này trong cơ chế Attention.32

## ---

**5\. Hiện thực hóa Thuật toán (Implementation Details)**

Dưới đây là mô tả kỹ thuật (Technical Specification) cho module chuyển đổi, có thể dùng để cài đặt trực tiếp bằng Swift.

### **5.1 Biến đổi Tọa độ (Coordinate Transformation)**

Công thức chuyển đổi từ Vision ($P\_v$) sang UniSign/COCO ($P\_c$):

$$P\_{c}.x \= P\_{v}.x \\times W\_{input}$$

$$P\_{c}.y \= (1 \- P\_{v}.y) \\times H\_{input}$$  
Trong đó:

* $(1 \- P\_{v}.y)$ là phép lật trục Y (Vision gốc dưới-trái, COCO gốc trên-trái).  
* $W\_{input}, H\_{input}$ là kích thước đầu vào mà mô hình UniSign mong đợi (ví dụ: 256x256 hoặc 1920x1080).

**Lưu ý quan trọng:** Nếu UniSign sử dụng tỉ lệ khung hình khác với camera (ví dụ: Camera 16:9 nhưng UniSign train trên ảnh vuông 1:1), cần thực hiện bước **Center Crop** ảo. Tọa độ Vision cần được ánh xạ lại theo vùng crop trước khi nhân với kích thước đích.

### **5.2 Mã giả thuật toán Fusion (Pseudo-code)**

Swift

struct WholeBodySnapshot {  
    var body: \[CGPoint\] // 17  
    var feet: \[CGPoint\] // 6 (Zeros)  
    var face: \[CGPoint\] // 68  
    var leftHand: \[CGPoint\] // 21  
    var rightHand: \[CGPoint\] // 21  
      
    func toFlatArray() \-\> \[Float\] {... }  
}

func fusionPipeline(sampleBuffer: CMSampleBuffer) {  
    let handler \= VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation:.up)  
      
    // Yêu cầu đồng thời  
    let bodyReq \= VNDetectHumanBodyPoseRequest()  
    let handReq \= VNDetectHumanHandPoseRequest()  
    handReq.maximumHandCount \= 2  
    let faceReq \= VNDetectFaceLandmarksRequest()  
      
    try? handler.perform()  
      
    // 1\. Xử lý Body  
    guard let bodyObs \= bodyReq.results?.first else { return } // Lấy người nổi bật nhất  
    let cocoBody \= mapBodyPoints(bodyObs) // Hàm mapping 19-\>17 điểm  
      
    // 2\. Xử lý Face  
    var cocoFace \= \[CGPoint\](repeating:.zero, count: 68\)  
    if let faceObs \= faceReq.results?.first { // Cần logic match face với body  
        cocoFace \= resampleFaceContours(faceObs) // Hàm tái lấy mẫu spline  
    }  
      
    // 3\. Xử lý Hand (Stitching)  
    var leftHandPoints \= \[CGPoint\](repeating:.zero, count: 21\)  
    var rightHandPoints \= \[CGPoint\](repeating:.zero, count: 21\)  
      
    // Phân loại trái/phải dựa trên chirality và khoảng cách tới cổ tay body  
    let hands \= classifyHands(handReq.results, bodyWrists: (cocoBody, cocoBody))  
      
    if let left \= hands.left {  
        let delta \= cocoBody \- left.wristPosition  
        leftHandPoints \= left.allPoints.map { $0 \+ delta } // Stitching  
    }  
      
    if let right \= hands.right {  
        let delta \= cocoBody \- right.wristPosition  
        rightHandPoints \= right.allPoints.map { $0 \+ delta } // Stitching  
    }  
      
    // 4\. Đóng gói & Filter  
    let snapshot \= WholeBodySnapshot(..., leftHand: leftHandPoints, rightHand: rightHandPoints)  
    let smoothedSnapshot \= oneEuroFilter.process(snapshot)  
      
    // 5\. Gửi sang UniSign  
    uniSignModel.predict(smoothedSnapshot.toFlatArray())  
}

## ---

**6\. Kết luận và Kiến nghị Triển khai**

Để đạt được sự "hoàn hảo" trong việc mapping từ Apple Vision sang RTMPose cho UniSign, chúng ta không chỉ đơn thuần là đổi tên các khớp xương. Đó là một quá trình **tái cấu trúc hình học** bao gồm:

1. **Loại bỏ nhiễu:** Lọc bỏ các điểm không chuẩn COCO (Neck, Root) của Vision.  
2. **Tái tạo dữ liệu:** Dùng thuật toán Spline để sinh ra đúng 68 điểm mặt từ các đường contour động.  
3. **Hợp nhất không gian:** Dùng kỹ thuật "Mỏ neo" (Anchoring) để đồng bộ hóa hệ tọa độ cục bộ của bàn tay vào hệ tọa độ toàn thân.  
4. **Tối ưu hóa phần cứng:** Sử dụng pipeline VNImageRequestHandler đơn nhất để tận dụng khả năng chia sẻ tính toán của ANE.

Giải pháp này cho phép UniSign hoạt động với độ trễ thấp nhất có thể trên thiết bị người dùng cuối, giải phóng tài nguyên CPU/GPU cho các tác vụ xử lý ngôn ngữ tự nhiên (NLP) phía sau, đồng thời vẫn đảm bảo độ chính xác về mặt hình thái học cần thiết cho việc nhận diện ngôn ngữ ký hiệu phức tạp. Đây là hướng tiếp cận bền vững, tận dụng sức mạnh của cả hai thế giới: sự tối ưu phần cứng của Apple và sự chuẩn hóa dữ liệu của cộng đồng nghiên cứu AI mở.  
