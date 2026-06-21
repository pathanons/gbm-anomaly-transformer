# รายงานเปรียบเทียบผลการทดลอง Phase 1: Legacy เทียบกับ Refactored

## บทสรุปสำหรับอาจารย์ที่ปรึกษา

รายงานฉบับนี้เปรียบเทียบผลการทดลอง Phase 1 ระหว่างวิธีให้คะแนนแบบเดิม (legacy) และแบบปรับปรุง (refactored) ภายใต้โครงสร้าง joint GBM เดียวกัน

แม้ฝั่ง refactored จะมีค่า ROC-AUC และ PR-AUC รวมสูงขึ้นเล็กน้อย แต่เมื่อประเมินระดับราย ticker พบว่า score ของฝั่ง refactored มีลักษณะแกว่งมากกว่า (noise สูงกว่า) และสอดคล้องกับช่วงที่ราคาเปลี่ยนแรงได้น้อยกว่าฝั่ง legacy ในหุ้นส่วนใหญ่ จึงสรุปเชิงปฏิบัติว่า ณ ปัจจุบัน legacy เหมาะเป็นค่าตั้งต้นที่ปลอดภัยกว่า

## การตั้งค่าการทดลอง

- โมเดลที่ใช้: joint GBM pipeline (สถาปัตยกรรมเดียวกัน และนโยบาย split เดียวกัน)
- การเปรียบเทียบ: phase1_legacy เทียบกับ phase1_refactored
- จำนวนหน้าต่างประเมินผล: 102,855 windows ต่อการทดลอง
- วัตถุประสงค์ของเฟสนี้: เปรียบเทียบพฤติกรรม score ก่อน thresholding

## ตารางที่ 1 ผลรวมระดับการทดลอง

| ตัวชี้วัด | phase1_legacy | phase1_refactored |
|---|---:|---:|
| โหมด score | legacy | refactored |
| จำนวน windows | 102,855 | 102,855 |
| ค่าเฉลี่ย score | 26.3372 | 27.1100 |
| ส่วนเบี่ยงเบนมาตรฐาน score | 372.2335 | 376.1500 |
| score p95 | -0.0759 | 1.3585 |
| score p99 | 45.2036 | 62.0307 |
| ROC-AUC จาก score | 0.5645 | 0.5964 |
| PR-AUC จาก score | 0.9909 | 0.9922 |

## ตารางที่ 2 การเปรียบเทียบรูปทรงเส้น score ราย ticker (107 Tickers)

| ตัวชี้วัด | ค่า |
|---|---:|
| ค่าเฉลี่ย correlation ของเส้น score legacy กับ refactored | 0.8589 |
| มัธยฐาน correlation ของเส้น score legacy กับ refactored | 0.9253 |
| ค่า correlation ต่ำสุด | -0.3569 |
| จำนวน ticker ที่ correlation < 0.7 | 9 |
| มัธยฐานการซ้อนทับของยอด top-1% (Jaccard) | 0.2727 |
| จำนวน ticker ที่ top-1% overlap เท่ากับ 0 | 22 |

ข้อสังเกต:
- หุ้นส่วนใหญ่ยังเคลื่อนไปทิศทางคล้ายกันในภาพรวม
- อย่างไรก็ตาม การซ้อนทับของยอดสูงสุดยังจำกัด และมีหลาย ticker ที่รูปทรงเปลี่ยนอย่างมีนัยสำคัญ ไม่ใช่แค่เปลี่ยนสเกล

## ตารางที่ 3 การวินิจฉัย Noise เทียบกับ Signal (107 Tickers)

นิยามตัวชี้วัด:
- Roughness: std(diff(score)) / std(score), ยิ่งต่ำยิ่งเรียบ
- Calm jitter: ความแกว่งของ score ในช่วงตลาดนิ่ง, ยิ่งต่ำยิ่งดี
- Separation z-score: ความสามารถแยกช่วงที่ |return| สูงออกจากช่วงปกติ, ยิ่งสูงยิ่งดี
- Spearman(|return|, score): ความสอดคล้องกับขนาดการเปลี่ยนราคา, ยิ่งสูงยิ่งดี

| ตัวชี้วัด | จำนวน / มัธยฐาน |
|---|---:|
| refactored rougher กว่า legacy | 90 tickers |
| refactored เรียบกว่า legacy | 17 tickers |
| refactored มี calm jitter สูงกว่า | 80 tickers |
| refactored แยก separation z-score ดีกว่า | 26 tickers |
| refactored แยก separation z-score แย่กว่า | 81 tickers |
| refactored ให้ Spearman(|return|, score) ดีกว่า | 23 tickers |
| มัธยฐาน delta roughness (ref - legacy) | +0.0409 |
| มัธยฐาน delta calm jitter (ref - legacy) | +0.0102 |
| มัธยฐาน delta separation z-score (ref - legacy) | -0.0863 |
| มัธยฐาน delta Spearman (ref - legacy) | -0.0298 |

ข้อสังเกต:
- ฝั่ง refactored มี noise สูงกว่าใน ticker ส่วนใหญ่
- ความสามารถในการแยกช่วงที่ราคาเคลื่อนไหวแรงอ่อนลงใน ticker ส่วนใหญ่
- ความสอดคล้องกับขนาดการเคลื่อนไหวของราคาลดลงใน ticker ส่วนใหญ่

## ข้อสรุปเชิงที่ปรึกษา

ในสถานะปัจจุบันของ Phase 1 หลักฐานระดับราย ticker สนับสนุนว่า legacy มีเสถียรภาพและความชัดเจนบริเวณช่วงราคาเปลี่ยนแรงดีกว่าในหุ้นส่วนใหญ่

ข้อเสนอแนะ:
1. ใช้ legacy เป็น baseline หลักสำหรับรอบถัดไป
2. คง refactored ไว้ในสถานะ exploratory branch ยังไม่แทนของเดิม
3. หากต้องการพัฒนา refactored ต่อ ให้ทำ ablation แบบจำกัดขอบเขตก่อน (เช่น weight tuning และ smoothing diagnostics ภายใต้นโยบาย validation-only) แล้วจึงค่อยเข้าสู่ thresholding

## แผนดำเนินงานถัดไปแบบเป็นเฟส

### Phase 2: Stabilization ของ Refactored (ยังไม่ threshold)
1. แยกสาเหตุว่าความต่างเกิดจากสูตร score ใหม่ หรือเกิดจากผลของการเทรนโมเดลใหม่
2. ทดลองปรับน้ำหนัก divergence แบบช่วงแคบ และวัด noise-signal metric เดิมซ้ำ
3. ทดลอง smoothing เฉพาะ score ที่ใช้วิเคราะห์ โดยไม่เปลี่ยน labeling policy

ผลลัพธ์ที่คาดหวัง:
- roughness และ calm jitter ลดลงโดยไม่ทำให้ separation z-score และ Spearman แย่ลง

### Phase 3: Selection Gate ก่อนเข้า Thresholding
1. กำหนดเกณฑ์ผ่านที่ชัดเจนในระดับ ticker (เช่น จำนวน ticker ที่ชนะ legacy ต้องเกินเกณฑ์)
2. ทำรายงาน pass/fail ต่อ ticker และสรุปผลรวมเพื่อคัดเลือกสูตรสุดท้าย

ผลลัพธ์ที่คาดหวัง:
- ได้ข้อสรุปเชิงหลักฐานว่าจะคง legacy หรือยกระดับ refactored เป็นตัวหลัก

### Phase 4: Thresholding และ Event-Level Evaluation (ทำเมื่อผ่านเฟสก่อนหน้า)
1. เริ่ม threshold calibration บน validation เท่านั้น
2. ประเมิน event-level recall, delay, false alarms per year
3. เปรียบเทียบผลกับ baseline เดิมภายใต้ budget เทียบเท่า

ผลลัพธ์ที่คาดหวัง:
- ได้ผลประเมินเชิงใช้งานจริงโดยไม่ละเมิดหลักวิธีวิทยา

## รูปประกอบที่จะแนบให้พิจารณา

เพื่อประกอบการพิจารณา แนะนำให้แนบทั้งชุดตัวอย่างและชุดเต็ม ดังนี้

1. ตัวอย่างกราฟที่ความต่างชัดเจน (sample set):
   - ADBE, DE, LIN, TMO, BKNG, MSFT
2. ชุดเต็มราย ticker:
   - กราฟเปรียบเทียบ legacy เทียบ refactored ครบ 107 ticker จากโฟลเดอร์ผลการทดลอง

## หมายเหตุ

- รายงานนี้เป็นการประเมินก่อน thresholding และเน้นพฤติกรรมของ score shape
- ไม่มีการใช้ test set เพื่อปรับ threshold สำหรับการตัดสินโมเดล
