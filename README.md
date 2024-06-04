test_240603_rsw.py : 가장 최신 구동 파일
logs.txt : samples/GenICamParameters.py 이용, SDK 이용해 얻을 수 있는 모든 파라미터 목록
- 형식 : Category:Parameter_name, Type: Value

* 240603 수정사항*
1. waterfall.py 참고해 waterfall 나오도록 변경
    - extract_band_image_3 기반으로 재작성
2. SDK 관련 모든 함수에 type hint 적용
3. output 파일들이 outputs 폴더에 저장되도록 함 (hdr, npy)
4. envi header에서 식별이 불가능한 몇몇 변수를 제외하고 sample.hdr와 동일한 형태가 되도록 작성, 알 수 없는 값은 "?"로 초기화
5. wavelegnth, fwhm 상수들은 constants.py 파일에 모아두었음.
6. 처음에 0 눌러서 (device selection) 시작하는 작업 없앰. # 만일 되돌릴려면 lib/PvSampleUtils.py의 96 line uncomment, 97 line comment하면 됌.
7. .raw 데이터로 저장 -> .npy로 변경
