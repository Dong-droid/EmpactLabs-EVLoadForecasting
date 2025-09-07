import json
import pandas as pd
import numpy as np
"""
generate_training_data.py
주요 기능:
이 모듈은 전기차 충전소 관련 JSON 데이터를 전처리하여 머신러닝 학습용 CSV 파일로 변환하는 기능을 제공합니다.
주요 기능으로는 충전소 상태 히스토리 데이터 처리(create_history_csv)와 충전소 정보 데이터 처리(create_station_csv)가 있습니다.

"""
def create_history_csv(json_file_path="./data/ChargersStatusHistories.json", 
                      output_path="./history.csv",
                      top_n_stations=400,
                      cutoff_date="2025-02-13",
                      datetime_index=True):
    """
    충전소 상태 히스토리 JSON 파일을 처리하여 history.csv를 생성하는 함수
    
    Parameters:
    -----------
    json_file_path : str
        ChargersStatusHistories.json 파일 경로
    output_path : str
        출력할 CSV 파일 경로
    top_n_stations : int
        상위 n개 충전소만 필터링 (기본값: 400)
    cutoff_date : str
        데이터 시작 날짜 필터 (기본값: "2025-02-13")
    datetime_index : bool
        datetime을 index로 설정할지 여부 (기본값: True)
    
    Returns:
    --------
    pandas.DataFrame
        처리된 데이터프레임
    """
    
    # 1. JSON 파일 불러오기
    print("ChargersStatusHistories.json 파일 로딩 중...")
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # 2. MongoDB ObjectId 중첩 필드 처리
    def flatten(d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict) and "$oid" in value:
                result[key] = value["$oid"]
            else:
                result[key] = value
        return result
    
    flat_data = [flatten(entry) for entry in data]
    
    # 3. DataFrame으로 변환 및 정규화
    print("DataFrame 변환 중...")
    df = pd.json_normalize(flat_data)
    
    # 4. 충전 중인 데이터만 필터링
    df = df[df['status.code'] == "charging"]
    print(f"충전 중인 데이터: {len(df)}개")
    
    # 5. 날짜 컬럼들을 datetime으로 변환
    print("날짜 컬럼 변환 중...")
    date_columns = ['status.updatedAt', 'startedAt', 'endedAt', 
                   'status.lastChargeEndedAt', 'status.currentChargeEndedAt']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 6. endedAt이 결측치가 아닌 데이터만 유지
    df = df[~df['endedAt'].isna()]
    print(f"endedAt 결측치 제거 후: {len(df)}개")
    
    # 7. startedAt으로 정렬
    df = df.sort_values(by="startedAt").reset_index(drop=True)
    
    # 8. 상위 N개 충전소만 필터링
    station_counts = df['_station'].value_counts()
    top_stations = station_counts.head(top_n_stations).index
    df = df[df['_station'].isin(top_stations)]
    print(f"상위 {top_n_stations}개 충전소 필터링 후: {len(df)}개")
    print(f"선택된 충전소 수: {df['_station'].nunique()}개")
    
    # 9. 시작 날짜 필터링
    if cutoff_date:
        df = df[df['startedAt'] >= cutoff_date]
        print(f"날짜 필터링 ({cutoff_date} 이후): {len(df)}개")
    
    # 10. 시간 컬럼 추가
    df['hour'] = df['startedAt'].dt.floor('H')
    
    # 11. datetime을 index로 설정 (옵션)
    if datetime_index:
        df.set_index('startedAt', inplace=True)
        print("startedAt을 index로 설정")
    
    # 12. CSV 파일로 저장
    df.to_csv(output_path, index=datetime_index)
    print(f"히스토리 파일 저장 완료: {output_path}")
    
    # 13. 기본 통계 정보 출력
    print("\n=== 히스토리 데이터 요약 ===")
    print(f"전체 레코드 수: {len(df)}")
    print(f"충전소 수: {df['_station'].nunique()}")
    print(f"충전기 수: {df['_charger'].nunique()}")
    print(f"데이터 기간: {df.index.min() if datetime_index else df['startedAt'].min()} ~ {df.index.max() if datetime_index else df['startedAt'].max()}")
    
    return df


def create_station_csv(json_file_path="./data/Stations.json",
                      output_path="./station.csv",
                      target_city="Gyeonggi",
                      target_district="성남시"):
    """
    충전소 정보 JSON 파일을 처리하여 station.csv를 생성하는 함수
    
    Parameters:
    -----------
    json_file_path : str
        Stations.json 파일 경로
    output_path : str
        출력할 CSV 파일 경로
    target_city : str
        대상 도시 (기본값: "Gyeonggi")
    target_district : str
        대상 구/시 (기본값: "성남시")
    
    Returns:
    --------
    pandas.DataFrame
        처리된 데이터프레임
    """
    
    print("=== 충전소 정보 처리 시작 ===")
    
    # 1. Stations.json 파일 불러오기
    print("Stations.json 파일 로딩 중...")
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # 2. MongoDB ObjectId 중첩 필드 처리
    def flatten(d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict) and "$oid" in value:
                result[key] = value["$oid"]
            else:
                result[key] = value
        return result
    
    flat_data = [flatten(entry) for entry in data]
    
    # 3. DataFrame으로 변환 및 정규화
    print("DataFrame 변환 중...")
    df = pd.json_normalize(flat_data)
    print(f"전체 충전소 수: {len(df)}")
    
    # 4. 한국 지역명 영어 매핑
    korea_regions_english = {
        '서울특별시': 'Seoul',
        '부산광역시': 'Busan',
        '대구광역시': 'Daegu',
        '인천광역시': 'Incheon',
        '광주광역시': 'Gwangju',
        '대전광역시': 'Daejeon',
        '울산광역시': 'Ulsan',
        '세종특별자치시': 'Sejong',
        '경기도': 'Gyeonggi',
        '강원특별자치도': 'Gangwon',
        '충청북도': 'Chungcheongbuk-do',
        '충청남도': 'Chungcheongnam-do',
        '전라북도': 'Jeollabuk-do',
        '전라남도': 'Jeollanam-do',
        '경상북도': 'Gyeongsangbuk-do',
        '경상남도': 'Gyeongsangnam-do',
        '제주특별자치도': 'Jeju'
    }
    
    # 5. 주소 정리 - 결측치 처리 및 오타 수정
    address_corrections = {
        '더 스탈릿': '인천광역시 연수구',
        '소노캄 고양': '경기도 고양시',
        '연꽃참한우': '경상북도 상주시',
        '강화씨사이드리조트': '인천광역시 강화군',
        '논산 라온빌리지': '충청남도 논산시',
        '대덕패션타운': '대전광역시 신탄진',
        '남제천IC 휴게소': '충청북도 제천시',
        '강남N타워': '서울특별시 강남구',
        '스타필드 하남': '경기도 하남시',
        '장흥정남진물과학관': '전라남도 장흥군',
        '부산 서비스': '부산광역시 해운대구',
        'N2WASH 일산전기차충전소': '경기도 고양시',
        '홀리데이인 광주호텔': '광주광역시 서구',
        '엑스코': '대구광역시 북구',
        '롯데월드타워': '서울특별시 송파구',
        '1413 Nakdongnam-ro': '부산광역시 사하구',
        '74-4 Saemaeul-ro': '경상북도 구미시',
        '인제스피디움': '강원도 인제군',
        'Yeonyang-dong': '경기도 여주시 여양동',
        '동부산점': '부산광역시 기장군',
        '구리남양주점': '경기도 구리시',
        '충주점': '충청북도 충주시',
        '이안스퀘어': '경기도 파주시',
        'Yeongweol': '강원도 영월군',
        '청라': '인천광역시 서구',
        '인천': '인천광역시',
        '천안': '충청남도 천안시',
        '경주': '경상북도 경주시',
        '기흥점': '경기도 용인시 기흥구',
        '센텀시티점': '부산광역시 해운대구',
        '대구수성아트스퀘어': '대구광역시 수성구',
        '부여점': '충청남도 부여군',
        '광교점': '경기도 수원시 영통구',
        '천안중앙교회': '충청남도 천안시',
        'OneMount': '경기도 고양시 일산서구',
        '롯데몰수원점': '경기도 수원시 권선구',
        '트리플스트리트': '인천광역시 연수구',
        '호수공원 외식타운': '경기도 고양시 일산동구',
        '임실치즈테마파크': '전라북도 임실군',
        '라카이샌드파인': '강원도 강릉시',
        '함양IC점': '경상남도 함양군',
        '마리오아울렛 1관': '서울특별시 금천구',
        '동탄점': '경기도 화성시 동탄',
        '잠실점': '서울특별시 송파구',
        '판교 테크원타워': '경기도 성남시 분당구',
        '타임빌라스': '경기도 의왕시',
        '호텔농심': '부산광역시 동래구',
        '양산': '경상남도 양산시',
        '평촌': '경기도 안양시 동안구',
        '파라다이스호텔 부산': '부산광역시 해운대구',
        '오크밸리 리조트': '강원도 원주시',
        '동해웰빙레포츠타운': '강원도 동해시',
        '김포공항점': '서울특별시 강서구',
        '진주점': '경상남도 진주시',
        'IFC 부산몰': '부산광역시 해운대구',
        '여주프리미엄아울렛': '경기도 여주시',
        '고양-행주': '경기도 고양시 덕양구',
        '세종-JB': '세종특별자치시',
        '안동노리카페': '경상북도 안동시',
        'Eumseong A': '충청북도 음성군',
        '57 Simgok-ro': '충청북도 음성군',
        '367-19 Illakgol-gil': '부산광역시 기장군',
        '606 Geumil-ro': '충청북도 진천군',
        '평창 휘닉스': '강원도 평창군',
        '414 Yeonyang-dong': '경상북도 영양군',
        '롯데프리미엄아울렛 동부산점': '부산광역시 기장군',
        '모다아울렛 구리남양주점': '경기도 남양주시',
        '모다아울렛 충주점': '충청북도 충주시',
        '엘리웨이 인천': '인천광역시 미추홀구',
        '소노벨 천안': '충청남도 천안시',
        '16 덕안로': '경기도 광명시',
        '라한셀렉트 경주': '경상북도 경주시',
        '나인블럭 중대동점': '경기도 광주시',
        '몰오브효자': '전라북도 전주시',
        '17 Wangsimnigwangjang-ro': '서울특별시 성동구',
        '롯데프리미엄아울렛 기흥점': '경기도 용인시',
        '커피명당': '경기도 고양시'
    }
    
    # 주소 정정 적용
    print("주소 데이터 정리 중...")
    for old_addr, new_addr in address_corrections.items():
        if old_addr in df["location.address"].values:
            df.loc[df["location.address"] == old_addr, 'location.address'] = new_addr
    
    # 6. 도시와 구/군 정보 추출
    city = []
    district = []
    
    for address in df['location.address']:
        addr_parts = address.split(' ')
        city.append(addr_parts[0] if len(addr_parts) > 0 else '')
        district.append(addr_parts[1] if len(addr_parts) > 1 else '')
    
    df['city'] = city
    df['district'] = district
    
    # 7. 도시명을 영어로 변환
    df['city'] = df['city'].map(korea_regions_english)
    
    # 8. 특정 도시/구 필터링
    if target_city and target_district:
        df = df[(df.city == target_city) & (df.district == target_district)].copy()
        print(f"{target_city} {target_district} 충전소 필터링 후: {len(df)}개")
    elif target_city:
        df = df[df.city == target_city].copy()
        print(f"{target_city} 충전소 필터링 후: {len(df)}개")
    
    # 9. station.csv용 컬럼 선택 및 저장
    station_columns = ["_id.$oid", "alternateId", "location.latitude", 
                      "location.longitude", "limit.status", "isParkingFree"]
    
    # 컬럼이 존재하는지 확인
    available_columns = [col for col in station_columns if col in df.columns]
    if len(available_columns) != len(station_columns):
        missing = set(station_columns) - set(available_columns)
        print(f"경고: 일부 컬럼이 존재하지 않습니다: {missing}")
    
    output_df = df[available_columns].copy()
    
    # statId 컬럼 추가 (history 데이터와 조인용)
    output_df['statId'] = output_df["_id.$oid"]
    
    # 10. CSV 파일로 저장
    output_df.to_csv(output_path, index=False)
    print(f"충전소 정보 저장 완료: {output_path}")
    
    # 11. 기본 통계 정보 출력
    print("\n=== 충전소 데이터 요약 ===")
    print(f"충전소 수: {len(output_df)}")
    if 'isParkingFree' in output_df.columns:
        print(f"주차비 무료 충전소: {output_df['isParkingFree'].sum()}개")
    if 'limit.status' in output_df.columns:
        print(f"이용 제한 상태별 분포:")
        print(output_df['limit.status'].value_counts())
    
    return output_df


# 사용 예제
if __name__ == "__main__":
    # 1. 충전소 정보 생성 (경기도 성남시)
    station_df = create_station_csv()
    
    # 2. 충전 히스토리 생성 (datetime index 포함)
    history_df = create_history_csv()
    
    print(f"\nStation DataFrame shape: {station_df.shape}")
    print(f"History DataFrame shape: {history_df.shape}")
    
    # 커스텀 설정 예제
    station_df = create_station_csv(
        json_file_path="./data/Stations.json",
        output_path="./custom_station.csv",
        target_city="Seoul",
        target_district="강남구"
    )
    
    history_df = create_history_csv(
        json_file_path="./data/ChargersStatusHistories.json",
        output_path="./custom_history.csv",
        top_n_stations=200,
        cutoff_date="2025-01-01",
        datetime_index=True
    )
