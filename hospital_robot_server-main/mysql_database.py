"""
MySQL 데이터베이스 연결 및 쿼리 실행을 위한 클래스입니다.
작성자: 김승우 (2026-04-23)
설명: MySQL 데이터베이스에 연결하여 쿼리를 실행하는 기능을 제공합니다.
 - MySQLDatabase 클래스: 데이터베이스 연결, 쿼리 실행, 연결 종료 기능 포함
 - 단위 테스트 코드 포함

 + MySQLDatabase 클래스:
   - __init__: 데이터베이스 연결 설정 및 연결 생성
   - execute_querys: SQL 쿼리를 실행하고 결과를 리스트 형태로 반환
   - execute_query: SQL 쿼리를 실행하고 결과를 딕셔너리 형태로 반환
   - close: 데이터베이스 연결 종료  
참고 자료: https://pymysql.readthedocs.io/en/latest/user/examples.html
"""
import pymysql

class MySQLDatabase:
    def __init__(self, host: str, user: str , 
                 password: str, database :str):
        """
        MySQL 데이터베이스에 연결하는 클래스입니다.
        [입력] Host 주소,User 사용자 이름,Password 비밀번호,Database 이름
        [출력] NONE
        """
        
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def execute_querys(self, query: str, params: tuple = None, commit: bool = False) -> list:
        """
        SQL 쿼리를 실행하는 메서드입니다.
        [입력] SQL 쿼리, 파라미터(선택적)
        [출력] 쿼리 결과 -> 리스트 형태로 반환
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            if commit:
                self.connection.commit()
            return result
        
    def execute_query(self, query: str, params: tuple = None, commit: bool = False) -> dict:
        """
        SQL 쿼리를 실행하는 메서드입니다.
        [입력] SQL 쿼리, 파라미터(선택적)
        [출력] 쿼리 결과 -> 딕셔너리 형태로 반환
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if commit:
                self.connection.commit()
            return result
        
    def close(self) -> None:
        """
        데이터베이스 연결을 종료하는 메서드입니다.
        [입력] NONE
        [출력] NONE
        """
        self.connection.close()

    

if __name__ == "__main__":
    # 단위 테스트 코드
    # MySQL 데이터베이스 연결 정보
    host = 'localhost'
    user = 'root'
    password = '1234'
    database = 'exampleDB'

    # MySQLDatabase 클래스 인스턴스 생성
    db = MySQLDatabase(host, user, password, database)
    # 데이터베이스 연결 테스트
    try:
        # 간단한 쿼리 실행 테스트
        result = db.execute_querys("SHOW TABLES;")
        print("Tables in the database:", result)
    except Exception as e:
        print("Error connecting to MySQL:", e)
    finally:
        db.close()