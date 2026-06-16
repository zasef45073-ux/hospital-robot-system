"""
쉽게 퀴리문을 작성하면서 생성 및 수정, 삭제, 조회 기능을 구현할 수 있도록
작성자 : 김승우
설명 : MySQL Database에 연결된 상태에서 테이블 변경을 한다.
mysql_database.py을 상속 받아서 CRUD 기능을 구현
MySQL_Execute 
    - create_table 함수: MySQL 데이터베이스에 테이블을 생성하는 기능
    - insert_data 함수: MySQL 데이터베이스에 데이터를 삽입하는 기능
    - select_data 함수: MySQL 데이터베이스에서 데이터를 조회하는 기능
    - update_data 함수: MySQL 데이터베이스에서 데이터를 수정하는 기능
    - delete_data 함수: MySQL 데이터베이스에서 데이터를 삭제하는 기능
"""
from mysql_database import MySQLDatabase as mysql_db

class MySQL_Execute(mysql_db):

    def __init__(self, host: str, user: str, password: str, database: str) -> None:
        """
        MySQL 데이터베이스에 연결하는 생성자입니다.
         - MySQLDatabase 클래스의 생성자를 호출하여 데이터베이스에 연결합니다.
        [입력] Host 주소,User 사용자 이름,Password 비밀번호,Database 이름
        [출력] NONE
        """
        super().__init__(host, user, password, database)

    def create_table(self, table_name: str, columns: str) -> None:
        """
        MySQL 데이터베이스에 테이블을 생성하는 메서드입니다.
        [입력] 테이블 이름, 컬럼 정의 (예: "id INT PRIMARY KEY, name VARCHAR(255)")
        [출력] NONE
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute_querys(query, commit=True)

    def insert_data(self, table_name: str, data: dict) -> None:
        """
        MySQL 데이터베이스에 데이터를 삽입하는 메서드입니다.
        [입력] 테이블 이름, 삽입할 데이터 (딕셔너리 형태)
        [출력] NONE
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.execute_querys(query, params=tuple(data.values()), commit=True)

    def select_data(self, table_name: str, columns: str = '*', where: str = None, once: bool = False) -> list:
        """
        MySQL 데이터베이스에서 데이터를 조회하는 메서드입니다.
        [입력] 테이블 이름, 조회할 컬럼 (기본값: '*'), 조건절 (예: "id = 1"), 단일 조회 여부 (기본값: False) 
        [출력] 조회 결과 -> 리스트 형태로 반환
        """
        query = f"SELECT {columns} FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        if once:
            return self.execute_query(query)
        else:
            return self.execute_querys(query)
    
    def update_data(self, table_name: str, data: dict, where: str) -> None:
        """
        MySQL 데이터베이스에서 데이터를 수정하는 메서드입니다.
        [입력] 테이블 이름, 수정할 데이터 (딕셔너리 형태), 조건절 (예: "id = 1")
        [출력] NONE
        """
        set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where}"
        self.execute_querys(query, params=tuple(data.values()), commit=True)

    def delete_data(self, table_name: str, where: str) -> None:
        """
        MySQL 데이터베이스에서 데이터를 삭제하는 메서드입니다.
        [입력] 테이블 이름, 조건절 (예: "id = 1")
        [출력] NONE
        """
        query = f"DELETE FROM {table_name} WHERE {where}"
        self.execute_querys(query, commit=True)

if __name__ == "__main__":
    # 단위 테스트 코드
    # MySQL 데이터베이스 연결 정보
    host = 'localhost'
    user = 'root'
    password = '1234'
    database = 'hospital_robot_db'
    mysql_exec = MySQL_Execute(host, user, password, database)
    # 테이블 생성 테스트
    mysql_exec.create_table('test_table', 'id INT PRIMARY KEY AUTO_INCREMENT, name VARCHAR(255), age INT')
    # 데이터 삽입 테스트
    mysql_exec.insert_data('test_table', {'name': 'Alice', 'age': 30})
    mysql_exec.insert_data('users', {'username': 'Erick', 'password': '1234'})
    # 데이터 조회 테스트
    results = mysql_exec.select_data(table_name='users', where="username='Erick'")
    print("조회 결과:", results)
    # 데이터 수정 테스트
    mysql_exec.update_data('test_table', {'age': 31}, "name = 'Alice'")
    # 데이터 삭제 테스트
    # mysql_exec.delete_data('test_table', "name = 'Bob'")
    _results = mysql_exec.select_data('test_table')
    print("조회 결과:", _results) 
    mysql_exec.delete_data('test_table', "name = 'Alice'")
    # mysql_exec.delete_data('users', "username = 'Erick'")
    # 최종 조회 결과
    final_results = mysql_exec.select_data('test_table')
    print("최종 조회 결과:", final_results) 