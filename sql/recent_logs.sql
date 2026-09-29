-- MySQL Workbench에서 필요한 SELECT 문을 선택해 실행하세요.
-- log_id가 클수록 나중에 저장된 행입니다.
SELECT * FROM testdb.action_logs ORDER BY log_id DESC LIMIT 1000;

SELECT * FROM testdb.failure_logs ORDER BY log_id DESC LIMIT 1000;

SELECT * FROM testdb.error_logs ORDER BY log_id DESC LIMIT 1000;
