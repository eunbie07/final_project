-- projectdb에 room_layouts 테이블만 추가

USE projectdb;

-- 방 레이아웃 테이블 생성
CREATE TABLE IF NOT EXISTS room_layouts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    layout_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_room_layouts_user_id ON room_layouts(user_id);
CREATE INDEX IF NOT EXISTS idx_room_layouts_created_at ON room_layouts(created_at);