CREATE TABLE IF NOT EXISTS group_blacklist (
    chat_id BIGINT NOT NULL REFERENCES groups(chat_id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    PRIMARY KEY (chat_id, user_id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON group_blacklist TO dashulya_app;
