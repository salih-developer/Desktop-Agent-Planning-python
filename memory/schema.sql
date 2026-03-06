-- Conversation metadata and full text store
CREATE TABLE IF NOT EXISTS conversations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input       TEXT    NOT NULL,
    assistant_output TEXT    NOT NULL,
    task_summary     TEXT    NOT NULL DEFAULT '[]',
    metadata         TEXT    NOT NULL DEFAULT '{}',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- sqlite-vec virtual table (768 dims for nomic-embed-text)
-- Embedding = concat of user_input + " " + assistant_output
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    embedding float[768]
);

CREATE INDEX IF NOT EXISTS idx_conversations_created
    ON conversations(created_at DESC);
