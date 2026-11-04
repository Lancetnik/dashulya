ALTER TABLE groups
    ADD COLUMN IF NOT EXISTS reply_audience TEXT NOT NULL DEFAULT 'all'
        CHECK (reply_audience IN ('all', 'admins')),
    ADD COLUMN IF NOT EXISTS whisper_audience TEXT NOT NULL DEFAULT 'all'
        CHECK (whisper_audience IN ('all', 'admins', 'disabled'));
