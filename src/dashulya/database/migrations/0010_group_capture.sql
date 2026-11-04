ALTER TABLE groups
    ADD COLUMN IF NOT EXISTS capture_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- Only currently active groups have verifiable owner approval on upgrade.
UPDATE groups SET capture_enabled=TRUE WHERE active;
