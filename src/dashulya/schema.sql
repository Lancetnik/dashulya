CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE TABLE IF NOT EXISTS groups (
    chat_id BIGINT PRIMARY KEY, title TEXT NOT NULL, username TEXT,
    active BOOLEAN NOT NULL DEFAULT FALSE
);
ALTER TABLE groups ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS collections (
    collection TEXT PRIMARY KEY, chat_id BIGINT UNIQUE,
    title TEXT NOT NULL, username TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    collection TEXT NOT NULL REFERENCES collections(collection),
    message_id BIGINT NOT NULL, chat_id BIGINT,
    body TEXT NOT NULL, sent_at TIMESTAMPTZ,
    author_id BIGINT, author TEXT NOT NULL, author_username TEXT, is_bot BOOLEAN,
    chat_title TEXT NOT NULL, chat_username TEXT,
    topic_id BIGINT, reply_to_id BIGINT, edited_at TIMESTAMPTZ, media_type TEXT,
    url TEXT, provenance TEXT NOT NULL, archive_key TEXT,
    update_id BIGINT NOT NULL DEFAULT 0, version_time DOUBLE PRECISION NOT NULL,
    revision BIGINT NOT NULL DEFAULT 1, indexed_revision BIGINT NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0, retry_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    body_folded TEXT NOT NULL, author_folded TEXT NOT NULL, username_folded TEXT,
    PRIMARY KEY (collection, message_id)
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages(chat_id, topic_id, message_id);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS image_file JSONB;
CREATE INDEX IF NOT EXISTS messages_time ON messages(collection, sent_at, message_id);
CREATE INDEX IF NOT EXISTS messages_pending ON messages(retry_at)
    WHERE indexed_revision < revision;

CREATE TABLE IF NOT EXISTS embeddings (
    cache_key TEXT PRIMARY KEY, embedding vector(512) NOT NULL
);

CREATE TABLE IF NOT EXISTS lexical_documents (
    document_key TEXT PRIMARY KEY, word_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lexical_terms (
    document_key TEXT NOT NULL REFERENCES lexical_documents(document_key) ON DELETE CASCADE,
    term TEXT NOT NULL, frequency INTEGER NOT NULL,
    term_key TEXT NOT NULL,
    PRIMARY KEY (document_key, term_key)
);
ALTER TABLE lexical_terms ADD COLUMN IF NOT EXISTS term_key TEXT;
UPDATE lexical_terms SET term_key=encode(sha256(convert_to(term,'UTF8')),'hex') WHERE term_key IS NULL;
ALTER TABLE lexical_terms ALTER COLUMN term_key SET NOT NULL;
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='lexical_terms'::regclass AND conname='lexical_terms_pkey' AND pg_get_constraintdef(oid) LIKE '%document_key, term)%') THEN
        ALTER TABLE lexical_terms DROP CONSTRAINT lexical_terms_pkey;
        ALTER TABLE lexical_terms ADD PRIMARY KEY(document_key,term_key);
    END IF;
END $$;
DROP INDEX IF EXISTS lexical_prefix;
CREATE INDEX IF NOT EXISTS lexical_short_prefix ON lexical_terms(term text_pattern_ops)
    WHERE octet_length(term)<2000;

CREATE TABLE IF NOT EXISTS source_snapshots (
    source_key TEXT PRIMARY KEY, metadata JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS legacy_records (
    collection TEXT NOT NULL REFERENCES collections(collection),
    record_key TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL,
    source_message_ids BIGINT[] NOT NULL,
    PRIMARY KEY (collection, record_key)
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='collections'::regclass AND conname='collections_identity') THEN
        ALTER TABLE collections ADD CONSTRAINT collections_identity CHECK (
            (chat_id IS NULL AND collection LIKE 'archive:%') OR
            (chat_id IS NOT NULL AND collection='telegram:' || chat_id::text)
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='messages'::regclass AND conname='messages_identity') THEN
        ALTER TABLE messages ADD CONSTRAINT messages_identity CHECK (
            (chat_id IS NULL AND collection LIKE 'archive:%') OR
            (chat_id IS NOT NULL AND collection='telegram:' || chat_id::text)
        );
    END IF;
END $$;
