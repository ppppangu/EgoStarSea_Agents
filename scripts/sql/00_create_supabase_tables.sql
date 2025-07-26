-- SQL schema for Supabase conversation history storage
-- Run these statements inside the Supabase dashboard or via `supabase db push`

-- -------------------------
-- Sessions table
-- -------------------------
CREATE TABLE IF NOT EXISTS public.sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, session_id)
);

-- -------------------------
-- Messages table
-- -------------------------
CREATE TABLE IF NOT EXISTS public.messages (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chat_id TEXT,
    CONSTRAINT fk_session FOREIGN KEY (user_id, session_id)
        REFERENCES public.sessions(user_id, session_id)
        ON DELETE CASCADE
);

-- Index to speed up chronological reads for a session.
CREATE INDEX IF NOT EXISTS idx_messages_user_session_time
    ON public.messages (user_id, session_id, timestamp);

-- (Optional) Row level security policies can be added here to restrict data
-- access on a per-user basis when Supabase Auth is enabled. 