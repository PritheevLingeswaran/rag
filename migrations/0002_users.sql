-- 0002_users.sql (Stage 9.6, per the Stage 9.5 decision doc)
-- Google-authenticated web users. Sessions live in Redis (7-day TTL),
-- NOT here. Deliberately NO relationship to query_logs: queries are
-- never stored, and nothing links query content to identity (Stage 8.9
-- invariant, enforced by test).

CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,          -- internal "u_<random>"; NOT the google sub
    google_sub    TEXT UNIQUE NOT NULL,      -- stable Google account id
    email         TEXT NOT NULL,
    display_name  TEXT,
    avatar_url    TEXT,                      -- URL only; the image is never fetched or stored
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
