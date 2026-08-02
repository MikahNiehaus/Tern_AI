-- Shared vector store for both corpora (SPEC.md Part 2). One vec0 virtual
-- table holds every embedding regardless of type; a plain metadata table
-- joined by rowid holds the type, title, and text. Plain float32, not
-- quantized (decided in SPEC.md: quantized mode broke the distance <=
-- threshold filter in a real reported sqlite-vec issue, and size is not a
-- real constraint here).
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[384]);

CREATE TABLE IF NOT EXISTS metadata (
    rowid INTEGER PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('fact', 'tool')),
    title TEXT,
    content TEXT NOT NULL
);
