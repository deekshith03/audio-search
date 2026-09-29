-- Word timings per sentence, [[word, start_s, end_s], ...], so keyword hits can be tightened to
-- the matched words instead of the whole sentence (a 12 s sentence fails a 1.6 s target).
ALTER TABLE sentences ADD COLUMN words JSONB NOT NULL DEFAULT '[]'::jsonb;
