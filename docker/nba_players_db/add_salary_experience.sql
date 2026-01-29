-- Add salary and experience columns to player_profile
-- These are high-value fields for NBA betting models

ALTER TABLE player_profile
ADD COLUMN IF NOT EXISTS salary_annual INTEGER,  -- Annual salary in dollars
ADD COLUMN IF NOT EXISTS years_experience INTEGER;  -- Years in NBA

-- Add indexes for filtering by salary/experience
CREATE INDEX IF NOT EXISTS idx_salary ON player_profile(salary_annual);
CREATE INDEX IF NOT EXISTS idx_experience ON player_profile(years_experience);

-- Add comments
COMMENT ON COLUMN player_profile.salary_annual IS 'Annual salary in USD (e.g., 53142264 for $53.1M)';
COMMENT ON COLUMN player_profile.years_experience IS 'Years of NBA experience (0 = rookie)';
