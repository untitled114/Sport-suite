# ADR-002: Multi-Database PostgreSQL Design

## Status

Accepted

## Date

2025-10-01

## Context

The NBA props system requires storing diverse data:
- Player profiles and game logs (high read, medium write)
- Game schedules and results (medium read, low write)
- Team statistics (low read, low write)
- Prop lines and betting intelligence (high write, high read)

A single database could handle this, but would face:
- Contention between write-heavy prop ingestion and read-heavy feature extraction
- Backup/restore affecting entire system
- Harder to scale individual components

## Decision

Implement a **multi-database PostgreSQL design** with 4 logical databases:

### Database Architecture

| Port | Database | Purpose | Characteristics |
|------|----------|---------|-----------------|
| 5536 | `nba_players` | Player data | Read-heavy, stable schema |
| 5537 | `nba_games` | Game data | Medium read/write |
| 5538 | `nba_team` | Team stats | Low volume |
| 5539 | `nba_intelligence` | Props/betting | High throughput |

### Connection Management
- Each database runs in its own Docker container
- Connections pooled per-database via application layer
- Feature extractor maintains 4 persistent connections

### Data Flow
```
API/Scrapers → intelligence_db (props)
                     ↓
Feature Extractor ← players_db, games_db, team_db
                     ↓
              Predictions → intelligence_db (results)
```

## Consequences

### Positive
- **Isolation**: Prop ingestion doesn't impact feature reads
- **Independent scaling**: Can add replicas to high-load DBs
- **Backup granularity**: Can backup intelligence more frequently
- **Schema evolution**: Changes isolated to single DB

### Negative
- **Complexity**: 4 database connections to manage
- **No cross-DB joins**: Must join in application layer
- **Docker overhead**: 4 containers vs 1 (mitigated by shared memory)
- **Credential management**: 4 sets of credentials (use same defaults)

### Neutral
- Total storage similar to single-DB approach
- Docker Compose simplifies orchestration

## Alternatives Considered

### 1. Single PostgreSQL Database
- Pros: Simpler setup, cross-table joins
- Cons: Lock contention, single point of failure
- Rejected because: Write-heavy prop ingestion would slow reads

### 2. PostgreSQL Schemas (Single DB)
- Pros: Logical separation, allows cross-schema joins
- Cons: Still shares connection pool, single WAL
- Rejected because: Insufficient isolation for our write patterns

### 3. Separate Read/Write Replicas
- Pros: Better read scaling
- Cons: Replication lag, complexity
- Rejected because: Overkill for current scale

### 4. Hybrid PostgreSQL + MongoDB
- Pros: Document store for props, relational for stats
- Cons: Two data platforms to maintain
- Status: MongoDB added later for specific use cases
