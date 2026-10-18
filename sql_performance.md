# SQL Performance

## Slow query
Run `EXPLAIN` before the query to see the execution plan. A full table scan on a large table usually means a missing index.

Add an index on columns used in `WHERE`, `JOIN` and `ORDER BY` clauses, for example `CREATE INDEX idx_user_email ON users(email);`.

Avoid `SELECT *`, and select only the columns you need to reduce data transfer.

## Too many connections
This error means the connection pool is exhausted. Close connections after use, use connection pooling, and check for long running idle sessions with `SHOW PROCESSLIST`.

## Deadlocks
Keep transactions short and access tables in the same order everywhere. Retry the transaction when a deadlock error is returned.
