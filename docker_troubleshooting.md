# Docker Troubleshooting

## Container keeps restarting
Run `docker logs <container>` to read the last error. A restart loop usually means the main process crashes on startup, often because of a missing environment variable or a wrong command.

Check the exit code with `docker inspect <container> --format '{{.State.ExitCode}}'`. Exit code 137 means the container was killed for running out of memory.

Check the restart policy: `restart: always` will hide crashes by restarting forever, so switch to `on-failure` while debugging.

## Port already in use
If Docker says the port is allocated, find the process with `lsof -i :8080` and stop it, or map a different host port such as `-p 8081:8080`.

## Cleaning up disk space
Run `docker system prune -a` to remove unused images, stopped containers and build cache. Use `docker system df` first to see what is taking space.
