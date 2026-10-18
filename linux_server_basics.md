# Linux Server Basics

## Disk is full
Run `df -h` to see which filesystem is full. Then run `du -ah / 2>/dev/null | sort -rh | head -20` to list the largest files and folders.

Old logs in `/var/log` are a common cause. Rotate or compress them with `logrotate`, and clear package caches with `apt clean`.

## High CPU or memory usage
Use `top` or `htop` to find the process using the most resources. Use `ps aux --sort=-%mem | head` to list top memory consumers.

If a service is stuck, restart it with `systemctl restart <service>` and check `journalctl -u <service> -n 50` for errors.

## Service will not start
Run `systemctl status <service>` and `journalctl -xe` to see the failure reason. Verify the config file syntax and file permissions before restarting.
