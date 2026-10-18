# Networking Basics

## Check if a port is open
Use `nc -zv <host> <port>` or `telnet <host> <port>` to test whether a port is reachable. On the server, `ss -tulpn` lists all listening ports and the processes using them.

## DNS issues
Run `nslookup <domain>` or `dig <domain>` to check name resolution. If it fails, check `/etc/resolv.conf` and try a public resolver such as 8.8.8.8.

## Website not reachable
Test with `ping <host>` and `curl -I https://<host>`. If ping works but curl fails, the problem is usually the firewall, the web server being down, or a wrong port.
