# IT Helpdesk Common Issues

## Forgot password, reset password or account locked (cannot login)
Use the self-service reset page or ask the helpdesk to unlock the account. Accounts usually lock after 5 failed login attempts and unlock automatically after 30 minutes.

Choose a new password with at least 12 characters, mixing letters, numbers and symbols. Do not reuse old passwords.

## VPN will not connect
Check that your internet connection works first. Then restart the VPN client and try again.

If it still fails, confirm your password has not expired, make sure the date and time on your computer are correct, and try a different network such as a mobile hotspot.

## Printer is not printing (printer not working, print stuck)
Check that the printer is powered on and connected to the same network. Open the print queue, cancel any stuck jobs, and print again.

If the job still fails, restart the Print Spooler service by running `net stop spooler` and then `net start spooler` in an administrator command prompt.

## Wi-Fi connected but no internet (wifi, network not working)
Forget the Wi-Fi network and reconnect. Run `ipconfig /release` followed by `ipconfig /renew` to get a new IP address, and `ipconfig /flushdns` to clear the DNS cache.

If other devices work fine on the same Wi-Fi, restart your computer or update the network adapter driver.

## Outlook is not sending or receiving email
Check that Outlook is not in Offline mode and that your mailbox is not full. Restart Outlook and test again.

If the problem continues, create a new Outlook profile from the Control Panel Mail settings, or run Outlook in safe mode with `outlook.exe /safe` to rule out faulty add-ins.

## Laptop or computer is running slow (PC slow, lagging, hanging)
Open Task Manager with Ctrl + Shift + Esc and check which process uses the most CPU or memory. Close apps you do not need and restart the computer.

Free disk space by running Disk Cleanup, uninstall unused programs, and disable unnecessary startup apps in Task Manager.

## Blue screen or computer keeps restarting
Note the stop code shown on the blue screen. Boot into safe mode and uninstall any recently installed driver or update.

Run `sfc /scannow` in an administrator command prompt to repair corrupted system files.

## Software installation request
Raise a ticket with the software name, version and business reason. Installation needs manager approval and administrator rights, so users cannot install software themselves.
