Xinbao board bundle

Recommended copy/install flow on Orange Pi:

1. Copy xinbao_board_bundle.tar.gz from the USB drive or SD boot partition to the home directory.

   mkdir -p /home/HwHiWiUser/xinbao
   tar -xzf xinbao_board_bundle.tar.gz -C /home/HwHiWiUser
   cd /home/HwHiWiUser/xinbao

2. If the archive was unpacked from a Windows-created drive, make scripts executable:

   chmod +x setup_pi.sh run.sh install_service.sh

3. Install runtime dependencies:

   bash setup_pi.sh

4. Start the service:

   bash run.sh start
   bash run.sh status
   tail -f xinbao.log

5. Open on the board itself:

   http://127.0.0.1:5000

If another computer cannot open http://<board-ip>:5000, it is probably campus network client isolation.
Use the browser on the Orange Pi desktop, a phone hotspot without isolation, Ethernet, or USB tethering.
