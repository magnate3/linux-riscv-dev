all:	test test.exe

test:	test.c
	gcc -o $@ $< -I/usr/include -L/usr/lib -luv

udp_cli.exe:  udp_cli.c
	x86_64-w64-mingw32-gcc -static -static-libgcc -o $@ $< -I/usr/x86_64-w64-mingw32/include -L/usr/x86_64-w64-mingw32/lib  -lws2_32 -lpsapi -liphlpapi
tcp_cli.exe:  tcp_cli.c
	x86_64-w64-mingw32-gcc -static -static-libgcc -o $@ $< -I/usr/x86_64-w64-mingw32/include -L/usr/x86_64-w64-mingw32/lib  -lws2_32 -lpsapi -liphlpapi
tcp_test.exe:  tcp_win_client.c
	x86_64-w64-mingw32-gcc -static -static-libgcc -o $@ $< -I/usr/x86_64-w64-mingw32/include -L/usr/x86_64-w64-mingw32/lib  -lws2_32 -lpsapi -liphlpapi
udp_test.exe:  udp_win_client.c
	x86_64-w64-mingw32-gcc -static -static-libgcc -o $@ $< -I/usr/x86_64-w64-mingw32/include -L/usr/x86_64-w64-mingw32/lib  -lws2_32 -lpsapi -liphlpapi
	#x86_64-w64-mingw32-gcc -static -static-libgcc -o $@ $< -I/usr/x86_64-w64-mingw32/include -L/usr/x86_64-w64-mingw32/lib -luv -lws2_32 -lpsapi -liphlpapi
