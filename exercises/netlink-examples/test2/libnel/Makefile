LDFLAGS+=

.PHONY: clean all libs

all: ip iw libs

%.o: %.c
	$(CC) $(CFLAGS) -c -fPIC -o $@ $^

ip: nlcore.o nlroute.o ip.o
	$(CC) $(LDFLAGS) -o $@ $^

iw: nlcore.o nlroute.o genlcore.o nl80211.o iw.o
	$(CC) $(LDFLAGS) -o $@ $^

libnel-route.so: nlcore.o nlroute.o
	$(CC) -o $@ -fPIC -shared $^

libnel-nl80211.so: nlcore.o genlcore.o nl80211.o
	$(CC) -o $@ -fPIC -shared $^

libnel.so: nlcore.o
	$(CC) -o $@ -fPIC -shared $^

libnel-genl.so: nlcore.o genlcore.o
	$(CC) -o $@ -fPIC -shared $^

libs: libnel-route.so libnel-nl80211.so

clean:
	rm ip iw *.o *.a *.so || true
