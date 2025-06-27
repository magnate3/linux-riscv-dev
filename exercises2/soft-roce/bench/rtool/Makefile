CFLAGS ?= -g -ftree-vectorize -Wformat=0
CFLAGS += -pedantic -std=gnu99 -Wall -Wextra
LIBS=-L/usr/local/lib -lm -lrt -libverbs -lhugetlbfs -lrdmacm -lpthread -lcap -lmlx5
PREFIX=/usr/local
BINDIR=$(PREFIX)/bin
MAN1DIR=$(PREFIX)/share/man/man1

MR_LAT_SRCS=resource_lat.c options.c ioctl.c
MR_LAT_OBJS=$(MR_LAT_SRCS:.c=.o)
MR_LAT_BINARY=rdma_resource_lat
RDMA_IO_SRCS=rdmaio.c options.c
RDMA_IO_OBJS=$(RDMA_IO_SRCS:.c=.o)
RDMA_IO_BINARY=rdmaio

MANS=rdma_resource_lat.1 rdmaio.1
MANS_F=$(MANS:.1=.txt) $(MANS:.1=.pdf)
DOCS=README.md LICENSE changelog
SPEC=rdma_resource_lat.spec rdmaio.spec

PACKAGE=rdma_resource_lat
GIT_VER:=$(shell test -d .git && git describe --tags --match 'v[0-9]*' \
		--abbrev=0 | sed 's/v//')
SRC_VER:=$(shell sed -ne 's/\#define VERSION \"\(.*\)\"/\1/p' resource_lat.c)
VERSION:=$(SRC_VER)
DISTDIR=$(PACKAGE)-$(VERSION)
DISTFILES=$(MR_LAT_SRCS) $(MANS) $(DOCS) $(SPEC) Makefile
PACKFILES=$(MR_LAT_BINARY) $(MANS) $(MANS_F) $(DOCS)

STRIP=strip
TARGET=$(shell ${CC} -dumpmachine)

all: checkver $(MR_LAT_BINARY) $(RDMA_IO_BINARY)

version: checkver
	@echo ${VERSION}

checkver:
	@if test -n "$(GIT_VER)" -a "$(GIT_VER)" != "$(SRC_VER)"; then \
		echo "ERROR: Version mismatch between git and source"; \
		echo git: $(GIT_VER), src: $(SRC_VER); \
		exit 1; \
	fi

clean:
	$(RM) -f $(MR_LAT_OBJS) $(MR_LAT_BINARY) $(MANS_F) rdma_resource_lat.tmp
	$(RM) -f $(RDMA_IO_OBJS) $(RDMA_IO_BINARY) $(MANS_F) rdmaio_lat.tmp

strip: $(MR_LAT_BINARY) $(RDMA_IO_BINARY)
	$(STRIP) $^

install: $(MR_LAT_BINARY) $(RDMA_IO_BINARY) $(MANS)
	mkdir -p $(DESTDIR)$(BINDIR)
	install -m 0755 $(MR_LAT_BINARY) $(DESTDIR)$(BINDIR)
	install -m 0755 $(RDMA_IO_BINARY) $(DESTDIR)$(BINDIR)
	mkdir -p $(DESTDIR)$(MAN1DIR)
	install -m 644 $(MANS) $(DESTDIR)$(MAN1DIR)

%.o: %.c %.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.ps: %.1
	man -t ./$< > $@

%.pdf: %.ps
	ps2pdf $< $@

%.txt: %.1
	MANWIDTH=80 man ./$< | col -b > $@

$(MR_LAT_BINARY): $(MR_LAT_OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(RDMA_IO_BINARY): $(RDMA_IO_OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

dist: checkver $(DISTFILES)
	tar -cz --transform='s,^,$(DISTDIR)/,S' $^ -f $(DISTDIR).tar.gz

binary-tgz: checkver $(PACKFILES)
	${STRIP} ${MR_LAT_BINARY}
	${STRIP} ${RDMA_IO_BINARY}
	tar -cz --transform='s,^,$(DISTDIR)/,S' -f ${PACKAGE}-${VERSION}-${TARGET}.tgz $^

binary-zip: checkver $(PACKFILES)
	${STRIP} ${MR_LAT_BINARY}
	${STRIP} ${RDMA_IO_BINARY}
	ln -s . $(DISTDIR)
	zip ${PACKAGE}-${VERSION}-${TARGET}.zip $(addprefix $(DISTDIR)/,$^)
	rm $(DISTDIR)

.PHONY: all version checkver clean strip test install dist binary-tgz binary-zip
