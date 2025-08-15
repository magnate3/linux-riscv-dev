FROM ghcr.io/stargz-containers/ubuntu:20.04-org AS build1
RUN echo hello > /hello

FROM ghcr.io/stargz-containers/ubuntu:20.04-org AS build2
RUN echo hi > /hi

FROM ghcr.io/stargz-containers/ubuntu:20.04-org
COPY --from=build1 /hello /
COPY --from=build2 /hi /
RUN cat /hello /hi
