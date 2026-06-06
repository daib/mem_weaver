FROM rust:1.87-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin mem-weaver-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/mem-weaver-server /usr/local/bin/mem-weaver-server

EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/mem-weaver-server"]
