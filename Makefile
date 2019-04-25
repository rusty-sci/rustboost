CA = cargo

run:
	RUST_LOG=rustboost $(CA) run ./tests/data/generated/regression.libsvm -o reg:tree

testlog:
	$(CA) test -- --nocapture

test:
	$(CA) test