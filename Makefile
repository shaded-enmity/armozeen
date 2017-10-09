.PHONY: clean

clean:
	@find . -name '*.pyc' -type f -not -wholename './.git*' -delete
	@find . -name '*.sw[nop]' -type f -not -wholename './.git*' -delete
	@echo Cleaned up
