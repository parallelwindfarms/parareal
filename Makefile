.PHONY: site clean watch watch-pandoc watch-browser-sync
ENV += PDOC_ALLOW_EXEC=1

serve:
	$(ENV) poetry run pdoc -t themes/dark-mode --math parareal

docs:
	$(ENV) poetry run pdoc -t themes/dark-mode --math -o docs/ parareal

