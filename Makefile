# Makefile for cleaning benchmark results and figures
.PHONY: clean clean-results clean-figures help

# Default target
help:
	@echo "Available targets:"
	@echo "  clean         - Clean both results and figures directories"
	@echo "  clean-results - Clean results directory only"
	@echo "  clean-figures - Clean figures directory only"
	@echo "  help          - Show this help message"

# Clean both results and figures directories with confirmation
clean:
	@echo "WARNING: This will delete all files in both results and figures directory!"
	@CODE=$$(cat /dev/urandom | tr -dc 'a-zA-Z' | head -c 4); \
	echo "Please type '$$CODE' to confirm deletion:"; \
	read -r INPUT; \
	if [ "$$INPUT" = "$$CODE" ]; then \
		echo "Confirmation accepted. Proceeding with cleanup..."; \
		$(MAKE) clean-results; \
		$(MAKE) clean-figures; \
		$(MAKE) clean-merged; \
		$(MAKE) clean-e2e; \
	else \
		echo "Confirmation failed. Aborting cleanup."; \
		exit 1; \
	fi

# Clean results directory - only delete files, preserve subdirectories
clean-results:
	@echo "Cleaning results directory..."
	@if [ -d "results" ]; then \
		find results -type f -delete; \
		echo "Results directory cleaned successfully (files removed, subdirectories preserved)"; \
	else \
		echo "Results directory not found"; \
	fi

# Clean figures directory - only delete files, preserve subdirectories
clean-figures:
	@echo "Cleaning figures directory..."
	@if [ -d "figures" ]; then \
		find figures -type f -delete; \
		echo "Figures directory cleaned successfully (files removed, subdirectories preserved)"; \
	else \
		echo "Figures directory not found"; \
	fi

# Clean merged directory - only delete files, preserve subdirectories
clean-merged:
	@echo "Cleaning merged directory..."
	@if [ -d "merged" ]; then \
		find merged -type f -delete; \
		echo "Merged directory cleaned successfully (files removed, subdirectories preserved)"; \
	else \
		echo "Merged directory not found"; \
	fi

# Clean e2e directory - only delete files, preserve subdirectories
clean-e2e:
	@echo "Cleaning e2e directory..."
	@if [ -d "e2e" ]; then \
		find e2e -type f -delete; \
		echo "E2E directory cleaned successfully (files removed, subdirectories preserved)"; \
	else \
		echo "E2E directory not found"; \
	fi