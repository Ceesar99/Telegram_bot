import os
import json
from universal_trading_launcher import SystemValidator

def main():
	validator = SystemValidator()
	results = validator.run_full_validation()
	print(json.dumps(results, indent=2))
	passed = sum(results.values())
	total = len(results)
	if passed < total * 0.8:
		raise SystemExit(1)

if __name__ == "__main__":
	main()

