set -e
cd /tmp/QuantLib

# Check and apply each patch only if needed
for patch in /workspace/patches/*.patch; do
  if [ -f "$patch" ]; then
    # Check if patch is already applied
    if patch -p1 -R --dry-run -s < "$patch" > /dev/null 2>&1; then
      echo "SKIP: $(basename $patch) already applied"
    else
      # Check if patch can be applied (not already applied)
      if patch -p1 --dry-run -s < "$patch" > /dev/null 2>&1; then
        echo "APPLY: $(basename $patch)"
        patch -p1 < "$patch"
      else
        echo "ERROR: $(basename $patch) cannot be applied (conflicts?)"
        exit 1
      fi
    fi
  fi
done
echo "Patches check complete"