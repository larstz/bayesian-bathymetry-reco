#!/bin/bash

# Process all Heat*.txt files
for file in Heat*.txt; do
    if [ -f "$file" ]; then
        sed -i \
            -e 's/\[\(cm\|[[:alpha:]]\)\]//g' \
            -e 's/ \{4,\}/\t/g' \
            -e 's/\t\s*/,/g' \
            -e 's/ //g' \
            -e 's/\r//g' \
            -e 's/,$//g' \
            "$file"
        echo "Processed: $file"
    fi
done

echo "Done!"
