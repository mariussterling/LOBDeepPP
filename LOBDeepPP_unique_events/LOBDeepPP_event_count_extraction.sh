#!/bin/bash
shopt -s extglob
shopt extglob

gzip -vd AMZN_2015-07-??_34200000_57600000_orderbook_200.csv.gz
for ((ol=200; ol>=1; ol--))
do
	echo $ol
	for ff in $( ls -1 AMZN_2015-07-??_34200000_57600000_orderbook_200.csv )
	do
		tmp=$(cut -d ',' -f 1-$((4*ol)) $ff | awk -F "," '$0==last{next} {last=$0} {print $0}' | wc)
		echo "$tmp $ff $ol" >> LOB_events_count.csv
	done
done

rm AMZN_2015-07-??_34200000_57600000_orderbook_200.csv
