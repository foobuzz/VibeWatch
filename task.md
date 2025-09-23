In the folder `~/Data/smart_watch` are multiple subfolders identified by dates (or partial dates). Each of those is an entire copy of the filesystem of my Garmin watch -- what I'm calling a "snapshot".

Your task is to build a static site generator taking in input a collection of those "snapshots" (basically, the `~/Data/smart_watch` directory), and creating a single HTML file being a dashboard of the data.

## Dashboard details

The dashboard will consists in charts showing data over time, for the following dimensions:

 - Average sleep over time
 - Average stress level over time
 - Average BPM over time

For each chart, the user is able to select a specific time frame, as well as zoom/unzoom over the time-period of what a "dot" is:

 - Month
 - Week
 - Day

## Implementation details

There are 2 programs to build:

 1. The static site generator itself, which will consist in producing a compact JSON file containing all the data exploitable by the "front-end". => You must use Python for this part.
 2. The "front-end", which is a single HTML page containing JavaScript that loads the compact JSON file, and dynamically uses the data inside to power the charts controlled by the user. => You must use native JavaScript, embedded in the HTML page, for this part.

The static site generator should also write to stdout any missing timeframe in the raw folder, if any (since the snapshots might not have been backuped often enough).
