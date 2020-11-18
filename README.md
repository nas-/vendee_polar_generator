# vendee_polar_generator

Pulls data from here https://exocet.cloud/grafana/d/bsbc_5MGz/malizia-public-dashboard?orgId=15&from=now-120h&to=now&theme=dark to generate polar charts of the speed of the boat with varius windspeeds
It will pull the data max once every hour, and it will store it on disk and use that up to the next update

example result
![alt text](https://raw.githubusercontent.com/nas-/vendee_polar_generator/master/Polars.png "polars")

The speed of the boat for each TWA is also output as an excel file
