use make to generate version

`make version`

let see the exact command

`COMMIT_HASH=$(git log -1 --pretty=format:"%h" | tail -1) && echo "VERSION = "0.17.0-${COMMIT_HASH}"" > frigate/version.py`

`cat frigate/version.py`

build and run frontend

`cd web 
npm install
npm run build
`

if get permission in build forntend

`cd ~/Projects/stable-forked/frigate
sudo chown -R $USER:$USER web
`

to use the app

`docker exec frigate-devcontainer bash -c "cd /workspace/frigate && /usr/bin/python3 -m frigate"`

use it for stop it
`docker exec frigate-devcontainer bash -c "pkill -f 'python3 -m frigate' || true"`

sample .env for frontend

VITE_GIT_COMMIT_HASH=e1545a8d
PROXY_HOST=127.0.0.1:8971
