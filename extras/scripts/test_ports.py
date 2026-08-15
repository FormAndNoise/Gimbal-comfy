import urllib.request, json
for p in [8000, 8188]:
    try:
        req = urllib.request.urlopen(f'http://127.0.0.1:{p}/object_info', timeout=2)
        print(f'Port {p}: Wayfinder Loaded=', 'WayfinderGPS_Anchor' in json.loads(req.read().decode('utf-8')))
    except Exception as e:
        print(f'Port {p}: Offline or error ({e})')
