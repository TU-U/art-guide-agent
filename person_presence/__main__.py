import uvicorn

uvicorn.run("person_presence.app:app", host="127.0.0.1", port=8090, reload=False)
