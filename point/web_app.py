import asyncio
import threading

import cv2
import rclpy
from aiohttp import web

import state as st
from node import HandDetectNode


async def video_feed(request):
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame"
        }
    )

    await response.prepare(request)

    while True:
        if st.latest_frame is not None:
            ret, buffer = cv2.imencode(
                ".jpg",
                st.latest_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            if ret:
                frame_bytes = buffer.tobytes()

                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

        await asyncio.sleep(0.03)


async def pointing_result(request):
    return web.json_response(st.latest_pointing_result)


async def index(request):
    html = """
    <html>
        <head>
            <title>Hand YOLO 3D Ray Stream</title>
        </head>
        <body>
            <h2>Hand + YOLO + Smoothed 3D Ray Stream</h2>
            <p>Blue boxes: YOLO objects</p>
            <p>Red line: smoothed 3D finger ray</p>
            <p>Red box: first object hit by 3D ray</p>
            <img src="/video" width="960">
        </body>
    </html>
    """

    return web.Response(
        text=html,
        content_type="text/html"
    )


def ros_thread():
    rclpy.init()
    node = HandDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    t = threading.Thread(target=ros_thread)
    t.daemon = True
    t.start()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/video", video_feed)
    app.router.add_get("/pointing_result", pointing_result)

    web.run_app(
        app,
        host="0.0.0.0",
        port=8081
    )
