import asyncio
import bpu_infer_lib
import numpy as np
import struct

inf = bpu_infer_lib.Infer(False)
inf.load_model("act_output.bin")

async def handle_client(reader, writer):
    try:
        header = await reader.readexactly(12)
        if len(header) < 12:
            print("Incomplete header received")
            writer.close()
            await writer.wait_closed()
            return

        state_size, top_image_size, images_size = struct.unpack("III", header)
        state_data = await reader.readexactly(state_size)
        top_image_data = await reader.readexactly(top_image_size)
        images_data = await reader.readexactly(images_size)

        state = np.frombuffer(state_data, dtype=np.float32).reshape(1,14)
        images = np.frombuffer(images_data, dtype=np.float32).reshape(1,1,3,480,640)

        inf.read_input(state, 0)
        inf.read_input(images, 1)
        inf.forward(more=True)
        inf.get_output()
        action = inf.outputs[0].data

        writer.write(action.tobytes())
        await writer.drain()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 12345)
    print(f"Server listening on 0.0.0.0:12345")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
