import time
from matplotlib import pyplot as plt
from env import DriftSimEnv

# Track pressed keys using matplotlib events (no admin/root needed)
pressed_keys = set()

def on_key_press(event):
    if event.key:
        pressed_keys.add(event.key)

def on_key_release(event):
    if event.key:
        pressed_keys.discard(event.key)

if __name__ == "__main__":
    env = DriftSimEnv()

    obs = env.reset()
    done = False
    start_time = time.time()
    frame_count = 0

    throttle = 0
    steering = 0

    # Set up interactive plotting and key handlers
    plt.ion()
    fig = plt.figure()
    cid_press = fig.canvas.mpl_connect('key_press_event', on_key_press)
    cid_release = fig.canvas.mpl_connect('key_release_event', on_key_release)

    print("Controls: w=throttle, a=left, d=right. Close window or press Ctrl+C to quit.")

    while not done:

        # Derive controls from currently pressed keys
        throttle = 1 if 'w' in pressed_keys else 0
        if 'a' in pressed_keys and 'd' in pressed_keys:
            steering = 0
        elif 'a' in pressed_keys:
            steering = -1
        elif 'd' in pressed_keys:
            steering = 1
        else:
            steering = 0

        action = [throttle, steering]
        
        obs, reward, done, _ = env.step(action)
        frame_count += 1

        if frame_count % 100 == 0:  # Print every 100 frames
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        frame = env.render_frame()
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.pause(0.001)
        plt.clf()

    env.close()

    # Disconnect event handlers and close figure
    try:
        fig.canvas.mpl_disconnect(cid_press)
        fig.canvas.mpl_disconnect(cid_release)
    except Exception:
        pass

    plt.close()