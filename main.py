import time

from matplotlib import pyplot as plt
from env import DriftSimEnv

if __name__ == "__main__":
    env = DriftSimEnv()

    obs = env.reset()
    done = False
    start_time = time.time()
    frame_count = 0

    while not done:
        # Set steering to -1 if frame_count // 100 is even, else 1
        steering = -1 if (frame_count // 100) % 2 == 0 else 1
        action = [1, steering]  # Full throttle with alternating steering
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
    plt.close() 