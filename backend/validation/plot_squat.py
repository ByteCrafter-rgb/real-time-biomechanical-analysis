import csv
import os

import matplotlib.pyplot as plt


CSV_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "validation",
        "output",
        "squat_knee_angles.csv",
    )
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "validation",
        "output",
        "squat_knee_angle_plot.png",
    )
)


def main():
    times = []
    left_angles = []
    right_angles = []

    with open(CSV_PATH, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            times.append(float(row["time_seconds"]))

            left = row["left_knee"]
            right = row["right_knee"]

            left_angles.append(
                float(left) if left else None
            )

            right_angles.append(
                float(right) if right else None
            )

    plt.figure(figsize=(12, 6))

    plt.plot(
        times,
        left_angles,
        label="Left Knee",
    )

    plt.plot(
        times,
        right_angles,
        label="Right Knee",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Knee Flexion (degrees)")
    plt.title("Knee Flexion During Squat")

    plt.ylim(0, 180)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
    )

    print("Plot saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()