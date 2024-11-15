import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def get_metrics(file):
    with open(file, 'rb') as f:
        metrics, acl, pl, acs, ps = pickle.load(f) # actual and predicted speeds and locations

    frame_count = len(acs)
    print(f"Total frames = {frame_count}")
    actual_speeds = np.full(frame_count, np.nan)
    for k, v in acs.items():
        actual_speeds[k-1] = v 

    predicted_speeds = np.full(frame_count, np.nan)
    for k, v in ps.items():
        predicted_speeds[k-1] = v 

    actual_locations = []
    for k, v in acl.items():
        actual_locations.append(v) 

    predicted_locations = np.full(frame_count, np.nan)
    #for k, v in pl.items():
    #    predicted_locations[k] = v 

    fd_any = metrics['fd_any']
    fd_car = metrics['fd_car']
    fd_car_valid = metrics['fd_car_valid']

    return {'frames':frame_count, 'positions': actual_locations, 'actual':actual_speeds, 'predicted':predicted_speeds, 'fd':fd_any, 'fd_car':fd_car, 'vfd_car': fd_car_valid}

def plot_metrics(frame_count, actual_speeds, predicted_speeds, fd_any, fd_car, fd_car_valid):

    frame_range = np.arange(frame_count)
    # Calculate the difference between the two arrays
    difference = np.abs(predicted_speeds - actual_speeds)

    # Plot the two lines
    plt.plot(frame_range, actual_speeds, label='Actual Speed', linestyle='-', color='blue')
    plt.plot(frame_range, predicted_speeds, label='Predicted Speed', linestyle='--', color='orange')

    # Add shaded area showing the difference as error
    plt.fill_between(frame_range, actual_speeds, predicted_speeds, color='gray', alpha=0.1, label='Error')
    plt.axvline(x=fd_any['frame'], color='red', linestyle=':', label=f"First detection, class={fd_any['class']}, conf={fd_any['conf']}")
    plt.axvline(x=fd_car['frame'], color='yellow', linestyle=':', label=f"First car detection, conf={fd_car['conf']}")
    plt.axvline(x=fd_car_valid['frame'], color='green', linestyle=':', label=f"First valid car detection, conf={fd_car_valid['conf']}")

    # Labels and title
    plt.xlabel('Number of Frames')
    plt.ylabel('Speed')
    plt.title('Speed')
    plt.legend()

    # Show plot
    plt.show()


def plot_in_3d():
    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two sets of points
    ax.plot(acl[:, 0], acl[:, 1], acl[:, 2], label='Actual location', color='blue', marker='o')
    ax.plot(pl[:, 0], pl[:, 1], pl[:, 2], label='Predicted location', color='Orange', marker='^')

    # Plot error lines between corresponding points
    for p1, p2 in zip(acl, pl):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linestyle='--', alpha=0.5)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Comparison of Two Arrays with Error Lines')
    ax.legend()

    plt.show()

def f2d(f):
    return 98.8200712934672 - 0.8090149861777975 * f

def main():
    cloud = get_metrics('cloud_data_dump.pkl')
    edge = get_metrics('edge_data_dump.pkl')
    rpi = get_metrics('rpi_data_dump.pkl')

    min_val = min(cloud['actual'][1], edge['actual'][1], rpi['actual'][1])
    #max_val = max(cloud['actual'][-1], edge['actual'][-1], rpi['actual'][-1])

    cloud_start, edge_start, rpi_start = 1, 1, 1
    cloud_start = np.abs(cloud['actual'] - min_val).argmin()
    edge_start = np.abs(edge['actual'] - min_val).argmin()
    rpi_start = np.abs(rpi['actual'] - min_val).argmin()

    frame_count = min(len(cloud['actual']), len(edge['actual']), len(rpi['actual']))

    print(frame_count)
    print(cloud_start, edge_start, rpi_start)

    cloud['actual'] = cloud['actual'][cloud_start:cloud_start+frame_count-1]
    cloud['predicted'] = cloud['predicted'][cloud_start:cloud_start+frame_count-1]
    cloud['positions'] = cloud['positions'][cloud_start:cloud_start+frame_count-1]
    
    edge['actual'] = edge['actual'][edge_start:edge_start+frame_count-1]
    edge['predicted'] = edge['predicted'][edge_start:edge_start+frame_count-1]
    edge['positions'] = edge['positions'][edge_start:edge_start+frame_count-1]
    
    rpi['actual'] = rpi['actual'][rpi_start:rpi_start+frame_count-1]
    rpi['predicted'] = rpi['predicted'][rpi_start:rpi_start+frame_count-1]
    rpi['positions'] = rpi['positions'][rpi_start:rpi_start+frame_count-1]

    print(cloud['actual'][0], cloud['actual'][-1]) #, cloud['actual'][cloud_frames[0]], cloud['actual'][cloud_frames[-1]])
    print(edge['actual'][0], edge['actual'][-1]) #, edge['actual'][edge_frames[0]], edge['actual'][edge_frames[-1]])
    print(rpi['actual'][0], rpi['actual'][-1]) #, rpi['actual'][rpi_frames[0]], rpi['actual'][rpi_frames[-1]])

    error_cloud = np.abs(np.where(np.isnan(cloud['predicted']), np.nan, cloud['predicted'] - cloud['actual']))
    error_edge = np.abs(np.where(np.isnan(edge['predicted']), np.nan, edge['predicted'] - edge['actual']))
    error_rpi = np.abs(np.where(np.isnan(rpi['predicted']), np.nan, rpi['predicted'] - rpi['actual']))

    cloud['predicted'] = cloud['actual'] + error_cloud
    edge['predicted'] = cloud['actual'] + error_edge
    rpi['predicted'] = cloud['actual'] + error_rpi

    edge['predicted'][-3:] = np.nan
    rpi['predicted'][-3:] = np.nan
    cloud['actual'][-5:] = np.nan

    print(cloud['predicted'][80:])
    print(edge['predicted'][80:])
    print(rpi['predicted'][80:])
    
    cloud_smooth = moving_average(cloud['predicted'],5)
    edge_smooth = moving_average(edge['predicted'],5)
    rpi_smooth = moving_average(rpi['predicted'],5)

    frame_range = np.arange(frame_count-1)
    distances = [200 - x[0] for x in cloud['positions']]
    plt.plot(distances, cloud['actual'], label='Actual Speed', linestyle='-', color='blue')
    plt.plot(distances, cloud_smooth, label='Cloud Speed', linestyle='--', color='green')
    plt.plot(distances, edge_smooth, label='Edge Speed', linestyle='--', color='orange')
    plt.plot(distances, rpi_smooth, label='Rpi Speed', linestyle='--', color='red')

    plt.axvline(x=distances[cloud['vfd_car']['frame']-cloud_start+1], color='green', linestyle=':', label=f"first cloud detection")
    plt.axvline(x=distances[edge['vfd_car']['frame']-edge_start+1], color='orange', linestyle=':', label=f"first edge detection")
    plt.axvline(x=distances[rpi['vfd_car']['frame']-rpi_start+1], color='red', linestyle=':', label=f"first rpi detection")

    # Labels and title
    plt.xlabel('Distance from Camera (m)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.xlim(100, 0)

    # Show plot
    plt.show()

    #plot_metrics(frame_count, cloud['actual'], cloud['predicted'], cloud['fd'], cloud['fd_car'], cloud['vfd_car'])

if __name__=='__main__':
    main()
