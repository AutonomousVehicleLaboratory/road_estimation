Road estimation from point cloud

NOTE: data for point cloud is not uploaded since it is large.

- [x] Get data from YOLO and our vehicle
- [ ] check the synchronization
- [ ] test with real data
- [ ] Calibrate the lidar and camera
- [ ] combine two planes
- [X] RANSAC based implementation of plane estimation tested.
- [X] Implemented bin statistics, this can be used to evaluate the performance.
And also form the basis of adaptive modeling.

1. As said, the hyper parameter for ransac is very important.
    - A very tight threshold might leads to fitting to a small flat patch. Even with 5000 samples times, the model is biased to one side of the road, which will not fit the other side.
2. the reweight of the points are important.
    - with weighting by distance, samples yeilds good result that covers the main road mostly.
    - without weightinhg by distance, samples tends to performan bad for close range points on the side of the road (the concave part), even with 500 hundred samples
3. sample number
    - with reweighting, with sample time as 10, out of 10 trials, 5 yeilds similar result as 50 samples, 3 gives slightly less performance and 2 gives performance similar to without reweighting. When sample time goes to 50, almost perfect always.

Concerning issue
1. synchronization
2. c++ library for efficient implementation
3. What a flat plane look like？

Mark examples
- 20-25 side
- 33-36 far
- 41 side
- 46 side
- 55 17-22 0
- 57 skew road
- 5 flat road

ROS
- add #!/usr/bin/env python at the beginning of the file
- chmod +x to make it executable (everytime you modify it)
- rosrun road_estimation road_estimation
- rviz add panel, image viewer, then you can add bounding box to it.
