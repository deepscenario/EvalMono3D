# Format
## Ground Truth Format
### Coordinate System

All 3D annotations are provided in a shared camera coordinate system with 
+x right, +y down, +z toward screen. 

The vertex order of bbox3D_cam:
```
                v4_____________________v5
                /|                    /|
               / |                   / |
              /  |                  /  |
             /___|_________________/   |
          v0|    |                 |v1 |
            |    |                 |   |
            |    |                 |   |
            |    |                 |   |
            |    |_________________|___|
            |   / v7               |   /v6
            |  /                   |  /
            | /                    | /
            |/_____________________|/
            v3                     v2
```

### Annotation Format
Each dataset is formatted as a dict in python in the below format.
Please ensure that category ids are starting at 0 and going up at increments of 1: [0, 1, 2, ...]

```python
ground_truth {
    "info"			    : info,
    "images"			: [image],
    "categories"		: [category],
    "annotations"		: [object],
}

info {
	"id"			: str,
	"source"		: int,
	"name"			: str,
	"split"			: str,
	"version"		: str,
	"url"			: str,
}

image {
	"id"			: int,
	"dataset_id"    : int,
	"width"			: int,
	"height"		: int,
	"file_path"		: str,
	"K"			    : list (3x3),
	"src_90_rotate" : int,					# im was rotated X times, 90 deg counterclockwise 
	"src_flagged"	: bool,					# flagged as potentially inconsistent sky direction
}

category {
	"id"			: int,
	"name"			: str,
	"supercategory"	: str
}

object {
	
	"id"			: int,					# unique annotation identifier
	"image_id"		: int,					# identifier for image
	"category_id"	: int,					# identifier for the category
	"category_name"	: str,					# plain name for the category
	
	# General 2D/3D Box Parameters.
	# Values are set to -1 when unavailable.
	"valid3D"		    : bool,				            # flag for no reliable 3D box
	"bbox2D_tight"		: [x1, y1, x2, y2],			    # 2D corners of annotated tight box
	"bbox2D_proj"		: [x1, y1, x2, y2],			    # 2D corners projected from bbox3D
	"bbox2D_trunc"		: [x1, y1, x2, y2],			    # 2D corners projected from bbox3D then truncated
	"bbox3D_cam"		: [[x1, y1, z1]...[x8, y8, z8]]	# 3D corners in meters and camera coordinates
	"center_cam"		: [x, y, z],				    # 3D center in meters and camera coordinates
	"dimensions"		: [width, height, length],		# 3D attributes for object dimensions in meters
	"R_cam"			    : list (3x3),				    # 3D rotation matrix to the camera frame rotation
	
	# Optional dataset specific properties,
	# used mainly for evaluation and ignore.
	# Values are set to -1 when unavailable.
	"behind_camera"		: bool,					# a corner is behind camera
	"visibility"		: float, 				# annotated visibility 0 to 1
	"truncation"		: float, 				# computed truncation 0 to 1
	"segmentation_pts"	: int, 					# visible instance segmentation points
	"lidar_pts" 		: int, 					# visible LiDAR points in the object
	"depth_error"		: float,				# L1 of depth map and rendered object
}
```

Source: https://github.com/facebookresearch/omni3d/blob/main/DATA.md

## Prediction Format

```python
predictions [{
    "image_id"			: int,
    "K"			        : list (3x3),
    "width"		        : int,
    "height"		    : int,
    "instances"		    : [instance],
}]

instance {
    "image_id"          : int,
    "category_id"       : int,
    "bbox"              : [x1, y1, w, h],
    "score"             : float,
    "depth"             : float,
	"bbox3D"		    : [[x1, y1, z1]...[x8, y8, z8]]	# 3D corners in meters and camera coordinates
	"center_cam"		: [x, y, z],				    # 3D center in meters and camera coordinates
	"center_2D" 		: [xc, yc],
	"dimensions"		: [width, height, length],		# 3D attributes for object dimensions in meters
	"R_cam"			    : list (3x3),				    # 3D rotation matrix to the camera frame rotation
}
