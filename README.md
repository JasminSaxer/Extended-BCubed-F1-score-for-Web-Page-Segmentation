# Extended BCubed F1-score for Web Page Segmentation

This project is a modified version of the [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset). The original code has been adapted to Python and tailored for specific use.

## Changes from [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset)
The extended BCubed F1-score calculation in [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset) takes in only bounding Boxes and matches these to the HTML nodes by their bounding Boxes. 

This project takes as input for each segment: 
- annotations as polygons (same as previous)
- annotations as hyu values (hyu values are specific for each node, and added by using this [content extraction software](https://github.com/JasminSaxer/content-extraction-framework).)

Thus the BCubed F1-score which are pixel-based (pixel, edges_fine, edges_coarse) are calculated using the polygons, 
and the node-based (node, characters) are calculated using the hyu values (speific nodes).

## Description

- main.py: The main script that processes the folders in the data directory, evaluates the segmentations, and prints the results.
- src: Contains the source code for various functionalities:
    - bcubed.py: Implements the B-Cubed evaluation metrics.
    - canny.py: Contains functions for applying Canny edge detection.
    - classes.py: Defines the main classes used in the project, such as Task, Segmentation, and Segmentations.
    - node_clustering.py: Implements node-based clustering.
    - pixel_clustering.py: Implements pixel-based clustering.

The evaluation results are saved in bcubed_res.csv files within each test data folder. The results include B-Cubed F1 and Max scores for each page.

## Installation

Install the required dependencies (using python version 3.11.0):
```bash
pip install -r requirements.txt
```

## Usage


```
usage: main.py [-h] [--folder_path FOLDER_PATH] [--class_system {FC,MC}] [--operation {prediction,pairwise_agreement}] [--per_class]
               [--measure {F1,MaxPR,Precision,Recall}] [--use_pool] [--processes PROCESSES] [--verbose]

options:
  -h, --help            show this help message and exit
  --folder_path FOLDER_PATH
                        The path to the folder containing the data
  --class_system {FC,MC}
                        The class system to use (FC or MC)
  --operation {prediction,pairwise_agreement}
                        The operation to perform (prediction or pairwise_agreement)
  --per_class           Calculate score per class
  --measure {F1,MaxPR,Precision,Recall}
                        The measure to use for per class calculation (default: F1). Can use F1, MaxPR and Recall or Precision for predicition only.
  --use_pool            Use multiprocessing pool to process folders
  --processes PROCESSES
                        Number of processes to use in the pool (default: 4)
  --verbose             Increase output verbosity
```

For example: 
```bash
python main.py --folder_path test_data_pairwise --class_system MC --operation pairwise_agreement
python main.py --folder_path test_data_prediction --class_system FC --operation prediction --per_class --measure F1
```

## Data Structure

The data directory should have the following structure:

```
data/
├── page1/
│   ├── annotations_FC.json
│   ├── annotations_MC.json
│   ├── dom.html
│   └── screenshot.png
├── page2/
│   ├── annotations_FC.json
│   ├── annotations_MC.json
│   ├── dom.html
│   └── screenshot.png
└── ...
```

Each page should have its own directory containing:
- `annotations_MC.json`: Contains the segmentation annotations for the page for the specific classification system.
- `dom.html`: The HTML DOM of the web page (with hyu Indexes), extracted from the MHTML generated using [content extraction software](https://github.com/JasminSaxer/content-extraction-framework).
- `screenshot.png`: A screenshot of the web page for visual reference.

### annotations.json

The `annotations.json` file contains the segmentation annotations for a specific web page. It includes the following fields:

- `id`: A unique identifier for the web page.
- `height`: The height of the web page screenshot.
- `width`: The width of the web page screenshot.
- `segmentations`: A dictionary where each key is an annotator's identifier and the value is a list of segmentation coordinates. Each segmentation is represented as a list of polygons, where each polygon is a list of points (x, y coordinates).
- `nodes`: A dictionary where each key is an annotator's identifier and the value is a dictionary with 'hyuIndex': index value, corresponding to the dom.html.

Example structure:
```json
{
  "id": "66ffde79306dfe2088fd9ff7",
  "height": 3716,
  "width": 2560,
  "segmentations": {
    "annotator1": [
      [
        [
          [480, 0],
          [2080, 0],
          [2080, 221],
          [480, 221],
          [480, 0]
        ]
      ],
      ...
    ] 
  },
  "nodes": {
    "annotator1": [
      {
        "tagType": "header",
        "hyuIndex": 994
      },
      ...
    ]
  }
}
```

## License

This project is licensed under the terms of the original repository.

## Acknowledgments

- The original code are from the [CIKM 2020 Web Page Segmentation Revisited Evaluation Framework and Dataset](https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset).