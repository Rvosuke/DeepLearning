# Speech Signal Processing and Pitch Detection

This repository contains Python code for speech signal processing, pitch detection, and post-processing techniques. The goal of this project is to accurately detect the fundamental frequency (pitch) of speech signals and apply various smoothing techniques to enhance the pitch contour.

## Table of Contents

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
- [Usage](#usage)
  - [Pitch Detection](#pitch-detection)
  - [Post-Processing](#post-processing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Speech signal processing is a crucial aspect of many applications, including speech recognition, speaker identification, and speech synthesis. This project focuses on the task of pitch detection, which involves estimating the fundamental frequency of a speech signal over time. Additionally, various post-processing techniques are implemented to smooth the pitch contour and mitigate the effects of noise and outliers.

## Dependencies

The following dependencies are required to run the code in this repository:

- Python 3.x
- NumPy
- Matplotlib

## Code Structure

The repository is organized as follows:

- `recurrent_nn.py`: Contains functions for defining and initializing the parameters of a recurrent neural network used for pitch detection.
- `masked_softmax.py`: Implements the masked softmax operation, which is useful for handling variable-length sequences in the pitch detection model.
- `pitch_detection_post_processing.py`: Includes various functions for post-processing the pitch contour, such as median smoothing, linear smoothing, combined smoothing, and quadratic combined smoothing.

## Usage

### Pitch Detection

The `recurrent_nn.py` and `masked_softmax.py` files provide the necessary functions for building and training a recurrent neural network model for pitch detection. You can use these functions to preprocess your speech data, define and train your model, and obtain pitch predictions.

### Post-Processing

The `pitch_detection_post_processing.py` file contains several functions for post-processing the pitch contour obtained from the pitch detection model. These functions include:

1. `wild_points(pitch, p=0.1)`: Introduces random outliers in the pitch contour by multiplying or dividing the pitch values by a factor of 2.
2. `add_gaussian_noise(pitch, sigma=5)`: Adds Gaussian noise to the pitch contour.
3. `median_smoothing(pitch, window_size=5)`: Applies median smoothing to the pitch contour using a sliding window.
4. `linear_smoothing(pitch, L)`: Performs linear smoothing on the pitch contour using a window function of length `L`.
5. `window_function(L)`: Defines the window function used for linear smoothing.
6. `delay_signal(x, D)`: Introduces a delay of `D` samples in the signal `x`.
7. `combined_smoothing1(pitch)`: Applies a combination of median smoothing with window sizes 5 and 3.
8. `combined_smoothing2(pitch)`: Applies a combination of median smoothing (window size 5) and linear smoothing (window size 3).
9. `quadratic_combined_smoothing1(pitch)`: Applies a quadratic combination of median smoothing and linear smoothing.
10. `quadratic_combined_smoothing2(pitch)`: Applies a quadratic combination of median smoothing and linear smoothing with a delay component.

You can use these functions to experiment with different post-processing techniques and evaluate their effectiveness on your pitch detection results.

## Results

The results of the pitch detection and post-processing techniques can be visualized using the `matplotlib` library. The `pitch_detection_post_processing.py` file includes an example of plotting the original pitch contour and the smoothed pitch contour after applying median smoothing.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).