CH2_final_evaluation.csv contains the ground truth labels for testing images.

turn left = positive steering angles
turn right = negative steering angles

final_example.csv contains false randomly generated labels.

The gt labels are in radians, turn it into degress:
mae_error = mae_error / 1.0 * 180 / 3.1415
rmse_error = np.sqrt(rmse_error / 1.0) * 180 / 3.1415