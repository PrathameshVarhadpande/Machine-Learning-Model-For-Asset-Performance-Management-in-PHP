<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Hamming;
use Rubix\ML\CrossValidation\Metrics\Accuracy;


//Loading Dataset

$dataset=Labeled::fromIterator(new CSV('heat_efficiency.csv'));

// Splitting the training and testing dataset in the 80:20 ratio

[$training,$testing]=$dataset->stratifiedSplit(0.8);

// Applying KNN Algorithm using the Hamming Distance Kernels

$estimator = new KNearestNeighbors(3, true, new Hamming(8));

// Training the ML Model on the training dataset

$estimator->train($training);

// Testing/Making predictions on the test dataset

$predictions = $estimator->predict($testing);

// Storing the predicted values in a form of an array

$output = array_slice($predictions, 0, 5);

// Printing the predicted value

echo '<br>Efficiency of the Machine : ' . ($output[0]);

// Defining the Accuracy of the model

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo '<br>Accuracy of the model is ' . (string) ($score * 100.0) . '%' . PHP_EOL;
?>