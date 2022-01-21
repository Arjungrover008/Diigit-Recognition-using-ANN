#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <graphics.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS 1000
#define BATCH_SIZE 100

// scales value according to x-axis
#define SCALE_X(x) ((((int) (x)) / 2) + 80)

// scales value according to y-axis
#define SCALE_Y(y) (400 - (((float) (y)) * 300))

/**
 * mnist data set files for training and testing
 */
const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

/**
 * Makes a graph with the proper labels over X and Y-axis
 */
void setGrid_Labels()
{
	// setting x-axis
	setcolor(BLUE);
	line(80, 400, 580, 400);

	// setting y-axis 
	setcolor(BLUE);
	line(80, 400, 80, 100);

	// setting grids
	setcolor(RED);
	line(81, SCALE_Y(0.1), 580, SCALE_Y(0.1));
	line(81, SCALE_Y(0.2), 580, SCALE_Y(0.2));
	line(81, SCALE_Y(0.3), 580, SCALE_Y(0.3));
	line(81, SCALE_Y(0.4), 580, SCALE_Y(0.4));
	line(81, SCALE_Y(0.5), 580, SCALE_Y(0.5));
	line(81, SCALE_Y(0.6), 580, SCALE_Y(0.6));
	line(81, SCALE_Y(0.7), 580, SCALE_Y(0.7));
	line(81, SCALE_Y(0.8), 580, SCALE_Y(0.8));
	line(81, SCALE_Y(0.9), 580, SCALE_Y(0.9));
	line(81, SCALE_Y(1.0), 580, SCALE_Y(1.0));

	line(SCALE_X(100), 400, SCALE_X(100), 100);
	line(SCALE_X(200), 400, SCALE_X(200), 100);
	line(SCALE_X(300), 400, SCALE_X(300), 100);
	line(SCALE_X(400), 400, SCALE_X(400), 100);
	line(SCALE_X(500), 400, SCALE_X(500), 100);
	line(SCALE_X(600), 400, SCALE_X(600), 100);
	line(SCALE_X(700), 400, SCALE_X(700), 100);
	line(SCALE_X(800), 400, SCALE_X(800), 100);
	line(SCALE_X(900), 400, SCALE_X(900), 100);
	line(SCALE_X(1000), 400, SCALE_X(1000), 100);

	// moving to the graph's origin (80, 400)
	moveto(80, 400);

	// labelling the x-y axis
	outtextxy(270, 435, "NUMBER OF STEPS");
	outtextxy(35, 70, "ACCURACY (in %)");

	// unit labelling on X-axis
	outtextxy(SCALE_X(100)-13, 407, "100");
	outtextxy(SCALE_X(200)-13, 407, "200");
	outtextxy(SCALE_X(300)-13, 407, "300");
	outtextxy(SCALE_X(400)-13, 407, "400");
	outtextxy(SCALE_X(500)-13, 407, "500");
	outtextxy(SCALE_X(600)-13, 407, "600");
	outtextxy(SCALE_X(700)-13, 407, "700");
	outtextxy(SCALE_X(800)-13, 407, "800");
	outtextxy(SCALE_X(900)-13, 407, "900");
	outtextxy(SCALE_X(1000)-15, 407, "1000");

	// unit labelling on Y-axis
	outtextxy(75, 405, "0");
	outtextxy(56, SCALE_Y(0.1)-4, "10");
	outtextxy(56, SCALE_Y(0.2)-4, "20");
	outtextxy(56, SCALE_Y(0.3)-4, "30");
	outtextxy(56, SCALE_Y(0.4)-4, "40");
	outtextxy(56, SCALE_Y(0.5)-4, "50");
	outtextxy(56, SCALE_Y(0.6)-4, "60");
	outtextxy(56, SCALE_Y(0.7)-4, "70");
	outtextxy(56, SCALE_Y(0.8)-4, "80");
	outtextxy(56, SCALE_Y(0.9)-4, "90");
	outtextxy(52, SCALE_Y(1.0)-4, "100");
}

int main(int argc, char *argv[])
{
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;

    // setting graphics.h
    int gd = DETECT, gm; 
	initgraph(&gd, &gm, NULL); 


    // makes the graph and its labels
	setGrid_Labels();

	// setting color for the graph's line
	setcolor(GREEN);

	float acc[STEPS];
	int stp[STEPS];

    for (i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, 100, i % batches);

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(&batch, &network, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(test_dataset, &network);

        //printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / batch.size, accuracy);

        lineto(SCALE_X(i), SCALE_Y(accuracy));

        acc[i] = accuracy;
        stp[i] = i;
    }

    float maxA;
    int maxS;

    for(i = 0; i < STEPS - 1; i++){
    	maxA = acc[i];
    	maxS = stp[i];
    	for (int j = 0; j < STEPS; j++)
    	{
    		if(acc[j] > maxA) {
    			maxA = acc[j];
    			maxS = stp[j];
    		}

    	}
    }

    setcolor(YELLOW);

    char str[20];

    line(SCALE_X(maxS), SCALE_Y(maxA), 400, 40);
    sprintf(str, "MAX ACCURACY = %.2f", maxA*100);
    outtextxy(330, 22, str);

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
    delay(500000); 
    closegraph(); 

    return 0;
}
