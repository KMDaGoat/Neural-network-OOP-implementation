Neural-network-OOP-implementation
Understanding the maths behind neural networks and undertsnading the translation between mathematical ideas into clean structured code

Overview:
-model uses make_moons dataset from sklearn.datasets
-model performs forward/back propogation , gradient checking , L2 regularisation and weight initialisation 

Layer:
-Multi layerd neural network
-first layer has 2 nodes representing two x features
-second layer is the first hidden layer and has 5 nodes which the nodes into the first layer acts as inputs to 
-third layer is the second hidden layer and has 4 nodes which the nodes in teh second layer (first hidden layer) output acts as an input to 
-output layer is the last layer and has 1 node in which the third layer (second hidden layer) acts as an input to 
-the output then goes into the cost fucntion to output the cost for that iteration of the neural network
-goal is to get get the minimum cost

Optimisations:
-L2 regularisation used to reduce overfitting 
-input normlaisation to make training data quicker
-gradient checking to make sure back propogation implementation is valid

functions and maths: 
-In forward propogation:  
    -z is gained by combining the weight and the current input of the node (this resembles y = mx + c where a straight line is formed)
    -z then goes under an activation function resulting in an output
    -this output is then used as the next current input for the next layer 
    -for hidden layers the activation function is tanh 
    -for output layers the activation function is sigmoid
    
-In back propogation:
    -used to figure out how much to change w and b by to further get closer to the minimum point by finding the gradient using the derivitive of z and then derivtive of w and b 
    -this is done using the chain rule
    -specifically the derivitve of z is found using the derivtive of A found on the layer after it example: to find the deritive of z on hidden layer 2 you get the derivitive of A calculated in the output layer as input
    -then using derivitve of z you can get derivtive of w and b 

-In gradient checking:
    -gradient is calculated purely using the parameters recorded of w and b 
    -this is done to check if implementaiton of backprop had been succesful , this is why in the code all the derivtives calculated are stored in the attribute grads
    -specifically the gradient is calculated through adding and subtrating epsilon to each value of the parameters and then feeding the dictionary of value into cost function to get the cost.
    -getitng the cost is important for iteration through parameter as we are going to compare the value returned using gradient checking on the parameter dictionary and the actual gradient calculated using the grads vector 
    -this is done by dividing size of (vector holding grads - vector holding approxgrad (storees costs based on parameters) with (size of actual gradient vector ) + (size of approxgrad vector)
    -this would return a very small number in which if the number returned is >1e-07 then soemthing is wrong and if it is less then back propogation had been implemented properly
